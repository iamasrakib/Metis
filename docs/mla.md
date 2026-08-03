# Multi-head Latent Attention (MLA)

A from-scratch implementation of the attention mechanism introduced by
DeepSeek-V2 (arXiv:2405.04434) and carried into DeepSeek-V3, simplified for a
tiny model: **no query-side latent** (`c_t^Q`), a single latent shared across
all heads, per-head key/value up-projections, and RoPE applied only to a
sliced key part.

`kv_backend="mla"` swaps the attention module in every
`TransformerBlock` for `MLAAttention` (`metis/mla.py`).

---

## 1. Why MLA?

Standard MHA stores two full tensors per token per layer in the KV cache:
`2 * n_heads * head_dim` values. GQA shrinks this by *sharing whole KV heads*
across a query group (`2 * n_kv_heads * head_dim`) — a coarse compression that
limits how much information a group can retain.

MLA instead compresses the KV state into a **low-rank latent vector** `c_t`
shared across all heads, plus the RoPE part of the key:

```
c_t    = W_DKV · h_t                  # (c_d,)     shared latent
k_t^C  = W_UK · c_t                   # (head_dim) content part (not rotated)
k_t^R  = RoPE( W_KR · c_t )           # (rope_head_dim) rope part (rotated)
v_t    = W_UV · c_t                   # (head_dim)
```

Per token per layer the cache holds `c_d + n_heads * rope_head_dim` values
instead of `2 * n_heads * head_dim`. The latent is *shared across heads*, so
per-head expressiveness is kept at a fraction of the cache. The win widens
with `n_heads` — for DeepSeek-V2's 128 heads the KV cache drops ~56x.

For Metis's small presets the per-token cache is:

| preset | MHA (`2·n_h·d_h`) | GQA-2 (`2·n_kv·d_h`) | MLA (`c_d + n_h·rD`) |
|--------|------------------:|---------------------:|---------------------:|
| small (d=256, h=4) | 512 | 256 | **192** (`c_d=64, rD=32`) |

MLA beats MHA by 2.7x and GQA-2 by 1.3x at the default latent/rope split, and
the gap grows with `n_heads` (see the ratio table below).

## 2. Weight absorption at inference

The content key is **never materialized from the latent during decode**. The
query is folded into latent space once per layer and attends against the
cached latent directly:

```
q_latent = W_UKᵀ · q_content
score    = q_latent · c  +  RoPE(q_R) · k^R
```

The value up-projection folds into the output projection,
`W_OV = W_O · W_UV` (a `(d_model, c_d)` matmul per layer, cached and
version-guarded), so:

```
o_lat  = softmax(score) · c          # attend over the latent
u      = W_OV · o_lat                # folded output projection
```

This is **algebraically identical** to reconstructing k/v and running ordinary
attention — `W_O (A · (W_UV c)) = (W_O W_UV) (A · c)`. The equality is verified
to floating-point precision in `benchmarks/verify_kv_parity.py` (max logit
diff `5.8e-6`).

## 3. Forward paths

`MLAAttention.forward` has two modes, selected by whether a cache is passed:

* **Cold prefill** (`kv_cache=None`): explicit K/V are reconstructed from the
  latent and dispatched through `causal_attention` — so on CUDA the fused
  FlashAttention / memory-efficient kernels engage. A fresh `MLALayerCache`
  (latent + rope keys) is returned.
* **Warm decode** (`kv_cache=MLALayerCache`): absorbed-path attention against
  the latent cache only; the backend logs `"mla_absorbed"`.

This mirrors DeepSeek's actual runtime: fused kernels at prefill, the
absorbed latent path at decode.

## 4. Architecture change (read this)

MLA is **not** a cache-only optimisation. It introduces new trainable
parameters per layer (`W_DKV`, `W_UK`, `W_KR`, `W_UV`, and a taller `W_Q`), so:

* a model built with `kv_backend="mla"` has **different weights** from a
  GQA/MHA checkpoint and **must be trained from scratch**;
* `n_kv_heads` is ignored (the latent is shared across all query heads);
* `mla_kv_latent_dim` and `mla_rope_head_dim` control the quality/memory
  trade-off (larger latent = more expressive but bigger cache).

The public API is unchanged — `MetisLM.forward(..., kv_cache=...) ->
(logits, loss, new_kv_cache)` with an opaque per-layer cache — so
`generate_text`, the server, the web UI and the scheduler work unmodified.

## 5. Config flags

| flag | default | meaning |
|------|---------|---------|
| `kv_backend` | `"default"` | `"mla"` selects MLA attention |
| `mla_kv_latent_dim` | `0` (= `d_model // n_heads`) | shared latent dim `c_d` |
| `mla_rope_head_dim` | `0` (= `head_dim // 2`, even) | RoPE part of the key |
| `mla_scale_head_dim` | `False` | scale scores by `sqrt(content+rope)` (True) or `sqrt(content)` (False, DeepSeek choice) |
| `use_rope` | `True` | required — MLA's rope-split key needs RoPE |

Validate: `ModelConfig(kv_backend="mla", n_kv_heads=2)` raises (MLA is MHA-only
in this implementation).

## 6. Memory ratio

Per-layer cache at context `T` (fp32):

```
default    = 2 · B · n_kv · T · d_h          bytes (growable, reallocates each step)
static     = 2 · B · n_kv · max_seq · d_h    bytes (preallocated, constant)
quantized  = B · n_kv · T · (2·d_h + 2)      bytes (int8 + per-token scales)
mla        = B · T · (c_d + n_heads · rD)    bytes (latent + rope keys)
```

The MLA ratio vs MHA `= (c_d + n_heads·rD) / (2·n_heads·d_h)`:
with `c_d = d_h`, `rD = d_h/2` this is `(1 + n_heads/2) / (2·n_heads)`
-> ~1.5·n_heads in the numerator / 2·n_heads... for n_heads=4: 192/512 =
0.375 (2.7x); for n_heads=8: 3.1x; for n_heads=32: 12x. MLA's cache advantage
scales with the number of heads.

## 7. Verified in this repo

* `benchmarks/verify_kv_parity.py` — absorbed-vs-explicit parity (`< 1e-4`),
  determinism, cross-backend `cached_len_of`.
* `tests/test_kv.py` — cold prefill, decode, overflow reset, absorbed-vs-explicit.
* `benchmarks/benchmark_kv.py --mode throughput` — CPU decode ms/step
  (MLA is comparable to the default backend at small scale).

## 8. Limitations

* **Prefill does not use the latent path** on this CPU build; the explicit
  K/V path is used (needed for fused-kernel dispatch). The latent cache is
  still returned and used for all subsequent decode.
* The absorbed decode attention is manual math (softmax over `T_k`), not a
  fused kernel — on CUDA this trades some decode speed for the large cache
  win, exactly as in DeepSeek's design.
* `position_ids`-based packing uses the explicit path; the absorbed decode
  path assumes contiguous positions starting at `cached_len`.

## 9. Further reading

* DeepSeek-AI. *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model*. arXiv:2405.04434.
* DeepSeek-AI. *DeepSeek-V3 Technical Report*. arXiv:2412.19437.
* The MLA section of `docs/kv_cache.md` for the comparison against the other
  KV-cache backends.
