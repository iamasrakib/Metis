# KV Cache Subsystem

An optional, flag-gated KV-cache engine that replaces the legacy growable
`(K, V)` tuple cache during inference while preserving the public API and
output quality.

`kv_backend` selects the engine (config field, CLI `--kv-backend`, or
`METIS_KV_BACKEND` env var). Implementations live in `metis/kv.py`
(static / quantized) and `metis/mla.py` (MLA).

---

## 1. The four backends

| Backend | Mechanism | Output quality | When to use |
|---------|-----------|----------------|-------------|
| `default` | legacy `[(K, V), ...]` per layer, `torch.cat` each step | bit-identical reference | baseline; debugging |
| `static` | preallocated `(B, n_kv, max_seq, head_dim)` buffers, in-place writes | **bit-identical** (`torch.equal`) | production decode, no per-step allocation |
| `quantized` | static layout + int8 K/V, per-token scales | near-lossless (max logit diff ~6e-3) | long-context / memory-bound serving |
| `mla` | Multi-head Latent Attention (architecture change) | n/a (train from scratch) | research; large-`n_heads` models |

---

## 2. Motivation

The legacy cache appends each new token's K/V with `torch.cat`, which
allocates and copies the whole prefix on every decode step:
`O(T)` allocation + copy per step -> `O(T^2)` total during a generation of
`T` tokens. It also stores full fp32 K/V tensors, so memory grows linearly
with context and dominates at long sequences or large batches.

The subsystem addresses three axes:

* **layout** — preallocate once, write in place (`static`);
* **compression** — store int8 with per-token scales (`quantized`);
* **architecture** — replace the K/V cache with a learned latent (`mla`).

## 3. Cache object design

All backends preserve the model contract:

```python
logits, loss, new_kv_cache = model(idx, ..., kv_cache=previous_cache)
```

The cache is **opaque** to callers — it round-trips through `generate_text`,
the REST server, the web UI and the execution scheduler unchanged.

```python
# default / mla: plain list of per-layer caches (rebuilt each forward)
cache = [(k_0, v_0), (k_1, v_1), ...]           # default
cache = [MLALayerCache(c_0, k_rope_0), ...]     # mla

# static / quantized: a KVCache object (list-like over LayerKV)
cache = KVCache("static", config, n_layers)
cache[i]            # -> LayerKV for layer i
cache.cached_len    # -> live context length
cache.allocated_bytes() / cache.used_bytes()
```

`metis.kv.cached_len_of(cache)` returns the live length of **any** cache form
(used by `generate_text` for the sliding-window overflow check), and
`metis.kv.cached_bytes` reports resident bytes. `LayerKV` is index-compatible
with a `(K, V)` tuple (`cache[i][0]` / `cache[i][1]`), so generic code keeps
working.

**Lazy allocation**: a `LayerKV` allocates its buffers on the first
`append()` — it needs `B`, device, storage dtype and head count from the
incoming tensor. A configured backend that is never exercised (e.g. training
forwards under a `"static"` config passing `kv_cache=None`) allocates nothing.

## 4. Static backend

```python
# preallocate once
self._k = torch.empty(B, n_kv, max_seq_len, head_dim, ...)
self._v = torch.empty(B, n_kv, max_seq_len, head_dim, ...)

# each step: write in place, read the live prefix
k[..., length:length+T, :] = k_new
k_all = k[..., :length+T, :]
```

* bit-identical to `default` (same tensors, same kernels — verified with
  `torch.equal` across MHA, GQA and attention-sink configs);
* flat memory (preallocates `max_seq_len`, never reallocates);
* `kv_cache_dtype` = `fp16`/`bf16` halves the footprint with negligible error
  (values are re-cast to the compute dtype on read);
* overflow raises a clear error; `generate_text` triggers `reset()` (buffer
  reuse) instead of dropping the cache, so the sliding-window path does not
  re-allocate.

## 5. Quantized backend

Per-token symmetric int8 (KVQuant-style, simplified):

```python
scale = max|k| over head_dim / 127              # one scale per (B, n_kv, T)
q     = clamp(round(k / scale), -127, 127)      # int8 payload
k_hat = q * scale                                # dequantized on read
```

* ~3.8x cache-memory reduction vs fp32 (int8 payload + fp32 scales);
* near-lossless for the small presets: measured **max logit diff ~6e-3**
  (median ~2e-3), far below sampling noise — see the parity numbers below;
* dequantization on read costs ~25-40% CPU decode time (measured) — a
  memory-for-speed trade; on CUDA this is usually a win (memory-bound decode);
* `kv_quant_scheme` is validated to `"int8"` (per-token); grouped / per-channel
  schemes are future work.

## 6. Integration points

* `metis/model.py` — `CausalSelfAttention.forward` accepts a `LayerKV`
  (writes in place via `append`, reads via `keys_values`); `MetisLM.forward`
  builds/round-trips a `KVCache` for `static`/`quantized` and a list for
  `default`/`mla`; `TransformerBlock` builds `MLAAttention` when
  `kv_backend="mla"`.
* `metis/generate.py` — the sliding-window overflow check uses
  `cached_len_of`, and a static/quantized cache is `reset()` (buffers reused)
  instead of dropped.
* `metis/scheduler/runtime.py` — indexes `cache[i]` and round-trips the cache
  object, unchanged.
* `metis/cli.py` / `generate.py` — `--kv-backend` flag + `METIS_KV_BACKEND`
  env var; `load_model_and_tokenizer(..., kv_backend=...)`.
* `metis/__init__.py` — exports `KVCache`, `LayerKV`, `KVBackendInfo`,
  `cache_memory_bytes`, `cached_bytes`, `cached_len_of`, `kv_cache_ratio`,
  `quantize_per_token`, `dequantize_per_token`.

## 7. Parity verification

`benchmarks/verify_kv_parity.py` checks, on identical weights:

* `static` vs `default` — **bit-identical** (`torch.equal`) for MHA, GQA-2,
  and attention-sink configs, over multi-step decode;
* `quantized` vs `default` — max logit diff `< 1.0` (actual ~6e-3);
* `mla` absorbed decode vs full re-prefill (explicit K/V) — max diff `5.8e-6`;
* MLA determinism; `cached_len_of` across all cache types; analytic memory
  formulas vs actual tensor sizes.

## 8. Measured results

Environment: CPU, torch 2.13, `small` preset (d_model=256, n_heads=4,
n_kv_heads=2, max_seq_len=512). Full numbers in
`benchmarks/results/benchmark_kv_*.{json,md}`.

### Memory (per-layer, fp32)

| T | default | static | quantized | mla | quant/default | mla/default |
|---|--------:|-------:|----------:|----:|--------------:|------------:|
| 32 | 32 KB | 512 KB | 8.5 KB | 24 KB | 3.8x | 1.3x |
| 64 | 64 KB | 512 KB | 17 KB | 48 KB | 3.8x | 1.3x |
| 128 | 128 KB | 512 KB | 34 KB | 96 KB | 3.8x | 1.3x |
| 256 | 256 KB | 512 KB | 68 KB | 192 KB | 3.8x | 1.3x |
| 512 | 512 KB | 512 KB | 136 KB | 384 KB | 3.8x | 1.3x |

Static preallocates `max_seq_len` so it is flat; at `T = max_seq_len` it
equals default. Quantized cuts default ~3.8x at every length. MLA's ratio vs
MHA grows with `n_heads` (see `docs/mla.md`).

### Throughput (CPU decode, ms/step, median of 10)

| T prefix | default | static | quantized | mla |
|---------:|--------:|-------:|----------:|----:|
| 64 | 3.03 | 2.89 | 3.93 | 3.34 |
| 128 | 3.11 | 2.97 | 3.78 | 3.92 |
| 256 | 3.67 | 3.51 | 3.91 | 3.60 |

Static is comparable to default (and avoids the per-step `cat` allocation —
more visible on CUDA). Quantized pays dequant overhead on CPU. MLA's absorbed
decode is comparable at this scale.

### Parity (max logit diff vs default baseline)

| backend | prefill | decode |
|---------|--------:|-------:|
| static | 0 (bit-identical) | 0 (bit-identical) |
| quantized | 5.9e-3 | 5.9e-3 |
| mla | n/a (different arch) | 5.7e-5 (absorbed vs explicit) |

## 9. Limitations and future work

* **Paged attention** (vLLM-style block tables) was evaluated and deliberately
  **not** implemented: for single-sequence tiny-model serving the flat static
  buffer delivers the same benefit with far less machinery. Block tables earn
  their complexity only with many concurrent sequences / prefix sharing.
* **Token-level eviction** (H2O, StreamingLLM, SnapKV) is **lossy** and
  violates the "preserve output quality" requirement; the sliding-window reset
  is the only eviction path.
* **fp8 / grouped quantization** (`kv_quant_scheme="fp8"`, per-channel/grouped
  scales) and **KIVI-style hybrid precision** are natural next steps on
  hardware with fp8 kernels.
* **Warm-start chunked prefill** is supported via `KVCache.from_legacy` (a
  legacy `[(K, V), ...]` list is re-appended and compressed on ingestion).

## 10. Further reading

* `docs/mla.md` — the MLA backend and weight absorption.
* `benchmarks/verify_kv_parity.py`, `benchmarks/benchmark_kv.py`,
  `tests/test_kv.py` — verification and benchmarks.
* DeepSeek-AI, *DeepSeek-V2* (arXiv:2405.04434); KVQuant (arXiv:2401.18079);
  KIVI (arXiv:2402.02750); vLLM PagedAttention (arXiv:2309.06180).
