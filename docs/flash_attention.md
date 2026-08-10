# FlashAttention-2 in Metis

Metis runs its causal attention through a FlashAttention-2 dispatch layer
([`metis/attn.py`](../metis/attn.py)). The same call path serves training,
validation, and KV-cache inference, and transparently picks the fastest kernel
available on the machine — falling back to PyTorch's fused SDPA kernels, then
to an exact manual reference, without any code change at the call site.

```
┌──────────────────────────────────────────────────────────────────┐
│ CausalSelfAttention.forward                                      │
│   q/k/v projections → QK-Norm → RoPE → KV-cache → causal_attention│
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │  resolve_backend  │   pure, data-independent
                    └───────────────────┘
                              │
        ┌──────────────┬──────┴───────────────┬─────────────────┐
        ▼              ▼                      ▼                 ▼
 dao-AILab         torch SDPA            torch SDPA        exact manual
 flash_attn        FLASH_ATTENTION       EFFICIENT         math reference
 package (Linux)   (FA2 kernel)          (mem-efficient)   (byte-identical
                                              │            to legacy)
                                              └── unavailable → degrade
```

---

## 1. Backend priority and names

Dispatch priority (best → worst). The first backend that can run the call wins;
every fallback is automatic and warned-once.

| # | Backend | What it is | Fused? |
|---|---------|-----------|--------|
| 1 | `flash_attn` | [dao-AILab FlashAttention-2 package](https://github.com/Dao-AILab/flash-attention). Linux-only wheels. | ✅ |
| 2 | `flash` (`sdpa_flash`) | torch SDPA `FLASH_ATTENTION` — the FlashAttention-2 kernel shipped in torch CUDA builds. | ✅ |
| 3 | `mem_efficient` (`sdpa_mem_efficient`) | torch SDPA `EFFICIENT_ATTENTION` — fused memory-efficient kernel (FlashAttention-family online softmax). Present in the Windows torch wheels. | ✅ |
| 4 | `math` (`sdpa_math` / manual) | exact manual `QKᵀ → mask → softmax → V`, byte-identical to the legacy implementation. | ❌ |

User-facing backend names accepted by `ModelConfig.attn_backend`, the
`METIS_ATTN_BACKEND` env var, and the CLI `--attn-backend` flag:

```
auto · flash_attn · sdpa · flash · mem_efficient · math
```

- `auto` — best available fused kernel (what the model uses by default).
- `sdpa` — best *torch* kernel (never the external package).
- `flash` / `mem_efficient` — pin that specific SDPA kernel.
- `flash_attn` — pin the external package.
- `math` — pin the exact manual reference (deterministic / debugging).

On the reference machine this document was written on (Windows, PyTorch
2.6.0+cu124, RTX 2050 / compute capability 8.6) the `flash_attn` package and
the torch FLASH kernel are both absent, so `auto` resolves to
`sdpa_mem_efficient` — a genuine FlashAttention-family fused kernel.

---

## 2. Automatic GPU capability detection

At first use (and cached for the process) Metis probes the machine empirically:
it runs a tiny real CUDA call pinned to each fused kernel and records what
actually works — far more reliable than inferring from architecture strings.
The snapshot is reported by [`detect_attention_backends()`](../metis/attn.py)
(also available as `metis.detect_attention_backends()`):

```python
>>> from metis import detect_attention_backends, fused_attention_supported
>>> detect_attention_backends()
{'device': 'cuda',
 'gpu_name': 'NVIDIA GeForce RTX 2050',            # detected GPU
 'compute_capability': (8, 6),                     # sm_86, Ampere
 'torch': '2.6.0+cu124',
 'flash_attn': None,                               # package not installed
 'flash_attn_gqa': False,
 'torch_flash': False,                             # SDPA FA2 kernel absent
 'torch_mem_efficient': True,                      # mem-efficient available
 'torch_math': True,
 'fused_gqa': False,                               # fused GQA via enable_gqa?
 'fused_available': True,                          # some fused kernel works
 'recommended': 'sdpa_mem_efficient'}              # what "auto" resolves to

>>> fused_attention_supported()
True
```

`ModelConfig.summary()` and `metis train` print the requested vs. recommended
backend at startup; `MetisLM.get_attention_backend()` additionally reports the
kernel each layer most recently used (so you can confirm the fused path engaged
during a run or a benchmark).

---

## 3. Numerical behavior

**Exact path.** With `use_flash_attn=False` (or `attn_backend="math"`) Metis
uses `math_attention`, which is *bit-identical* to the legacy manual
`CausalSelfAttention` implementation — same `Q @ Kᵀ`, same sliced lower-triangle
mask, same softmax/dropout/`@ V` order. This is the deterministic reference and
is covered by `tests/test_attn.py::TestMathAttentionReference` (`torch.equal`
assertions).

**Fused path.** FlashAttention-2 and the memory-efficient kernel accumulate in
fp32 and are cast to fp16/bf16 on output; results agree with the exact path to
~1e-2. The model-level tests assert agreement within `rtol/atol = 2e-2` for
fp16 and bf16, for prefill and decode, MHA and GQA.

**Semantics preserved across backends:**

- *Scale* — `1/sqrt(head_dim)` is the kernel default. A custom scale is folded
  into Q for SDPA (which hard-codes `1/sqrt(d)`) and passed as `softmax_scale`
  to the package.
- *Causal masking* — prefill (`T_q == T_k`) uses a causal mask. KV-cache decode
  (`T_q < T_k`) attends to all cached keys, exactly reproducing the legacy
  sliced-`tril(max_seq_len)` buffer (torch's `is_causal` would *mis*-mask this
  case, so it is deliberately disabled).
- *GQA / MQA* — if the fused build supports `enable_gqa` it is used; otherwise
  KV heads are expanded explicitly (`_repeat_kv`). Same math either way.
- *Dropout* — applied only in training, matching `nn.Dropout` semantics.

---

## 4. Feature support matrix

| Feature | Status |
|---------|--------|
| Training (forward + backward, autograd) | ✅ fused kernels have custom backward |
| Inference (`generate_text`, `metis generate/chat/serve/ui`) | ✅ |
| KV cache (prefill + incremental decode) | ✅ |
| Causal masking (prefill) | ✅ |
| RoPE (including decode position offsets) | ✅ |
| GQA / MQA / MHA | ✅ |
| AMP (fp16 + bf16 autocast, GradScaler) | ✅ fused kernels are fp16/bf16 |
| Gradient checkpointing (`use_checkpointing`) | ✅ `torch.utils.checkpoint` |
| QK-Norm | ✅ applied before the dispatcher; stays in the fused path under AMP (see §6) |
| Attention sink | ✅ |
| MoE | ✅ orthogonal to attention |
| `torch.compile` | ✅ dispatch is traceable; `set_backend_flags` mirrors the selection under compilation |
| CPU (no GPU) | ✅ degrades to the exact manual path |

Fused kernels engage only when the call is eligible: **fp16/bf16** and
**head_dim a multiple of 8 in [8, 256]**. Any other dtype, shape, or device
automatically and silently uses the next best backend.

---

## 5. Configuration and controls

All existing configuration keeps working — nothing is removed, defaults are
unchanged.

```python
from metis import ModelConfig

cfg = ModelConfig(d_model=384, n_heads=6, n_kv_heads=3,
                  use_flash_attn=True,   # default True
                  attn_backend="auto")   # default "auto"
```

| Control | Values | Effect |
|---------|--------|--------|
| `config.use_flash_attn` | `True` / `False` | `False` pins the exact manual math reference. |
| `config.attn_backend` | see §1 | Selects a specific backend; `auto` = best available. |
| `METIS_ATTN_BACKEND` env var | same set | Overrides the config for a whole process (highest precedence). |
| CLI `--attn-backend` | same set | `metis train/generate/chat/info/serve --attn-backend ...` |

The env var is the convenient escape hatch for deployments:
`METIS_ATTN_BACKEND=math metis chat` reproduces exact legacy behavior;
`METIS_ATTN_BACKEND=mem_efficient metis train` forces a specific kernel.

---

## 6. Integration fixes found during review

Two real bugs were found and fixed in [`metis/model.py`](../metis/model.py)
during the FlashAttention integration — both are silent fused-path regressions
that only show up under AMP autocast on CUDA, so neither is visible on CPU or in
fp32.

**KV-cache decode RoPE dtype.** The KV-cache RoPE branch cast the rotated Q/K
back to the *layer-input* dtype. Under AMP the input is the fp32 embedding
output (`nn.Embedding` is not autocast), so Q/K silently became fp32 during
decode — excluded from the fp16/bf16 fused kernels — and cached inference fell
back to the slow manual path. Prefill was unaffected because `apply_rope`
preserves the projected dtype, which made the regression invisible to prefill
benchmarks. The branch now casts back to the *projected* dtype, so decode
engages the same fused kernel as prefill.

**QK-norm output dtype.** `QKNorm`'s RMSNorm ends with `x * weight` where the
weight is a fp32 `nn.Parameter`; AMP autocast promotes that elementwise multiply
to fp32, so the normalized q/k reached the dispatcher as fp32 and were excluded
from the fused kernels — `use_qk_norm` silently forced the fp32 manual path on
every call, in training *and* inference. `QKNorm.forward` now returns
`self.norm(x).type_as(x)` (normalize in fp32, output in the input dtype), which
keeps q/k in the projected dtype. Every other projection output under autocast
is already fp16/bf16, so this makes QK-norm consistent with the rest of the
model.

Both regressions are locked in by
`tests/test_attn.py::TestFusedEquivalence`:
`test_kv_cache_decode_uses_fused_backend` asserts a fused kernel on prefill and
decode and that cached step-by-step decode still matches the full forward;
`test_qk_norm_uses_fused_backend` asserts the QK-norm path uses the same fused
kernel as the plain path.

---

## 7. Benchmarks and memory comparison

Run the benchmark to reproduce old-vs-new numbers on any machine:

```bash
python benchmarks/benchmark_attention.py                       # everything
python benchmarks/benchmark_attention.py --mode kernel         # raw kernels
python benchmarks/benchmark_attention.py --mode model          # train + decode steps
python benchmarks/benchmark_attention.py --mode memory         # peak activation memory
python benchmarks/benchmark_attention.py --iters 10 --backend mem_efficient
```

Results are written to `benchmarks/results/` as JSON + Markdown:

- **Kernel** — raw attention timings (prefill MHA/GQA at T=128–512, decode at
  T_k=64–512), old `math_attention` vs. new dispatched kernel.
- **Model train** — full forward+backward AMP steps, with and without gradient
  checkpointing; records median time, tokens/s, and peak GPU memory.
- **Model decode** — legacy re-prefix-per-token vs. working KV-cache decode
  (the same weights; only the decode strategy differs).
- **Memory** — peak activation memory of one train step vs. sequence length
  (T = 128…2048), math vs. fused at *fixed* no-checkpointing, isolating the
  attention kernel's contribution.

**Why the memory gap grows with T:** the manual path materializes the
`(B, H, T, T)` scores matrix and retains the softmax output for backward —
O(T²) activation memory. FlashAttention-2 / mem-efficient keep attention
activations O(T) by tiling the computation in SRAM and recomputing the softmax
statistics in the backward pass. At long sequences the saving is the difference
between fitting a training run on the GPU or not.

### Results on the reference machine

Machine: **RTX 2050 (sm_86, 4 GB), torch 2.6.0+cu124, Windows**, `auto` →
`sdpa_mem_efficient`. Full report: timestamped Markdown/JSON in `benchmarks/results/`.

**Kernel-level attention (fp16)** — speedup of the fused dispatch over the
exact manual math, MHA and GQA, prefill and decode:

| Workload | Speedup range |
|----------|--------------:|
| prefill MHA (T=128–512) | 2.4× – 8.5× |
| prefill GQA (T=128–512) | 1.4× – 5.6× |
| decode MHA (T_k=64–512) | 2.9× – 5.3× |
| decode GQA (T_k=64–512) | 1.4× – 1.6× |

**Training step** (forward + backward, AMP, B=2, T=256, d_model=128, 4 layers):

| Backend | Checkpointing | Time (ms) | Peak mem (MB) |
|---------|--------------:|----------:|--------------:|
| math (old) | off | 25.2 | 30.0 |
| fused (new) | off | 24.3 | **16.8 (−44%)** |
| math (old) | on | 41.3 | 10.0 |
| fused (new) | on | 38.3 | **4.3 (−57%)** |

**Peak activation memory vs. sequence length** (one train step, no
checkpointing) — the O(T²) vs. O(T) gap:

| T | math peak (MB) | fused peak (MB) | saving |
|---|---------------:|----------------:|-------:|
| 128 | 102.6 | 101.0 | 1.6% |
| 256 | 112.4 | 105.6 | 6.0% |
| 512 | 146.5 | 115.0 | 21.5% |
| 1024 | 269.6 | 133.6 | 50.4% |
| 2048 | 736.0 | 171.0 | **76.8%** |

**Decode step** (legacy re-prefix vs. working KV cache; kernel view): the
fused decode kernel is 2.9–5.3× faster than the manual one. At the model level
on this *tiny* test model the per-step overhead of projections/norms dominates
(~1.0–1.1×), which is expected — the attention kernel's share of a decode step
grows with model size and KV length.

---

## 8. Platform notes

- **Windows** — the dao-AILab `flash_attn` package has no Windows wheels, and
  some Windows torch builds omit the SDPA FLASH kernel. Metis therefore uses
  torch SDPA `EFFICIENT_ATTENTION` (memory-efficient, FlashAttention-family),
  which the Windows wheels do ship. This is FlashAttention-2-class performance
  with zero extra installation.
- **Linux (CUDA, Ampere+)** — installing `flash-attn` makes `auto` prefer the
  package; otherwise torch's own FA2/mem-efficient kernels are used.
- **Older GPUs / CPU** — fused kernels are unavailable and the dispatcher
  degrades automatically to the exact manual path. Nothing crashes.
- **torch.compile** — the dispatch is traceable and `set_backend_flags` mirrors
  the kernel selection under compilation, but Inductor needs Triton, which the
  Windows torch wheels do not ship. On such builds `torch.compile` is a
  platform limitation orthogonal to the attention integration; on Linux CUDA
  wheels (which include Triton) it works as documented.

The integration is deliberately dependency-free: torch's fused SDPA kernels are
the default "FlashAttention-2 everywhere", and the external package is optional
and only used when present.
