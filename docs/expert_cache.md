# Persistent expert execution cache in Metis

Metis caches the stacked + dtype-cast expert weight tensors that the grouped
MoE engine re-materialises on every forward, keeping active experts resident
in GPU memory across forwards and eliminating the per-forward `torch.stack(views).to(dtype)`
overhead.

The cache is **on by default** for all MoE models (`moe_cache_size=64`), active
in both training and inference, and disabled only when `torch.compile` is
enabled (the cache's Python control flow is not Dynamo-traceable).

## Why this matters

Without the cache, every `forward_grouped` call re-reads every active expert's
fp32 master weights from GPU global memory, writes a stacked fp32 copy, re-reads
it, and writes a bf16/fp16 cast copy — per group, per MoE layer, per token
step. In generation (1 token/step, weights frozen) this is pure repeated loading;
in training it repeats across gradient-accumulation micro-batches and validation.

The traffic model for one group of `A` experts with `D`-dimensional inputs and
`hidden`-dimensional hidden states:

| Operation | Bytes moved |
|---|---|
| Read fp32 sources (w1 + w2) | `2·A·D·hidden·4` |
| Write stacked fp32 | `2·A·D·hidden·4` |
| Read stacked fp32 (for cast) | `2·A·D·hidden·4` |
| Write cast tensors (bf16) | `2·A·D·hidden·2` |
| **Stack+cast total (remat)** | **`28·A·D·hidden` bytes** |
| Read cast tensors (GEMM) | `2·A·D·hidden·2` (every forward) |
| **Total MoE weight traffic** | **`30·A·D·hidden` bytes** |

For d_model=128, hidden=256, A=8 experts: **7.2 MB per rebuild**, **0.5 MB per forward GEMM read**.
With the cache on a hit, the 7.2 MB stack+cast is eliminated; only the 0.5 MB
GEMM read remains. This gives a **total MoE weight-traffic reduction of ~84%**
(not 95%, because the GEMM reads still happen every forward).

## Design

### Entry structure

Each cache entry stores:

- **Key**: `(sorted_group_tuple, dtype_str, requires_grad)` — `requires_grad`
  is `torch.is_grad_enabled() and any(t.requires_grad for t in sources)`,
  so inference (no-grad) and training (grad) builds never collide.
- **Value**: cached `(w1_group, w2_group)` stacked + cast tensors.
- **Staleness signature**: `tuple((data_ptr, _version, shape) for t in sources)`.
  On lookup, the current signature is recomputed; a mismatch → stale → rebuild.
- **Byte accounting**: `remat` (bytes avoided on a hit) and `resident` (bytes
  the entry occupies).

### Lookup flow

```
get_or_build(group, dtype, sources, build)
    ├── key = (group, dtype, requires_grad)
    ├── sig = _signature(sources)
    ├── entry = cache[key]
    │   ├── hit  (entry exists, sig matches) → return cached tensors
    │   └── miss (or stale) → build(), insert, evict if needed
    └── return (w1_group, w2_group)
```

### Eviction

LRU via `OrderedDict`. Evicts the oldest entry while `entries > capacity`, and
while `resident_bytes > byte_capacity` (if byte budget is set).

### Staleness detection

Signature-based: `(data_ptr, _version, shape)` per source view. Detected automatically:

| Operation | data_ptr | _version | Detected |
|---|---|---|---|
| Non-fused AdamW step | same | bumps | ✅ `_version` |
| Fused AdamW step | same | same | ❌ needs explicit `invalidate()` |
| `copy_()` | same | bumps | ✅ `_version` |
| `load_state_dict(copy_)` | same | bumps | ✅ `_version` |
| `load_state_dict(assign=True)` | changes | — | ✅ `data_ptr` |
| `.to(device)` reassignment | changes | — | ✅ `data_ptr` + `_apply` hook |
| EMA `apply_shadow` / `restore` | changes | — | ✅ `data_ptr` + `invalidate()` |
| `.data.copy_()` / `.data.add_()` | same | same | ❌ needs explicit `invalidate()` |

**Fused AdamW** (the CUDA optimiser used by `MetisLM.configure_optimizers`)
mutates param storage **without** bumping `_version`. The framework's own
training loop (`training.py`, `cuda_graphs.py`) calls
`model.invalidate_moe_caches()` after every `optimizer.step()` /
`scaler.step()` to handle this correctly. Custom training loops should do the
same.

### Byte accounting

- `remat` per miss/rebuild = `3·src_bytes + 2·cast_bytes` (the full
  stack+cast pipeline traffic avoided on a future hit).
- `bytes_saved` = sum of `remat` over all hits.
- `bytes_built` = sum of `remat` over all misses.
- `bytes_read` = per-lookup GEMM weight reads (cast tensors), accumulated
  on every lookup (hit or miss). Equal to `resident` per entry.
- `bandwidth_reduction_pct` = `100·bytes_saved/(bytes_saved+bytes_built+bytes_read)`.
  Total MoE weight-traffic reduction including per-forward reads.
- `stackcast_avoided_pct` = `100·bytes_saved/(bytes_saved+bytes_built)`.
  Stack+cast remat avoided (legacy metric, equals hit rate).

## Configuration

`ModelConfig` fields (round-trip through `save_json`/`from_json`):

| Field | Type | Default | Description |
|---|---|---|---|
| `moe_cache_size` | int | 64 | Max group entries (0 = disabled) |
| `moe_cache_bytes` | int | 0 | Optional byte budget (0 = unbounded) |

CLI flags (`metis train`, `metis generate`, `metis chat`, etc.):

```
--moe-cache-size N     Max entries in the persistent expert weight cache
--moe-cache-bytes N    Optional byte budget for the expert cache
```

Environment variables (override at runtime):

| Variable | Effect |
|---|---|
| `METIS_MOE_CACHE_SIZE` | Override `moe_cache_size` |
| `METIS_MOE_CACHE_BYTES` | Override `moe_cache_bytes` |

```bash
metis train --preset tiny --use-moe --moe-cache-size 128
METIS_MOE_CACHE_SIZE=0 metis serve    # disable cache for serving
```

## Accessing cache statistics

```python
# Per-MoE-layer stats (list, one per layer; None for non-MoE layers):
stats = model.get_moe_cache_stats()
for i, s in enumerate(stats):
    if s:
        print(f"layer {i}: hit_rate={s['hit_rate']:.1%}, "
              f"bw_reduction={s['bandwidth_reduction_pct']:.1f}%")

# Or from metis.expert_cache helpers:
from metis import expert_cache_hit_rate, expert_cache_bandwidth_reduction
print(f"hit rate: {expert_cache_hit_rate(model.layers[0].ffn._cache):.1%}")
```

## Numerical parity

`python benchmarks/verify_expert_cache_parity.py --device cuda` (19/19 PASS):

```
[forward]   cached vs uncached fp32  max_err=0.00e+00
[forward]   cached vs uncached fp16  max_err=4.29e-05
[forward]   cached vs uncached bf16  max_err=1.53e-02
[bit-id]    logits bit-identical     PASS
[bit-id]    loss bit-identical       PASS
[hit-rate]  repeated forwards        90.0% hit rate
[staleness] param.copy_ rebuild      PASS
[grad]      cached backward match    max_grad_diff=0.00e+00
[bytes]     accounting consistency   66.7% bw reduction
[lru]       eviction within bounds   entries=2, evictions=3
[config]    negative sizes raise     PASS
[device]    .to() resets cache       PASS
```

## Benchmark results

`python benchmarks/benchmark_expert_cache.py --device cuda` — RTX 2050,
torch 2.6.0+cu124 (10 runs, median).

### Hit rate (same-input eval forwards, 4 layers, 8 experts)

- Same-input hit rate: **95.0%** (114/120 lookups)
- Mixed-input hit rate: **53.8%** (realistic steady-state)
- Stack+cast avoided: **95.0%**
- Total MoE weight-traffic reduction: **84.0%**
- Resident memory: **11.3 MB**

### AMP behavior

Under bf16/fp16 AMP autocast, the cache now pre-casts to the compute dtype
rather than storing fp32 copies. This halves resident cache memory and
eliminates the per-bmm autocast weight cast that previously defeated the
cache's purpose. The `--device cpu` benchmark shows fp32 caching (CPU has no
autocast optimization).

### Throughput

| workload | cache_off | cache_on | speedup |
|---|---:|---:|---:|
| steady-state train step (warm cache) | 36.89 ms | 36.07 ms | 1.02× |
| real train step (+ optimizer.step + invalidate) | ~37 ms | ~37 ms | ~1.00× |
| autoregressive decode (tiny, 4×8) | 11.64 ms | 11.64 ms | 1.00× |
| autoregressive decode (larger 8×16) | 24.08 ms | 23.55 ms | **1.02×** |

On the tiny benchmark model the stack+cast traffic is not the wall-clock
bottleneck, so throughput is near-par (1.00×). The **primary win is the GPU
memory-traffic reduction** from keeping active experts resident. Real train-step
throughput (with optimizer.step + invalidation each step) shows minimal cache
benefit because hits only occur within gradient-accumulation micro-batches. The
wall-clock win appears as expert count × hidden size grows: **1.02× decode
speedup** on a larger MoE model.

**CPU note**: On CPU, the cache is neutral-to-slightly-slower (0.96×–0.98×)
because the signature/lookup overhead exceeds the stack+cast savings when
memory bandwidth is not the bottleneck. The cache targets GPU memory traffic.

### Peak GPU memory

| T | cache_off (MB) | cache_on (MB) | delta (MB) |
|---:|---:|---:|---:|
| 128 | 45.5 | 56.6 | +11.1 |
| 256 | 51.4 | 60.4 | +9.0 |

The resident cached tensors add ~9–11 MB of GPU memory (the cached expert
weight tensors for all layers and groups). Under AMP, this is halved (~5–6 MB)
because the cache stores bf16 instead of fp32.

## Reproduce

```bash
# correctness + parity
python -m pytest tests/test_expert_cache.py -q
python benchmarks/verify_expert_cache_parity.py --device cuda

# full benchmark (hit rate, throughput, decode, memory — JSON + Markdown in benchmarks/results/)
python benchmarks/benchmark_expert_cache.py --device cuda
python benchmarks/benchmark_expert_cache.py --device cuda --mode decode   # decode-only
python benchmarks/benchmark_expert_cache.py --device cuda --mode hitrate_mixed  # realistic hit rate

# also: existing MoE tests unchanged
python -m pytest tests/test_moe.py -q
python benchmarks/verify_moe_parity.py --device cuda
```
