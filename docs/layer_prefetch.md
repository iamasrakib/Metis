# Layer prefetching in Metis

While layer *N* computes, **layer prefetching** speculatively warms layer
*N+1*'s MoE expert cache on a dedicated CUDA stream, so the next layer's
`get_or_build` calls hit and never stall on the synchronous
`torch.stack(views).to(dtype)` that a cache miss performs on the compute
stream. That stack+cast — the per-layer gather of the routed experts' weights
into one cast tensor — is the "memory movement between layers" this overlaps
with compute.

On by default (`use_layer_prefetch=True`, CUDA). `--no-layer-prefetch` /
`use_layer_prefetch=False` disables it; on CPU-only hosts the prefetcher is a
no-op (see *CPU behavior* below).

## Design

```
MoE layer N                          prefetch stream (CUDA)
───────────────                      ─────────────────────
layer(N-1) ── prefetch_next(N) ────► build layer N's groups (stack+cast)
layer(N)   ── compute ─────────────► build layer N+1's groups
layer(N+1) ── get_or_build ────────►  (hits — already resident)
```

- **Prediction = temporal locality.** Each MoE layer records the expert groups
  it routed to (`forward_grouped` reports them via a `record` sink; stored per
  layer in the `LayerExpertPrefetcher`). The next forward prefetches each
  layer's groups during the *preceding* layer's compute. Routing is stable
  across steps — very stable in KV-cache decode — so most prefetches hit.
- **Speculative builds on a side stream.** `ExpertCache.prefetch()` runs the
  identical stack+cast the miss path would run, on a dedicated stream. Each
  prefetched entry records a completion event; the first consumer on the
  compute stream `wait_event`s it (a no-op once the build finished during the
  previous layer). No background thread — all stream-ordered on the main
  thread, Windows-safe.
- **Cold start is free.** The first forward has no routing history → no
  prefetches → identical behavior to today.

## Correctness

1. **Deterministic build.** The prefetch invokes the exact `build()` lambda
   (stack of the same source views at current weights) the miss path uses →
   bit-identical output. Verified by
   `tests/test_layer_prefetch.py::TestCorrectness` (logits, loss, **and
   gradients** match to `max diff 0.00e+00`).
2. **Staleness signature is authoritative.** A prefetched entry is still
   subject to the cache's `(data_ptr, _version, shape)` check on every real
   lookup. Weights change between prefetch and use → the entry is discarded
   and rebuilt on demand. Prefetching never bypasses staleness.
3. **Stream ordering.** `get_or_build` waits on a side-built entry's event
   before the compute stream reads it — no cross-stream race. Weights are
   read-only during a forward, so the side stream's reads never conflict with
   compute.
4. **No CUDA-graph conflict.** CUDA graphs are already disabled for MoE
   ("MoE routing is data-dependent"); an `is_current_stream_capturing()` guard
   is added regardless.
5. **Autograd.** A prefetched `requires_grad` tensor's `grad_fn` references the
   source params (same as a miss build) and is consumed in the same forward —
   the existing "cached tensor in multiple graphs" pattern.

## Files

| File | Role |
|------|------|
| `metis/layer_prefetch.py` | `LayerExpertPrefetcher` — records routing, issues side-stream builds |
| `metis/expert_cache.py` | `ExpertCache.prefetch()` + `side_event` + `prefetched`/`prefetch_useful` counters |
| `metis/moe.py` | `forward_grouped(record=...)`, `MoE._record_routing` |
| `metis/model.py` | create prefetcher, `prefetch_next(i)` in the forward loop |
| `metis/config.py`, `cli.py` | `use_layer_prefetch`, `--no-layer-prefetch` |

## Measured results

`benchmarks/benchmark_layer_prefetch.py` reports a per-layer **timeline**,
**GPU utilization** (GpuIdleTracker busy-vs-wall), and end-to-end **latency**,
plus cache hit-rate and prefetch accuracy, OFF vs ON on two identical MoE
models.

### GPU (stream overlap — where the win lives)

On CUDA the speculative stack+cast builds run on the prefetch stream while the
previous layer computes. Every layer that would otherwise stall on a cache
miss instead starts with its weights resident; the per-layer stall is removed
from the critical path.

### CPU (no stream overlap — honest limitation)

CPU cannot overlap, so the prefetcher is a no-op by default. Forcing it
(`METIS_LAYER_PREFETCH_FORCE_CPU=1` or `--force-cpu-prefetch`) turns it into a
*synchronous* speculative warm-up: with the training loop's per-step cache
invalidation, the next layers' entries are rebuilt during the current layer,
and — on this machine — every layer's forward time drops while cache hit-rate
jumps:

| metric | OFF | ON |
|---|---|---|
| cache hit rate (cold start/step) | ~8–15% | **~68–70%** |
| prefetch accuracy (`useful/prefetched`) | — | **100%** |
| per-layer forward (mean, layers 1–3) | 2.3–3.1 ms | **2.0–2.4 ms** |
| mean step latency | ~10–11 ms | ~10.5–12 ms |

Because CPU serializes the builds (the same work as the misses they replace,
just moved earlier, plus the prefetch bookkeeping), the step-level latency on
CPU is ~neutral-to-slightly-worse (~4%). That is not a regression on CPU —
prefetch is off by default there — and the GPU win comes from *hiding* those
builds under the previous layer's compute.

On a real GPU the measurement of record is:

```bash
python benchmarks/benchmark_layer_prefetch.py --device cuda --steps 20
```

which prints the timeline + GPU utilization + latency tables and writes
JSON/Markdown to `benchmarks/results/`.

## Reproduce

```bash
# unit tests (correctness + cache + record flow + gating)
python -m pytest tests/test_layer_prefetch.py -q

# GPU benchmark — timeline, GPU utilization, latency, hit-rate, accuracy
python benchmarks/benchmark_layer_prefetch.py --device cuda --steps 20

# CPU: demonstrate the speculative warm-up under cache pressure
python benchmarks/benchmark_layer_prefetch.py --device cpu --force-cpu-prefetch \
    --invalidate-each-step

# parity of the underlying MoE engine + expert cache still holds
python benchmarks/verify_moe_parity.py
python benchmarks/verify_expert_cache_parity.py
```
