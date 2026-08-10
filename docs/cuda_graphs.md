# CUDA Graphs in Metis — compatibility report

Metis can capture the entire gradient-accumulation training iteration of
[`metis/training.py`](../metis/training.py) — `N` micro-batch forward+backward
passes, loss scaling, gradient clipping and the optimizer update — as a single
CUDA graph and replay it once per step. The implementation lives in
[`metis/cuda_graphs.py`](../metis/cuda_graphs.py) behind the
`CUDAGraphStep` class.

```
┌──────────────────────────────────────────────────────────────────────────┐
│ one training iteration                                                     │
│                                                                            │
│  eager host work                captured region (ONE graph)               │
│  ┌──────────────┐               ┌───────────────────────────────────┐     │
│  │ fetch N micro│  copy_ into   │  for i in 0..N-1:                 │     │
│  │ batches      │ ────────────▶ │    slot_x[i], slot_y[i]           │     │
│  └──────────────┘               │    forward  (autocast bf16/fp16,  │     │
│  zero_grad(in place)            │      cache_enabled=False)         │     │
│  scale_buf ← scaler._scale      │    loss ÷ N                      │     │
│  graph.replay() ───────────────▶│    loss_buf += loss.fp64         │     │
│  loss = loss_buf.item()         │    scaled = loss × scale_buf     │     │
│  scaler.unscale_ → clip →       │    scaled.backward()  ─────────────┐  │
│  scaler.step → scaler.update    │                                    │  │
│  EMA update (host)              │◀───────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
```

## Requirements coverage

| Requirement | How Metis meets it |
|---|---|
| **Capture one full training iteration** | One `torch.cuda.CUDAGraph` contains all `N` micro-batch forwards, the `loss/N` divide, an fp64 loss accumulator, the scale multiply, and all `backward()` passes. |
| **Replay captured graph** | `graph.replay()` per step; the next iteration's data is copied into the static input slots on the stream immediately before replay. |
| **Handle optimizer** | The update (AdamW, fused when available) runs eagerly after replay, so per-step LR scheduling, weight decay and optimizer state are untouched. Grad buffers are zeroed **in place** (`set_to_none=False`) to keep capture-time pool addresses valid. |
| **Handle gradient scaling** | The scale factor is fed into the graph as a **device tensor** (`scale_buf`) synced from `scaler._scale` before every replay — a growing/backing-off scale is never baked into the capture. `unscale_ → step → update` run eagerly, exactly as the eager path does them, including overflow handling (step skipped, scale halves, training recovers). |
| **Handle mixed precision** | The forward is captured under `torch.autocast` (bf16 on Ampere+, else fp16) with `cache_enabled=False`. Autocast's weight-cast *caching* would pin stale fp16/bf16 copies into the graph; disabling it turns the cast into a captured op that re-reads the current master weights. |
| **Fall back automatically** | Capture is attempted lazily. Any failure — CPU, missing CUDA, MoE, DDP, `torch.compile`, a capture-time `RuntimeError` — degrades to the eager loop with a logged `reason`. |

## Numerical parity

**With `dropout=0`, the graph path is bit-identical to the eager path.** The
parity suite
([`benchmarks/verify_cuda_graphs_parity.py`](../benchmarks/verify_cuda_graphs_parity.py))
confirms, over multiple steps: `loss_accum` identical, `grad_norm` identical,
weights identical, gradients identical — all with `torch.equal`.

**With `dropout>0`, masks come from a graph-private Philox RNG.** PyTorch's
CUDA-graph capture re-advances the RNG offset on every replay, so dropout
consumes a **fresh mask each step** (verified: two replays of the same inputs
give different losses; the masks are not frozen at the capture-time mask). The
concrete mask sequence is *not* reproducible from `torch.cuda.set_rng_state`,
so per-step values differ from an eager run by the normal dropout variation
(mean `|Δloss| ≈ 1e-2` for `dropout=0.3` on the test model) while remaining
statistically identical. This is a documented CUDA-graph property, not a bug —
see [Limitations](#limitations).

The fp64 loss accumulator reproduces the eager loop's Python-double sum
exactly, so logged `loss_accum` values match.

## Compatibility matrix

Verified on `torch 2.6.0+cu124` / `NVIDIA GeForce RTX 2050 (sm_86, Ampere)`:

| Configuration | Behaviour |
|---|---|
| CUDA GPU, standard dense model | ✅ captured & replayed |
| fp16 AMP (`GradScaler`) | ✅ captured (graph multiplies by `scale_buf`) |
| bf16 AMP (`GradScaler`) | ✅ captured (recommended on Ampere+) |
| fp32 (no AMP) | ✅ captured (`autocast` disabled; scaler disabled) |
| Dropout > 0 | ✅ captured; fresh mask per replay |
| `gradient_accumulation_steps` > 1 | ✅ captured — all micro-batches in one graph |
| Gradient checkpointing | ⚠️ **off inside the graph** (see below); eager fallback keeps it on |
| Autocast weight-cast caching | ⚠️ **off inside the graph** (`cache_enabled=False`); results unchanged |
| MoE | ⚠️ **falls back to eager** — routing shapes are data-dependent (`nonzero`, `bucketize`) and cannot be captured safely |
| DDP | ⚠️ **falls back to eager** — NCCL collectives are captured only with special handling |
| `torch.compile` | ⚠️ **falls back to eager** — nested graph capture unsupported |
| CPU / no CUDA | ⚠️ **falls back to eager** |
| torch < 2.0 (no `torch.cuda.CUDAGraph`) | ⚠️ **falls back to eager** |
| Fixed `micro_batch_size` / `max_seq_len` (drop-last) | ✅ required — graphs forbid dynamic shapes |

### Why gradient checkpointing is off inside the graph

`torch.utils.checkpoint` saves/restores the CUDA RNG state around each segment,
which calls `CUDAGeneratorImpl::current_seed` — an operation that is not
permitted during stream capture. Inside the graph the RNG restriction does not
cost anything: the graph pool already pins the activations in memory, so the
memory checkpointing saves would be held by the pool anyway. **Enabling CUDA
graphs therefore switches the training step to non-checkpointed execution.**
The eager fallback path keeps checkpointing on, matching the original loop.
On the benchmark model the graph's total footprint (`26.2 MB`) is only
`+5.0 MB` over the checkpointed eager peak (`21.2 MB`), because the pool pins
the iteration's activations but the `N`-micro-batch structure is flattened
into one replay.

## Benchmark (RTX 2050, bf16, d_model=128 · 4 layers · 256 ctx · B=8 · N=4)

| Strategy | ms/step | Peak act. mem | Notes |
|---|---:|---:|---|
| eager + checkpointing (baseline) | 128.0 | 21.2 MB | the original `metis train` loop |
| eager, checkpointing off | 77.0 | 72.9 MB | same compute the graph runs |
| **CUDA graph replay** | **40.7** | 26.2 MB total | one replay per iteration (pool 19.6 + replay 6.6) |

* **3.15× faster than the baseline** (includes the checkpointing-off effect)
* **1.89× faster than the same compute run eagerly** (the pure CUDA-graph win —
  kernel launches and per-micro-batch host/device syncs removed)

The win grows with the accumulation budget because a single replay replaces `N`
eager micro-batch loops:

| `gradient_accumulation_steps` | eager (ms) | graph (ms) | speedup |
|---|---:|---:|---:|
| 1 | 33.7 | 12.1 | 2.79× |
| 2 | 66.0 | 21.6 | 3.06× |
| 4 | 128.6 | 40.7 | 3.16× |
| 8 | 250.8 | 78.7 | 3.19× |

Run it yourself: `python benchmarks/benchmark_cuda_graphs.py` (writes JSON +
Markdown under `benchmarks/results/`). Re-run the parity suite with
`python benchmarks/verify_cuda_graphs_parity.py`.

## Usage

```bash
metis train --dataset data/sample.txt --preset tiny       # graphs auto-enabled on CUDA
metis train ... --no-cuda-graphs                          # opt out
```

* Config flag: `use_cuda_graphs: bool = True` (`metis/config.py`).
* The training log prints the runtime decision:
  `CUDA Graphs: active (graph replay)` or
  `CUDA Graphs: inactive — eager (<reason>)`.

## Limitations

1. **Dropout masks are not reproducible** from the host RNG state under replay —
   each replay draws fresh masks from the graph's own Philox tracking. For
   bit-exact reproducibility set `dropout=0` (typical for production LLM
   training); otherwise the run is statistically equivalent, not identical.
2. **No dynamic shapes.** `micro_batch_size` and `max_seq_len` are fixed by
   capture; the DataLoader's `drop_last=True` guarantees a uniform shape.
3. **Memory held for the whole iteration.** The pool pins the iteration's
   activation high-water mark; on this model the total was only `+5.0 MB` over
   the checkpointed baseline, but very deep / high-context models should
   re-check with `--mode memory`.
4. **MoE, DDP, torch.compile, CPU** run eager — those fallbacks are silent apart
   from the one-time log line.
5. **One capture cost per run** (a few warmup steps, then a state snapshot is
   restored so the capture itself is invisible to training).
