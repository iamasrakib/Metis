# Overlapped training pipeline in Metis

Metis software-pipelines the training loop so disk I/O, tokenization, CPU
preprocessing, H2D copies, GPU compute, and checkpoint writes overlap rather
than serialise. The pipeline is **on by default** (`use_pipeline=True`) and
opt-out via `--no-pipeline` / `use_pipeline=False` restores the exact serial
path bit-for-bit.

## Design

Four primitives in `metis/pipeline.py` (threads only, not worker processes —
Windows-safe; all CUDA ops issued from the main thread):

### ThreadPrefetcher (disk I/O + tokenization + CPU preprocessing)

A daemon thread runs the DataLoader ahead into a bounded `queue.Queue`. The
training loop pulls a full step's worth of micro-batches with no blocking on
`next(iter)`, so the per-sample disk page-fault / tokenize / preprocess time
is hidden behind GPU compute. With `pin=True` (CUDA only) each batch's tensor
fields are page-locked in the producer thread, so the main-thread H2D staging
issues pure non-blocking copies — the synchronous pin-memory copy never lands
on the hot path (the DataLoader's own `pin_memory=True` already does this for
the default path; `pin` covers custom/unpinned loaders).

PyTorch's DataLoader `num_workers>0` crashes on Windows under spawn/IPC, so the
prefetcher uses a **background thread** (which shares the process's memory) and
works identically on all platforms.

### GpuBatchStager (H2D copies on a copy stream)

A dedicated `torch.cuda.Stream` copy stream + a ring of device staging buffers
(`pipeline_buffer_depth`, default 3). The producer issues `non_blocking`
pinned→device copies for micro-batch `i+1` on the copy stream *before* the
forward of micro-batch `i`, so the H2D transfer runs concurrently with compute.
Events gate both directions:

- copy stream: `wait_event(done_event[prev use of this slot])` → copy → record
  `copy_event`.
- compute stream: `wait_event(copy_event)` → forward/backward → record
  `done_event`.

**Cross-step overlap:** the eager training loop pulls the *next* step and
stages its first micro-batch immediately after the current step's last
`mark_done`, so that H2D runs during the optimizer step / EMA / logging tail
instead of stalling the next step. The CUDA-graph path stages the next step's
static slots during the current graph replay.

The stager handles `(x, y)` tuples and `PackedBatch`-style objects (attention
mask, position ids) uniformly. On CPU it is a FIFO passthrough (stage→queue,
device→pop) that preserves staging order — a single overwritten `_current`
slot used to hand out the *next* batch, breaking order.

### AsyncCheckpointer (checkpoint D2H + disk writes)

Two levels of overlap:

- **Async D2H snapshot** (`submit_async`): the `state_dict` → host transfer is
  issued on a dedicated checkpoint copy stream, overlapped with the next step's
  forward/backward instead of stalling it. The caller records a `compute_done`
  event after the optimizer/EMA update; the checkpoint stream waits on it,
  snapshots via `non_blocking` copies, and records a `d2h_done` event. The
  writer thread calls `d2h_done.synchronize()` (waits only for that event, not
  all GPU work) before pickling. `wait_pending()` makes the compute stream wait
  for `d2h_done` before the *next* `optimizer.step()`, so the snapshot can never
  read a half-updated weight — no torn checkpoints.
- **Background disk write**: a daemon thread owns `torch.save` (atomic: write
  `.tmp`, then `os.replace`). `flush()` blocks until all writes land; called
  before the final save and on shutdown.

On CPU (no D2H to overlap) `submit_async` hands the dict straight to the
writer thread.

### GpuIdleTracker (GPU idle measurement + per-stage attribution)

CUDA-event measurement of GPU busy vs wall time per step:

```
idle% = 1 − (gpu_busy_ms / wall_ms)
```

`begin()` / `end()` bracket each step. `tick(name)` accumulates wall time per
stage so the **per-stage breakdown** shows *where* the wall time goes:

| stage | meaning |
|---|---|
| `data_wait` | blocked on `prefetcher.next_step()` (loader not ready) |
| `h2d` | issuing H2D staging |
| `compute` | forward/backward wall time |
| `optimizer` | unscale/clip/step |
| `checkpoint` | checkpoint snapshot + submit |
| `other` | remainder |

`stats()` aggregates both the idle % and the per-stage wall totals. On CPU
there is no CUDA busy time (`gpu_ms=0`), but the per-stage wall breakdown is
still recorded — it is the tool for seeing the overlap win on any machine.

## Wiring into the training loop

`metis/training.py`:

```
pipeline OFF:
  serial:  data_iter → next() → .to(device) → forward/backward → sync
pipeline ON:
  1. seed: prefetcher.next_step(); stager.stage(first micro-batch)
  2. eager path: stage micro[i+1] → device(micro[i]) → forward/backward
     → mark_done; then pull next step + stage its first micro-batch so its
     H2D overlaps the optimizer/EMA tail below (cross-step overlap)
  3. graph path: train_step(prefetch_next=nxt) → wait staged copy → replay
     the static slots → stage nxt on the copy stream during the replay
  4. checkpointer.wait_pending() before scaler.step (tear-free snapshot)
  5. save point: compute_done.record() → build_checkpoint_raw →
     submit_async (D2H on copy stream, writer thread owns pickle+disk)
  6. idle_tracker.begin/tick(...)/end wraps each step
```

## Configuration

| Field | Default | Description |
|---|---|---|
| `use_pipeline` | `True` | Master switch for prefetch + async H2D + async checkpoints |
| `prefetch_depth` | `2` | Steps read ahead by the prefetch thread |
| `pipeline_buffer_depth` | `3` | H2D staging ring depth (copy-stream slack) |
| `async_checkpoint` | `True` | Write checkpoints on a background thread |

CLI flags (`metis train`):

```
--no-pipeline               Disable the overlapped pipeline
--prefetch-depth N          Steps read ahead (default: 2)
--pipeline-buffer-depth N   H2D staging ring depth (default: 3)
--no-async-checkpoint       Write checkpoints synchronously
```

## Bit-identical parity

The overlapped path is **bit-identical** to the serial path for the same model,
batches, and RNG state. Verified by `benchmarks/verify_pipeline_parity.py`
(checks: losses over N steps, stager ring/FIFO correctness, prefetch order +
restart semantics, async checkpoint atomicity) and by the unit
`tests/test_pipeline.py`.

Note: the parity check re-seeds the global CUDA RNG before each model's
forward to isolate the pipeline machinery from RNG-sequence divergence that
would otherwise arise from interleaving two models' forwards on the same global
RNG.

## Measured results

### GPU (RTX 2050, torch 2.6.0+cu124 — saved results)

MoE (d=128, 4 layers, 8 experts, top-2, gradient_accumulation_steps=4),
dropout=0.0; 25 steps per mode.

| scenario | serial idle | overlapped idle | speedup |
|---|---:|---:|---:|
| fast disk (cached mmap) | 0.35% | 0.02% | **1.08×** |
| simulated slow disk (5 ms/batch) | 0.32% | 0.02% | **1.18×** |

On fast hardware the DataLoader fetch is already quick, so GPU idle is low
(~0.3 %); the pipeline removes it completely. The win grows with slower disks,
larger datasets, or CPU-heavy tokenizers.

### CPU (this repo's dev machine, torch 2.13.0, 8 steps, fp32)

The `--stage` flag makes the overlap measurable even without a GPU. Fast disk
(in-memory char tokenizer):

| mode | wall (ms) | data_wait | compute | optimizer | tok/s |
|---|---:|---:|---:|---:|---:|
| serial | 2922 | 6 ms | 2843 ms | 74 ms | 5606 |
| overlapped | 2869 | 2 ms | 2797 ms | 70 ms | 5711 |
| **delta** | **1.02×** | 6→2 ms | | | |

Simulated slow disk / tokenizer (5 ms per batch, `--slow-ms 5`):

| mode | wall (ms) | data_wait | compute | optimizer | tok/s |
|---|---:|---:|---:|---:|---:|
| serial | 3280 | 180 ms | 3021 ms | 79 ms | 4995 |
| overlapped | 2894 | 22 ms | 2804 ms | 68 ms | 5660 |
| **delta** | **1.13×** | 180→22 ms | | | |

With a fast in-memory loader the data_wait is already small, so the pipeline's
wall-time win is modest on CPU; the prefetch thread hides it almost entirely
once the loader has any real latency (5 ms/batch → 180 ms → 22 ms of data
wait). Losses are bit-identical in every run (mean 2.5739 both modes).

### Checkpoint stress (per-step save, CPU)

| variant | ms/step |
|---|---:|
| no checkpointing | ~240 |
| synchronous `torch.save` | ~249 |
| `AsyncCheckpointer` (async D2H + writer thread) | ~248 |

Async checkpointing saves a few ms/step vs synchronous on CPU (the tiny model
pickles fast); the D2H-overlap win is larger on GPU where the snapshot would
otherwise stall the compute stream.

## Reproduce

```bash
# correctness + parity
python benchmarks/verify_pipeline_parity.py --device cuda
python -m pytest tests/test_pipeline.py -q

# idle + per-stage attribution (fast disk vs slow disk)
python benchmarks/benchmark_pipeline_overlap.py --device cuda --stage
python benchmarks/benchmark_pipeline_overlap.py --device cuda --slow-ms 5

# checkpoint overlap stress
python benchmarks/benchmark_pipeline_overlap.py --device cuda --checkpoint-stress

# also: the pipeline and parity harness are CPU-safe
python benchmarks/benchmark_pipeline_overlap.py --device cpu --stage
python benchmarks/verify_pipeline_parity.py
```
