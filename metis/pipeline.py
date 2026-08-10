"""
Metis — Overlapped training pipeline
=====================================
Software-pipelines the training loop so disk I/O, tokenization, CPU
preprocessing, H2D copies, GPU compute, and checkpoint writes overlap instead
of serialising behind one another.

Four primitives (threads only, no subprocesses — worker processes crash on
Windows under spawn/IPC; all CUDA ops are issued from the main thread):

* ``ThreadPrefetcher`` — a daemon thread runs the DataLoader ahead into a
  bounded queue, hiding disk + tokenization + CPU preprocessing behind GPU
  compute.
* ``GpuBatchStager``  — double-buffered H2D: a dedicated copy stream transfers
  the next micro-batch (pinned host -> device, ``non_blocking``) while the
  compute stream runs the current one. Events gate both directions.
* ``AsyncCheckpointer`` — background-thread ``torch.save`` so checkpoint disk
  writes overlap training (state is snapshotted, the thread owns the write).
* ``GpuIdleTracker``  — CUDA-event measurement of GPU busy vs wall time per
  step (the "GPU idle time" metric).

Everything degrades gracefully on CPU (staging becomes a passthrough, the
prefetcher still hides loader I/O).
"""

from __future__ import annotations

import os
import queue
import threading
import time
from dataclasses import dataclass, field

import torch

# ── Prefetching ───────────────────────────────────────────────────────────────

class ThreadPrefetcher:
    """Run a DataLoader ahead of the training loop on a daemon thread.

    The producer thread consumes the loader and stages complete steps
    (``micro_batches`` batches each) into a bounded queue. The training loop
    pulls a full step with ``next_step()`` — no loader fetch on the hot path,
    so disk I/O / tokenization / CPU preprocessing hide behind GPU compute.

    With ``pin=True`` (CUDA only) each batch's tensor fields are page-locked
    in the producer thread, so the main-thread H2D staging issues pure
    non-blocking copies — the synchronous pin-memory copy never lands on the
    hot path. The DataLoader's own ``pin_memory=True`` already does this for
    the default path; ``pin`` covers custom/unpinned loaders.

    Consumption semantics match ``_fetch_micro_batches``: the loader is
    restarted on exhaustion, so every epoch is walked exactly once.
    """

    def __init__(
        self,
        loader,
        micro_batches: int = 1,
        prefetch_depth: int = 2,
        pin: bool = False,
    ) -> None:
        self.loader = loader
        self.micro_batches = max(1, int(micro_batches))
        self.pin = pin and torch.cuda.is_available()
        # The queue holds whole steps, so `prefetch_depth` = steps read ahead.
        self._queue: queue.Queue = queue.Queue(maxsize=max(1, prefetch_depth))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    def start(self) -> ThreadPrefetcher:
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._error = None
            self._thread = threading.Thread(
                target=self._produce, name="metis-prefetch", daemon=True
            )
            self._thread.start()
        return self

    def _produce(self) -> None:
        try:
            it = iter(self.loader)
            while not self._stop.is_set():
                step = []
                for _ in range(self.micro_batches):
                    try:
                        step.append(next(it))
                    except StopIteration:
                        it = iter(self.loader)
                        step.append(next(it))
                if self.pin:
                    step = [_pin_batch(b) for b in step]
                self._queue.put(step)
        except BaseException as e:  # surface loader crashes to the main thread
            self._error = e
            self._stop.set()

    def next_step(self):
        """Return the next step's ``micro_batches`` batches (blocking).

        Polls with a short timeout and re-checks ``self._error`` each cycle so a
        producer crash that lands *after* the error check (or before the first
        ``put``) is surfaced instead of leaving the consumer blocked on
        ``get()`` forever.
        """
        while True:
            if self._error is not None:
                raise RuntimeError(f"prefetch worker failed: {self._error}") from self._error
            try:
                return self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

    def stop(self) -> None:
        """Signal the producer to stop and wait for it to wind down.

        Also drains the queue so a producer blocked on ``put`` can exit.
        """
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            # Unblock a `put` parked on a full queue.
            try:
                while True:
                    self._queue.get_nowait()
            except queue.Empty:
                pass
            self._thread.join(timeout=5.0)

    def __enter__(self) -> ThreadPrefetcher:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


# ── H2D staging on a copy stream ──────────────────────────────────────────────

def _tensor_fields(batch) -> dict[str, torch.Tensor]:
    """Flatten a batch into ``{key: tensor}`` (x, y + extra tensor kwargs).

    Accepts ``(x, y)`` tuples and objects exposing ``input_ids``/``labels``
    plus a ``model_kwargs`` dict (e.g. ``PackedBatch``).
    """
    fields: dict[str, torch.Tensor] = {}
    if isinstance(batch, (tuple, list)):
        if len(batch) >= 2:
            fields["x"] = batch[0]
            fields["y"] = batch[1]
        extra = batch[2] if len(batch) >= 3 and isinstance(batch[2], dict) else {}
    else:
        fields["x"] = batch.input_ids
        fields["y"] = batch.labels
        extra = getattr(batch, "model_kwargs", {})
    for key, t in extra.items():
        if isinstance(t, torch.Tensor):
            fields[key] = t
    return fields


def _reassemble(staged: dict) -> tuple:
    """Rebuild ``(x, y, extra)`` from staged device tensors."""
    x = staged.get("x")
    y = staged.get("y")
    extra = {k: v for k, v in staged.items() if k not in ("x", "y")}
    return x, y, extra


def _pin_t(t: torch.Tensor) -> torch.Tensor:
    """Pin a CPU tensor (no-op if already pinned / on device)."""
    if t.is_cuda or t.is_sparse or t.is_pinned():
        return t
    try:
        return t.pin_memory()
    except (TypeError, RuntimeError):
        return t


def _pin_batch(batch):
    """Pin a batch's tensor fields, preserving its tuple form.

    ``(x, y)`` / ``(x, y, extra)`` tuples are returned with pinned tensors;
    object batches are returned unchanged (their loader already pinned them).
    """
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        pinned = (_pin_t(batch[0]), _pin_t(batch[1]))
        if len(batch) >= 3 and isinstance(batch[2], dict):
            extra = {k: _pin_t(v) if isinstance(v, torch.Tensor) else v
                     for k, v in batch[2].items()}
            return (*pinned, extra)
        return pinned
    return batch


class GpuBatchStager:
    """Double-buffer H2D copies on a dedicated copy stream.

    Holds a 2-deep ring of device staging buffers. The producer issues
    ``non_blocking`` pinned->device copies on ``self.copy_stream``; the
    consumer (compute stream) waits on the copy-complete event before using a
    slot. Because the copy for micro-batch ``i+1`` is issued *before* the
    forward of micro-batch ``i``, the transfer runs concurrently with compute.

    Correctness on the ring: before overwriting a slot, the copy stream waits
    for the compute-done event of that slot's previous use (its forward +
    backward), so a tensor is never clobbered while the GPU still reads it.

    On CPU this is a pass-through: ``stage``/``device`` return the CPU tensors
    unchanged (the prefetcher alone hides loader I/O).
    """

    def __init__(self, device: str, depth: int = 2) -> None:
        self._device = torch.device(device)
        self.is_cuda = self._device.type == "cuda"
        self.depth = max(2, int(depth))
        self._slots: list[dict | None] = [None] * self.depth
        self._copy_event: list = [None] * self.depth  # copy done -> device ready
        self._done_event: list = [None] * self.depth  # compute done -> slot free
        if self.is_cuda:
            self.copy_stream = torch.cuda.Stream(device=self._device)
        else:
            self.copy_stream = None
        self._stage_seq = 0  # monotonic producer counter (slot = counter % depth)
        self._ready: list = []  # staged-but-not-yet-consumed slot indices, in order
        self._current = None  # CPU passthrough of the just-staged batch
        self._consuming = None  # slot index handed out by device() awaiting mark_done
        # CPU passthrough FIFO: stage() appends, device() pops.  On CPU the
        # ring is not used (no async copies), but the staging order must be
        # preserved — a single ``_current`` slot would hand out the *next*
        # batch because stage(batch[i+1]) is called before device() consumes
        # batch[i].
        self._cpu_queue: list = []

    # ── producer side ────────────────────────────────────────────────────
    def stage(self, batch) -> None:
        """Issue the async H2D copy of ``batch`` into the next free slot.

        ``batch`` is a ``(x, y)`` pair or a batch object with ``input_ids`` /
        ``labels`` / ``model_kwargs``. Tensor fields are pinned and copied.
        Must be called before :meth:`device` consumes the slot, and *before*
        the forward whose compute it should overlap.
        """
        if not self.is_cuda:
            # CPU passthrough: keep the split fields so device() can return the
            # same (x, y, extra) contract as the CUDA path.
            tensors = _tensor_fields(batch)
            self._cpu_queue.append(
                (
                    tensors.get("x"),
                    tensors.get("y"),
                    {k: v for k, v in tensors.items() if k not in ("x", "y")},
                )
            )
            return

        slot = self._stage_seq % self.depth
        # Previous use of this slot must be fully consumed (fwd+bwd) first.
        if self._done_event[slot] is not None:
            self.copy_stream.wait_event(self._done_event[slot])

        tensors = {k: _pin_t(t) for k, t in _tensor_fields(batch).items()}
        staged = self._slots[slot]
        if staged is None:
            staged = {
                key: torch.empty_like(t, device=self._device)
                for key, t in tensors.items()
            }
            self._slots[slot] = staged
        else:
            # Variable-length batches (a trailing partial batch, a final smaller
            # packed bin) must regrow the slot buffer, not crash the copy_ with
            # a size mismatch. Fresh buffers keep the next copy_ non-blocking.
            for key, t in tensors.items():
                if key not in staged or staged[key].shape != t.shape:
                    staged[key] = torch.empty_like(t, device=self._device)

        with torch.cuda.stream(self.copy_stream):
            for key, t in tensors.items():
                staged[key].copy_(t, non_blocking=True)  # pinned -> device
            self._copy_event[slot] = torch.cuda.Event()
            self._copy_event[slot].record(self.copy_stream)

        self._stage_seq += 1
        self._ready.append(slot)

    # ── consumer side ────────────────────────────────────────────────────
    def device(self) -> tuple:
        """Return the staged ``(x, y, extra)`` device tensors for this batch.

        Blocks the compute stream (via event) until the async copy that
        produced this slot has completed. Call :meth:`mark_done` after the
        forward/backward so the slot can be reused.
        """
        if not self.is_cuda:
            if not self._cpu_queue:
                raise RuntimeError(
                    "GpuBatchStager.device() called with no staged batch (CPU path)"
                )
            return self._cpu_queue.pop(0)  # FIFO — the oldest staged batch
        idx = self._ready.pop(0)  # oldest staged-not-yet-consumed slot
        staged = self._slots[idx]
        torch.cuda.current_stream().wait_event(self._copy_event[idx])
        self._consuming = idx
        return _reassemble(staged)

    def mark_done(self) -> None:
        """Record the compute-done event for the last consumed slot.

        Must be called *after* the forward/backward that used the slot, so the
        copy stream knows when it may overwrite the slot with the next batch.
        """
        if not self.is_cuda or self._consuming is None:
            return
        idx = self._consuming
        self._done_event[idx] = torch.cuda.Event()
        self._done_event[idx].record(torch.cuda.current_stream())
        self._consuming = None


# ── Async checkpointing ───────────────────────────────────────────────────────

def _is_cuda_tensor(v) -> bool:
    return isinstance(v, torch.Tensor) and v.is_cuda


def _contains_cuda_tensor(obj) -> bool:
    """True if ``obj`` holds any CUDA tensor, recursing into nested containers.

    Checkpoint dicts from :func:`metis.training.build_checkpoint_raw` store the
    model/optimizer/EMA state as *nested* dicts, so a shallow top-level scan
    would miss every CUDA tensor and silently route the write onto the unsafe
    CPU path (live GPU weights pickled while the optimizer mutates them).
    """
    if isinstance(obj, dict):
        return any(_contains_cuda_tensor(v) for v in obj.values())
    if isinstance(obj, (tuple, list)):
        return any(_contains_cuda_tensor(v) for v in obj)
    return _is_cuda_tensor(obj)


def _to_cpu_deep(obj, non_blocking: bool = True):
    """Recursively clone CUDA tensors to CPU, preserving container structure.

    Non-tensor leaves (ints, floats, strings, plain dicts) pass through
    untouched so the checkpoint metadata survives the snapshot.
    """
    if isinstance(obj, dict):
        return {k: _to_cpu_deep(v, non_blocking) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu_deep(v, non_blocking) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_cpu_deep(v, non_blocking) for v in obj)
    if _is_cuda_tensor(obj):
        return obj.detach().to("cpu", non_blocking=non_blocking)
    return obj


class AsyncCheckpointer:
    """Background-thread ``torch.save`` with a bounded pending queue.

    ``submit_async`` queues a checkpoint and issues its D2H snapshot on a
    dedicated copy stream, so the state_dict → host transfer overlaps the next
    training step instead of stalling it (the pickle + disk write then run on
    the daemon writer thread). The caller must call :meth:`wait_pending`
    before the *next* weight mutation so the snapshot can never read a
    half-updated weight. ``submit`` is the synchronous-snapshot fallback.
    At most ``max_pending`` writes queue up; ``flush`` blocks until all are
    on disk (called before exit / final save).
    """

    def __init__(self, max_pending: int = 1) -> None:
        self._queue: queue.Queue = queue.Queue(maxsize=max(1, max_pending))
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._writer, name="metis-checkpoint", daemon=True
        )
        self._thread.start()
        self._errors: list[BaseException] = []
        self._ckpt_stream: torch.cuda.Stream | None = None
        self._pending_d2h = None  # cuda.Event; main thread waits before next mutation

    def _writer(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:  # sentinel
                return
            path, checkpoint, d2h_event = item
            try:
                # Wait for the async D2H snapshot before pickling host tensors.
                # Event-level synchronize only waits for the snapshot copies,
                # not all GPU work, so the main thread's compute continues.
                if d2h_event is not None and torch.cuda.is_available():
                    d2h_event.synchronize()
                tmp = f"{path}.tmp"
                torch.save(checkpoint, tmp)
                os.replace(tmp, path)
            except BaseException as e:
                self._errors.append(e)

    def _async_snapshot(self, checkpoint: dict, compute_done=None) -> tuple:
        """Clone CUDA tensors to CPU on the checkpoint stream; return (dict, event).

        ``compute_done`` is the CUDA event marking the weights final. When it is
        ``None`` the copy stream waits on an event recorded on the caller's
        *current* stream instead, so whatever compute preceded the submit is
        finalized before the snapshot reads the weights.
        """
        if self._ckpt_stream is None:
            self._ckpt_stream = torch.cuda.Stream(device="cuda")
        stream = self._ckpt_stream
        if compute_done is not None:
            stream.wait_event(compute_done)
        else:
            current_done = torch.cuda.Event()
            current_done.record(torch.cuda.current_stream())
            stream.wait_event(current_done)
        with torch.cuda.stream(stream):
            cpu_ckpt = _to_cpu_deep(checkpoint)
            event = torch.cuda.Event()
            event.record(stream)
        return cpu_ckpt, event

    def submit_async(self, path: str, checkpoint: dict, compute_done=None) -> None:
        """Queue a checkpoint, overlapping the D2H snapshot with compute.

        ``compute_done`` is the CUDA event recorded after the optimizer/EMA
        update that finalises the weights being snapshotted (None → the
        snapshot waits on the current stream instead). CUDA tensors — found
        anywhere in the checkpoint, including nested ``state_dict`` dicts — are
        cloned to CPU asynchronously on a dedicated copy stream; the writer
        thread pickles the CPU clone once the copy completes. Without the
        snapshot, pickling live GPU weight storage concurrently with
        ``optimizer.step`` would save a torn checkpoint.
        """
        if not torch.cuda.is_available() or not _contains_cuda_tensor(checkpoint):
            # CPU path: no D2H to overlap — hand the dict straight to the writer.
            self._queue.put((path, checkpoint, None))
            return
        cpu_ckpt, event = self._async_snapshot(checkpoint, compute_done)
        self._pending_d2h = event
        self._queue.put((path, cpu_ckpt, event))

    def submit(self, path: str, checkpoint: dict) -> None:
        """Queue ``checkpoint`` for asynchronous ``torch.save`` to ``path``.

        The snapshot must already be on CPU (e.g. via ``build_checkpoint``'s
        synchronous ``detach().cpu()``). Prefer :meth:`submit_async` on CUDA.
        """
        self._queue.put((path, checkpoint, None))

    def wait_pending(self) -> None:
        """Make the compute stream wait for the in-flight D2H snapshot.

        Call before the next ``optimizer.step()`` (or any in-place weight
        mutation) so the snapshot cannot race a weight update. A no-op when no
        snapshot is pending (the usual case between save steps).
        """
        if self._pending_d2h is not None:
            torch.cuda.current_stream().wait_event(self._pending_d2h)
            self._pending_d2h = None

    def pending(self) -> int:
        return self._queue.qsize()

    def flush(self) -> None:
        """Wait until all queued checkpoints are written (or the writer died)."""
        while not self._queue.empty():
            if self._errors:
                break
            time.sleep(0.01)
        if self._errors:
            raise RuntimeError(
                f"async checkpoint writer failed: {self._errors[0]}"
            ) from self._errors[0]

    def close(self) -> None:
        # Always send the sentinel so the writer thread winds down even when
        # flush() raises (disk-write error) — otherwise the thread spins on
        # queue.get() forever and a re-close() flushes a stuck writer.
        try:
            self.flush()
        finally:
            self._queue.put(None)
            self._thread.join(timeout=5.0)

    def __enter__(self) -> AsyncCheckpointer:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ── GPU idle measurement ──────────────────────────────────────────────────────

@dataclass
class GpuIdleTracker:
    """Measure GPU busy vs wall time per step with CUDA events.

    Usage::

        tracker = GpuIdleTracker(device)
        tracker.begin()
        <training step>            # interleave tracker.tick("stage")
        tracker.end()
        tracker.stats()            # aggregate + per-stage breakdown
    """

    device: str
    enabled: bool = True
    _wall_t0: float = field(default=0.0, repr=False)
    _start: object | None = field(default=None, repr=False)
    _end: object | None = field(default=None, repr=False)
    _tick_t: float = field(default=0.0, repr=False)
    _stages: dict = field(default_factory=dict, repr=False)
    rows: list = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self.is_cuda = torch.device(self.device).type == "cuda" and self.enabled

    def begin(self) -> None:
        if not self.enabled:
            return
        self._wall_t0 = time.perf_counter()
        self._tick_t = self._wall_t0
        self._stages = {}
        if self.is_cuda:
            self._start = torch.cuda.Event(enable_timing=True)
            self._start.record()

    def tick(self, name: str) -> None:
        """Accumulate wall time spent in ``name`` since the previous tick.

        Call ``begin()`` once per step, then ``tick("data_wait")`` around
        loader pulls, ``tick("h2d")`` around H2D staging, ``tick("compute")``
        around forward/backward, ``tick("checkpoint")`` around checkpoint
        submission, ``tick("optimizer")`` around the optimizer step, and a
        final ``tick("other")`` — the breakdown shows which stage the overlap
        eliminated.
        """
        if not self.enabled:
            return
        now = time.perf_counter()
        self._stages[name] = self._stages.get(name, 0.0) + (now - self._tick_t)
        self._tick_t = now

    def end(self) -> dict | None:
        """Record one step; return its ``{wall_ms, gpu_ms, idle_pct, stages}``.

        On CPU there is no CUDA busy time to measure, but wall time (and the
        per-stage breakdown) is still recorded — ``gpu_ms``/``idle_pct`` are 0.
        """
        if not self.enabled:
            return None
        self.tick("other")
        wall_ms = (time.perf_counter() - self._wall_t0) * 1e3
        stages = {k: round(v * 1e3, 2) for k, v in self._stages.items()}
        if not self.is_cuda:
            row = {"wall_ms": wall_ms, "gpu_ms": 0.0, "idle_pct": 0.0, "stages": stages}
            self.rows.append(row)
            return row
        self._end = torch.cuda.Event(enable_timing=True)
        self._end.record()
        torch.cuda.synchronize()
        gpu_ms = self._start.elapsed_time(self._end)
        row = {
            "wall_ms": wall_ms,
            "gpu_ms": gpu_ms,
            "idle_pct": max(0.0, 1.0 - gpu_ms / wall_ms) * 100 if wall_ms > 0 else 0.0,
            "stages": stages,
        }
        self.rows.append(row)
        return row

    def stats(self, last: int | None = None) -> dict:
        """Aggregate over recent steps (all if ``last`` is None)."""
        rows = self.rows if last is None else self.rows[-last:]
        if not rows:
            return {"steps": 0, "wall_ms": 0.0, "gpu_ms": 0.0, "idle_pct": 0.0,
                    "stages": {}}
        wall = sum(r["wall_ms"] for r in rows)
        gpu = sum(r["gpu_ms"] for r in rows)
        stages: dict = {}
        for r in rows:
            for k, v in r.get("stages", {}).items():
                stages[k] = stages.get(k, 0.0) + v
        # Clamp at 0 like end() does: event-timing jitter can make aggregate
        # gpu_ms exceed wall_ms, and a negative "GPU idle" misleads the metric.
        idle_pct = (1.0 - gpu / wall) * 100 if wall > 0 else 0.0
        return {
            "steps": len(rows),
            "wall_ms": wall,
            "gpu_ms": gpu,
            "idle_pct": max(0.0, idle_pct),
            "stages": stages,
        }

    def format(self, last: int | None = None) -> str:
        s = self.stats(last)
        return (
            f"wall={s['wall_ms']:.1f}ms gpu={s['gpu_ms']:.1f}ms "
            f"idle={s['idle_pct']:.1f}%"
        )
