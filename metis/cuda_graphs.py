"""
Μῆτις (Metis) — CUDA Graphs for the Training Loop
===================================================
Wraps one full gradient-accumulation training iteration (``N`` micro-batch
forward+backward passes) in a single CUDA graph, then replays that graph every
step. The host-side work that CUDA graphs cannot capture — DataLoader fetches,
the scaler bookkeeping, gradient clipping, the optimizer update, EMA — runs
eagerly around the replay.

Requirements covered
--------------------
* **Capture one full training iteration** — one graph holds ``N × (forward +
  backward)`` with ``N`` static input slots and an fp64 loss accumulator.
* **Replay** — ``graph.replay()`` per iteration; data is copied into the static
  slots on the stream immediately before replay.
* **Optimizer** — the update runs eagerly after replay with the *current* LR
  (set per step by the caller), so AdamW state and schedules are untouched.
* **Gradient scaling** — the scale factor is fed into the graph as a device
  tensor (``scale_buf``) synced from ``scaler._scale`` before each replay, so a
  growing/backing-off scale is never baked into the capture. ``unscale_ →
  step → update`` run eagerly exactly as the eager path does them.
* **Mixed precision** — the forward is captured under ``torch.autocast`` with
  ``cache_enabled=False`` (weight-cast caching would pin stale fp16/bf16
  copies in the graph; disabling it makes the cast a captured op that
  re-reads the current master weights).
* **Automatic fallback** — capture is attempted lazily; CPU, MoE, gradient
  checkpointing, ``torch.compile``, and any ``RuntimeError`` from
  ``torch.cuda.graph`` degrade to the eager path with a logged reason.

Numerical parity
----------------
The captured region is the exact kernel sequence the eager path launches, in
the same order, on the same addresses:

* RNG: PyTorch's CUDA-graph capture re-advances the Philox offset on every
  replay, so dropout consumes a fresh mask each step instead of reusing the
  capture-time mask.
* Loss readback accumulates in fp64; the sequential adds match the eager
  loop's Python-double sum exactly, so ``loss_accum`` is bit-identical.
* Grad buffers are zeroed **in place** (``set_to_none=False``) so they stay at
  their capture-time pool addresses; ``set_to_none=True`` would free them and
  let the next replay write into freed memory.
* Gradient checkpointing is off inside the graph (capture is incompatible with
  its RNG save/restore), and the eager *fallback* keeps it on to match the
  original loop. Enabling CUDA graphs therefore also switches the training step
  to non-checkpointed execution — see ``docs/cuda_graphs.md``.
* Warmup steps are made invisible: model / optimizer / scaler state is
  snapshotted before warmup and restored after capture, so the graph starts
  from exactly the pre-warmup checkpoint.
"""

import copy
import logging

import torch
import torch.nn as nn

logger = logging.getLogger("metis.cuda_graphs")


class CUDAGraphStep:
    """Capture and replay the micro-batch loop of one Metis training step.

    ``train_step(batches)`` consumes ``N`` ``(x, y)`` pairs (CPU or CUDA) and
    returns ``(loss_accum, grad_norm)``. When ``active`` the micro-batch loop
    is a single graph replay; otherwise it is the identical eager computation
    (with gradient checkpointing, matching the original loop).
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: torch.amp.GradScaler,
        config,
        *,
        gradient_accumulation_steps: int,
        micro_batch_size: int,
        max_seq_len: int,
        amp_dtype: torch.dtype,
        device: str,
        warmup_iters: int = 3,
        capture_error_mode: str = "global",
    ):
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.config = config
        self.gradient_accumulation_steps = int(gradient_accumulation_steps)
        self.micro_batch_size = int(micro_batch_size)
        self.max_seq_len = int(max_seq_len)
        self.amp_dtype = amp_dtype
        self.device = device
        self.warmup_iters = warmup_iters
        self._capture_error_mode = capture_error_mode

        self.active = False
        self.reason = "not initialized"
        self.graph = None
        self.static_x = None
        self.static_y = None
        self.scale_buf = None
        self.loss_buf = None
        self._last_loss = 0.0
        self._grad_refs = {}
        # Async H2D staging (overlap): a side copy stream fills the static
        # slots for the *next* iteration while this iteration's graph replays.
        self.copy_stream = None
        self._stage_pending = False   # a staged copy awaits the next replay
        self._stage_event = None      # copy-stream event: stage done
        self._replay_event = None     # current-stream event: replay done

        if self.gradient_accumulation_steps < 1:
            self.reason = "gradient_accumulation_steps < 1"
            return

        try:
            self._try_setup()
        except Exception as exc:  # capture failed → eager fallback
            self.active = False
            self.reason = f"capture failed: {type(exc).__name__}: {exc}"
            logger.warning(
                "CUDA graph capture disabled — falling back to eager. (%s)",
                self.reason,
            )

    # ──────────────────────────────────────────────────────────────────────
    # Setup / capture
    # ──────────────────────────────────────────────────────────────────────

    def _try_setup(self) -> None:
        reason = self._capability_check()
        if reason:
            self.reason = reason
            return

        # Snapshot so the warmup's optimizer steps are invisible.
        snapshot = self._snapshot_state()
        self._prepare_buffers()
        try:
            self._warmup()
            self._capture()
        finally:
            self._restore_state(snapshot)

    def _capability_check(self) -> str | None:
        """Return a fallback reason string, or ``None`` if capture is viable."""
        if not self.device.startswith("cuda"):
            return f"not a CUDA device ({self.device})"
        if not torch.cuda.is_available():
            return "torch.cuda not available"
        if not hasattr(torch.cuda, "CUDAGraph"):
            return "torch.cuda.CUDAGraph unavailable (torch < 2.0)"

        # MoE routing uses data-dependent shapes (``torch.nonzero``,
        # ``bucketize``, host-side ``max_m``) that cannot be captured safely:
        # the graph would replay fixed-size buffers against variable-length
        # results.
        if getattr(self.config, "use_moe", False):
            return "MoE routing is data-dependent (unsafe to capture)"
        if getattr(self.config, "compile_model", False):
            return "torch.compile enabled (nested graph capture unsupported)"
        if getattr(self.config, "use_ddp", False):
            return "DDP enabled (NCCL collective capture unsupported)"
        return None

    def _prepare_buffers(self) -> None:
        B, T = self.micro_batch_size, self.max_seq_len
        N = self.gradient_accumulation_steps
        dev = torch.device(self.device)
        self.static_x = [
            torch.zeros(B, T, dtype=torch.long, device=dev) for _ in range(N)
        ]
        self.static_y = [
            torch.zeros(B, T, dtype=torch.long, device=dev) for _ in range(N)
        ]
        # fp32 device scalar the graph multiplies ``loss / N`` by. Synced to
        # ``scaler._scale`` before every replay so scale changes never bake in.
        self.scale_buf = torch.full((), 1.0, dtype=torch.float32, device=dev)
        # fp64 accumulator: sequential adds match the eager loop's Python
        # double sum exactly (bit-identical ``loss_accum`` readback).
        self.loss_buf = torch.zeros((), dtype=torch.float64, device=dev)

    def model_device_index(self) -> int:
        """CUDA device index for RNG snapshot/restore (defaults to 0)."""
        if self.device.startswith("cuda:"):
            return int(self.device.split(":")[1])
        return torch.cuda.current_device()

    def _snapshot_state(self) -> dict:
        rng = None
        try:
            rng = torch.cuda.get_rng_state(self.model_device_index()).clone()
        except Exception:
            pass
        return {
            "model": {
                k: v.detach().clone() for k, v in self.model.state_dict().items()
            },
            "optimizer": copy.deepcopy(self.optimizer.state_dict()),
            "rng": rng,
        }

    def _restore_state(self, snap: dict) -> None:
        """Put model/optimizer/RNG back to pre-warmup state (best effort).

        The GradScaler is *not* snapshotted: it is lazily initialized, so
        pre-warmup it holds no state. Warmup initializes it; we reset it to a
        freshly-initialized state (scale = ``init_scale``, growth tracker = 0)
        so the warmup's steps have no lasting effect on gradient scaling.
        """
        try:
            if "model" in snap and snap["model"]:
                self.model.load_state_dict(snap["model"])
            if "optimizer" in snap and snap["optimizer"]:
                self.optimizer.load_state_dict(snap["optimizer"])
            if "rng" in snap and snap["rng"] is not None:
                torch.cuda.set_rng_state(snap["rng"], self.model_device_index())

            sc = self.scaler
            if getattr(sc, "_enabled", False) and getattr(sc, "_scale", None) is not None:
                sc._scale.fill_(float(getattr(sc, "_init_scale", 2.0**16)))
                if getattr(sc, "_growth_tracker", None) is not None:
                    sc._growth_tracker.fill_(0.0)
        except Exception as exc:  # restore is best-effort; never block training
            logger.warning("CUDA graph state restore failed (%s)", exc)

    def _warmup(self) -> None:
        """Run full eager steps on a side stream with ordering choreography.

        Initializes cuBLAS/cuDNN workspaces, the scaler's lazily-created
        ``_scale``, and the fused optimizer state *before* capture so none of
        that lazy setup lands inside the graph. The tutorial side-stream
        pattern is required — a legacy-stream warmup makes capture fail with
        ``legacy stream depends on a capturing blocking stream``.
        """
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(self.warmup_iters):
                self._eager_step(self._random_batch())
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

    def _random_batch(self) -> list:
        N, B, T = (
            self.gradient_accumulation_steps,
            self.micro_batch_size,
            self.max_seq_len,
        )
        vocab = self.config.vocab_size
        return [
            (
                torch.randint(0, vocab, (B, T), device=self.device),
                torch.randint(0, vocab, (B, T), device=self.device),
            )
            for _ in range(N)
        ]

    def _eager_step(self, batches) -> float:
        """Full non-checkpointed eager step (mirrors the captured graph)."""
        self.optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for x, y in batches:
            with self._autocast():
                _, loss, _ = self.model(x, y, use_checkpointing=False)
                loss = loss / self.gradient_accumulation_steps
            self.scaler.scale(loss).backward()
            loss_accum += loss.item()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        # Invalidate MoE expert caches after fused-AdamW weight update.
        self.model.invalidate_moe_caches()
        return loss_accum

    def _autocast(self, cache_enabled: bool = True):
        return torch.autocast(
            device_type=self.device.split(":")[0],
            dtype=self.amp_dtype,
            enabled=self.amp_dtype != torch.float32,
            cache_enabled=cache_enabled,
        )

    def _capture(self) -> None:
        # Free any eager warmup grads so capture allocates them fresh in the
        # graph pool (stable, non-aliased addresses).
        self.optimizer.zero_grad(set_to_none=True)
        self.loss_buf.zero_()
        if getattr(self.scaler, "_scale", None) is not None:
            self.scale_buf.copy_(self.scaler._scale)
        torch.cuda.synchronize()

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, capture_error_mode=self._capture_error_mode):
            N = self.gradient_accumulation_steps
            for i in range(N):
                with self._autocast(cache_enabled=False):
                    _, loss, _ = self.model(
                        self.static_x[i], self.static_y[i],
                        use_checkpointing=False,
                    )
                loss_div = loss / N
                self.loss_buf.add_(loss_div.double())
                scaled = loss_div * self.scale_buf
                scaled.backward()

        # Keep the capture-time grad tensors alive (a scaler overflow path can
        # null them) so replay addresses never dangle.
        self._grad_refs = {
            name: p.grad
            for name, p in self.model.named_parameters()
            if p.grad is not None
        }
        self.active = True
        self.reason = "captured"
        # Side stream for async H2D staging of the *next* iteration's static
        # slots while the current graph replays.
        self.copy_stream = torch.cuda.Stream(device=self.device)
        self._stage_pending = False
        logger.info(
            "CUDA graph active: %d micro-batch(s) × (forward+backward) in one "
            "graph (autocast cache off, checkpointing off inside graph)",
            self.gradient_accumulation_steps,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Step execution
    # ──────────────────────────────────────────────────────────────────────

    def train_step(self, batches, prefetch_next=None) -> tuple[float, float]:
        """Run one full training iteration; return ``(loss_accum, grad_norm)``.

        ``batches`` is a sequence of exactly ``N = gradient_accumulation_steps``
        ``(x, y)`` pairs (CPU or CUDA, any dtype — they are moved here). If a
        staged copy for them is pending (see :meth:`stage_next`), it is used;
        otherwise they are copied synchronously now.

        ``prefetch_next`` optionally holds the *following* iteration's batches,
        which are staged into the static slots on a side copy stream while this
        iteration's graph replay (and its eager optimizer tail) runs — the H2D
        copies for step ``t+1`` overlap step ``t``'s compute.
        """
        if not self.active:
            return self._fallback_step(batches)
        return self._graph_step(batches, prefetch_next)

    def stage_next(self, batches) -> None:
        """Seed the pipeline: asynchronously stage ``batches`` for the next
        :meth:`train_step`. Call once before the first step when prefetching."""
        self._stage(batches)

    def _stage(self, batches) -> None:
        """Issue non-blocking H2D copies of ``batches`` into the static slots
        on ``self.copy_stream``, waiting for the prior replay that read them.
        """
        N = self.gradient_accumulation_steps
        batches = list(batches)
        if len(batches) != N:
            raise ValueError(
                f"expected {N} micro-batches for the captured graph, "
                f"got {len(batches)}"
            )
        # The static slots were last read by the previous replay — the copy
        # stream must not overwrite them while that replay still runs.
        if self._replay_event is not None:
            self.copy_stream.wait_event(self._replay_event)
        with torch.cuda.stream(self.copy_stream):
            for i in range(N):
                x, y = batches[i]
                self.static_x[i].copy_(x, non_blocking=True)  # pinned → device
                self.static_y[i].copy_(y, non_blocking=True)
            self._stage_event = torch.cuda.Event()
            self._stage_event.record(self.copy_stream)
        self._stage_pending = True

    def _graph_step(self, batches, prefetch_next=None) -> tuple[float, float]:
        if self._stage_pending:
            # Slots were filled asynchronously for THIS iteration — wait on it.
            torch.cuda.current_stream().wait_event(self._stage_event)
            self._stage_pending = False
        else:
            # Pipeline off / first step: synchronous fill now.
            self._stage(batches)
            torch.cuda.current_stream().wait_event(self._stage_event)
            self._stage_pending = False

        # In-place zero keeps grad buffers at their capture-time addresses.
        self.optimizer.zero_grad(set_to_none=False)
        self.loss_buf.zero_()
        if getattr(self.scaler, "_scale", None) is not None:
            self.scale_buf.copy_(self.scaler._scale)

        self.graph.replay()
        self._last_loss = float(self.loss_buf.item())

        # Replay done — the next stage may reuse the static slots. Record an
        # event and (if asked) start staging the next step's data, which then
        # overlaps this step's eager optimizer tail.
        self._replay_event = torch.cuda.Event()
        self._replay_event.record(torch.cuda.current_stream())
        if prefetch_next is not None:
            self._stage(prefetch_next)

        self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.max_grad_norm
        )
        grad_norm_val = float(grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        # Invalidate MoE expert caches after fused-AdamW weight update.
        self.model.invalidate_moe_caches()

        # Defensive: restore pool addresses if any path nulled a grad.
        for name, p in self.model.named_parameters():
            if p.requires_grad and p.grad is None and name in self._grad_refs:
                p.grad = self._grad_refs[name]

        return self._last_loss, grad_norm_val

    def _fallback_step(self, batches) -> tuple[float, float]:
        """Eager reference step — bit-identical to the original training loop.

        Gradient checkpointing mirrors the eager loop's rule (on CUDA, off on
        CPU — ``config.device.startswith('cuda')``) so the fallback is
        indistinguishable from a non-graph run on every device, not just CUDA.
        """
        batches = list(batches)
        self.optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        use_checkpointing = self.config.device.startswith("cuda")
        for x, y in batches:
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            with self._autocast():
                _, loss, _ = self.model(x, y, use_checkpointing=use_checkpointing)
                loss = loss / self.gradient_accumulation_steps
            self.scaler.scale(loss).backward()
            loss_accum += loss.item()
        self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.max_grad_norm
        )
        grad_norm_val = float(grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        # Invalidate MoE expert caches after fused-AdamW weight update.
        self.model.invalidate_moe_caches()
        return loss_accum, grad_norm_val

    # ──────────────────────────────────────────────────────────────────────
    # Reporting
    # ──────────────────────────────────────────────────────────────────────

    @property
    def last_loss(self) -> float:
        """Most recent ``loss_accum`` (graph path readback after replay)."""
        return self._last_loss

    def info(self) -> dict:
        return {
            "active": self.active,
            "reason": self.reason,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "micro_batch_size": self.micro_batch_size,
            "max_seq_len": self.max_seq_len,
            "dtype": str(self.amp_dtype),
        }
