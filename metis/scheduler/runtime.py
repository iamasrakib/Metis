"""
Μῆτις (Metis) — Execution scheduler (runtime for an ExecutionPlan)
==================================================================
The third stage of the scheduler. :class:`ExecutionScheduler` runs a
:class:`~metis.scheduler.planner.ExecutionPlan` and is a drop-in replacement
for ``model(idx, ...)`` with the same return contract ``(logits, loss,
new_kv_cache)``.

Two execution paths:

* **Infer** (``mode="infer"``, requires ``torch.inference_mode``/``no_grad``):
  the residual stream is a **single rolling buffer** updated with in-place
  ``add_`` (bit-identical to ``x + out`` — the same kernel, same rounding),
  and a SwiGLU FFN folds its activation into the ``w13`` gate view
  (``F.silu(..., inplace=True)`` + ``mul_``). Every GEMM / norm / attention
  call is the **same module call** as eager — the runtime never re-implements
  numerics, so a stale cost model can only mis-estimate, never change outputs.
* **Train** (``mode="train"`` or grad enabled): forwards to ``MetisLM.forward``
  directly — order-identical to eager by construction, because autograd needs
  the forward activations to survive to backward. The plan attached to the
  scheduler still carries the full analysis (cost, critical path, liveness).

Synchronization discipline
--------------------------
The infer hot loop issues **zero** host-side syncs: no ``.item()``, ``.cpu()``,
or ``.synchronize()``, and on a single CUDA stream no events. ``tests/
test_scheduler.py`` statically asserts this by scanning the method source, so a
future change that introduces a sync in the hot path fails CI.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

from ..model import SwiGLU
from .planner import INFER, TRAIN, ExecutionPlan, plan_execution

logger = logging.getLogger("metis.scheduler")

__all__ = ["ExecutionScheduler", "build_scheduler", "INFER", "TRAIN"]


def _module_device(model) -> str:
    p = next(model.parameters(), None)
    return str(p.device) if p is not None else "cpu"


class ExecutionScheduler:
    """Execute a model under a precomputed execution plan.

    Args:
        model: the ``MetisLM`` (or any module with the same tree).
        plan: an :class:`ExecutionPlan`, or ``None`` to build one lazily on the
            first :meth:`execute` (needs ``config``).
        mode: ``"infer"`` (arena reuse, requires no-grad) or ``"train"``.
        config: required when ``plan`` is ``None``.
    """

    def __init__(self, model, plan: ExecutionPlan | None = None, *,
                 mode: str = INFER, config=None, **plan_kwargs):
        self.model = model
        self.config = config if config is not None else getattr(model, "config", None)
        self.mode = mode
        self.device = _module_device(model)
        self.plan = plan
        self._plan_kwargs = plan_kwargs
        self.counters = {
            "forwards": 0,
            "residual_allocs": 0,   # fresh buffers the scheduler owns per forward
            "syncs": 0,             # host-side sync calls made by the scheduler
        }

    # ── plan lifecycle ───────────────────────────────────────────────────

    def build_plan(self, **kw) -> ExecutionPlan:
        """Build the plan now (startup analysis) and cache it."""
        if self.config is None:
            raise ValueError(
                "ExecutionScheduler.build_plan needs a config; pass config= "
                "to the constructor or call plan_execution() yourself."
            )
        kwargs = dict(self._plan_kwargs)
        kwargs.update(kw)
        self.plan = plan_execution(
            self.model, self.config, mode=self.mode, device=self.device, **kwargs
        )
        return self.plan

    def ensure_plan(self) -> ExecutionPlan:
        if self.plan is None:
            return self.build_plan()
        return self.plan

    # ── public execution entry point ─────────────────────────────────────

    def execute(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        use_checkpointing: bool = False,
        kv_cache: list | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list | None]:
        """Drop-in for ``model(idx, targets, use_checkpointing, kv_cache, ...)``.

        Returns ``(logits, loss, new_kv_cache)`` — identical outputs to the
        eager forward. The infer path only engages under ``no_grad``; if grad
        is enabled (or ``mode == "train"``) execution forwards to the eager
        forward so autograd is untouched.
        """
        self.counters["forwards"] += 1
        if self.mode == INFER and not torch.is_grad_enabled():
            return self._execute_infer(
                idx, targets=targets, use_checkpointing=use_checkpointing,
                kv_cache=kv_cache, attention_mask=attention_mask,
                position_ids=position_ids,
            )
        return self._execute_train(
            idx, targets=targets, use_checkpointing=use_checkpointing,
            kv_cache=kv_cache, attention_mask=attention_mask,
            position_ids=position_ids,
        )

    # ── train path (order-identical to eager) ────────────────────────────

    def _execute_train(self, idx, *, targets, use_checkpointing, kv_cache,
                       attention_mask, position_ids):
        return self.model.forward(
            idx, targets=targets, use_checkpointing=use_checkpointing,
            kv_cache=kv_cache, attention_mask=attention_mask,
            position_ids=position_ids,
        )

    # ── infer path (arena + in-place folds) ──────────────────────────────

    def _execute_infer(self, idx, *, targets, use_checkpointing, kv_cache,
                       attention_mask, position_ids):
        model = self.model
        config = self.config
        B, T = idx.size()
        device = idx.device
        if T > config.max_seq_len:
            raise ValueError(
                f"Sequence length {T} exceeds max_seq_len {config.max_seq_len}"
            )

        # Embeddings — the residual stream starts here and rolls in place.
        residual = model.tok_emb(idx)
        self.counters["residual_allocs"] += 1
        if not config.use_rope:
            pos = position_ids if position_ids is not None else \
                torch.arange(0, T, dtype=torch.long, device=device)
            residual = residual + model.pos_emb(pos)
        residual = model.drop(residual)

        new_kv_cache = []
        for i, layer in enumerate(model.layers):
            if model._layer_prefetch is not None:
                model._layer_prefetch.prefetch_next(i)
            layer_cache = kv_cache[i] if kv_cache is not None else None
            ln1_out = layer.ln_1(residual)
            attn_out, new_cache = layer.attn(
                ln1_out, kv_cache=layer_cache,
                attention_mask=attention_mask, position_ids=position_ids,
            )
            residual = residual.add_(attn_out)          # in-place, bit-identical
            ln2_out = layer.ln_2(residual)
            ffn_out = self._ffn(layer, ln2_out)
            residual = residual.add_(ffn_out)           # in-place, bit-identical
            new_kv_cache.append(new_cache)

        x = model.norm_f(residual)
        if targets is not None:
            logits = model.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=0,
            )
        else:
            logits = model.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss, new_kv_cache

    def _ffn(self, layer, x: torch.Tensor) -> torch.Tensor:
        """FFN forward with the SwiGLU activation folded in place.

        ``silu(gate) * up`` becomes ``F.silu(gate, inplace=True); gate.mul_(up)``
        — the same kernels writing into the gate view of the ``w13`` output, so
        the intermediate product tensor is never allocated. Bit-identical
        (verified by the parity suite). Non-SwiGLU FFNs (MLP, MoE) run as-is.
        """
        ffn = layer.ffn
        if isinstance(ffn, SwiGLU):
            w13_out = ffn.w13(x)
            h = ffn.hidden
            gate = w13_out.narrow(-1, 0, h)   # single view (split disallows in-place)
            up = w13_out.narrow(-1, h, h)
            F.silu(gate, inplace=True)
            gate.mul_(up)
            return ffn.dropout(ffn.w2(gate))
        return ffn(x)


def build_scheduler(
    model,
    config=None,
    *,
    mode: str = INFER,
    device: str | None = None,
    ref_shape: tuple = (1, 64),
    decode: bool = False,
    cache_len: int | None = None,
    amp_dtype=None,
    calibrate_run: bool = True,
    plan: ExecutionPlan | None = None,
) -> ExecutionScheduler:
    """Analyze → estimate → plan → build the scheduler in one call.

    ``plan=None`` builds the plan at construction (the "at startup" analysis).
    Pass ``plan=some_plan`` to reuse a precomputed plan (e.g. saved to JSON).
    """
    config = config if config is not None else getattr(model, "config", None)
    if device is None:
        device = _module_device(model)
    if plan is None:
        plan = plan_execution(
            model, config, mode=mode, device=device, ref_shape=ref_shape,
            decode=decode, cache_len=cache_len, amp_dtype=amp_dtype,
            calibrate_run=calibrate_run,
        )
    sched = ExecutionScheduler(model, plan, mode=mode, config=config)
    return sched
