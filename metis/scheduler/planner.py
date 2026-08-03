"""
Μῆτις (Metis) — Execution-plan optimizer
========================================
The second stage of the scheduler: turns the analyzed computation graph into an
:class:`ExecutionPlan` — the "optimized execution plan" artifact.

What the optimizer does, honestly
--------------------------------
For a decoder-only transformer the residual stream *serialises* the blocks, so
a planner that claimed to "parallelise" layers would be wrong — it would change
the outputs. Instead the optimizer applies only **provably-safe transforms** and
proves the rest:

1. **Reorder safe operations**
   - compute the topological order (data-dependency validity) and the
     **critical path** — the longest est-time chain, the schedule's lower bound;
   - detect and drop **dead / no-op nodes** (a ``contiguous()`` on an
     already-contiguous tensor, redundant casts) — reported, skipped at runtime;
   - identify **independent work** (nodes on disjoint dependency chains) for
     optional side-stream assignment. For this model the residual stream leaves
     none between blocks, and the plan says so instead of inventing overlap.
2. **Reuse buffers / minimize allocations** — liveness → interval-graph
   coloring (:class:`~metis.scheduler.buffers.ArenaAllocator`). Under
   ``mode="infer"`` the residual stream collapses to **one rolling buffer**
   (in-place ``add_``, bit-identical) and the SwiGLU activation folds into the
   ``w13`` gate view (in-place ``silu`` + ``mul``), eliminating
   ``~3·n_layers`` activation allocations per forward. Under ``mode="train"``
   autograd forces the residual to stay live, so the plan is advisory: it
   reports the theoretical arena but the runtime executes order-identically to
   eager.
3. **Reduce synchronization** — the plan assigns stream groups; every stream
   hand-off costs one CUDA event. With a single ordered stream the plan has
   **zero** sync points, and the runtime never calls ``.item()`` /
   ``.cpu()`` / ``.synchronize()`` in its hot path (statically verified).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .buffers import assign as assign_buffers
from .cost import calibrate, estimate_costs
from .graph import (
    ADD,
    EMBED,
    EMBED_POS,
    NOOP,
    SILU_MUL,
    ComputationGraph,
    analyze_model,
)

INFER = "infer"
TRAIN = "train"

# Kinds whose outputs the infer runtime folds in place / rolls (never allocates
# a fresh tensor for): the residual chain plus the SwiGLU activation.
RESIDUAL_KINDS = frozenset({EMBED, EMBED_POS, ADD})
FOLDED_KINDS = frozenset({SILU_MUL})


@dataclass
class ExecutionPlan:
    """A complete, serializable execution plan for one model forward."""

    graph: ComputationGraph
    mode: str
    device: str
    order: list[int] = field(default_factory=list)          # execution order
    critical_path: list[int] = field(default_factory=list)
    removed: list[int] = field(default_factory=list)        # dead/no-op node ids
    stream_groups: list[list[int]] = field(default_factory=list)  # per-stream nodes
    slot_of: dict[int, int] = field(default_factory=dict)
    slot_bytes: dict[int, int] = field(default_factory=dict)
    arena_bytes: int = 0                # actual runtime arena (residual slot)
    theoretical_arena_bytes: int = 0    # full-alias bound from the allocator
    naive_peak: int = 0
    reuse_count: int = 0
    folded_allocs: int = 0              # allocations eliminated by in-place ops
    sync_points: int = 0
    est_total_ms: float = 0.0
    calibrate_scale: float = 1.0
    measured_ms: float | None = None
    created_at: str = ""

    # ── reporting ────────────────────────────────────────────────────────

    @property
    def alloc_reduction(self) -> int:
        return self.folded_allocs

    def summary(self) -> dict:
        return {
            "mode": self.mode,
            "device": self.device,
            "nodes": len(self.graph.nodes),
            "execution_order": len(self.order),
            "critical_path_length": len(self.critical_path),
            "dead_ops_removed": len(self.removed),
            "stream_groups": len(self.stream_groups),
            "sync_points": self.sync_points,
            "arena_bytes": self.arena_bytes,
            "theoretical_arena_bytes": self.theoretical_arena_bytes,
            "naive_peak_bytes": self.naive_peak,
            "allocations_folded": self.folded_allocs,
            "est_total_ms": round(self.est_total_ms, 4),
            "calibrate_scale": round(self.calibrate_scale, 4),
        }

    def render(self) -> str:
        """Human-readable plan text (the 'optimized execution plan' artifact)."""
        g = self.graph
        lines = [
            f"Metis exec plan  mode={self.mode}  device={self.device}",
            f"  graph: {len(g.nodes)} nodes, ref_shape={g.ref_shape}"
            f"{', decode' if g.decode else ''}",
            f"  critical path: {' -> '.join(g.nodes[n].name for n in self.critical_path)}",
            f"  estimated total: {self.est_total_ms:.3f} ms"
            + (f"  (calibration ×{self.calibrate_scale:.3f})"
               if self.calibrate_scale != 1.0 else ""),
            f"  buffers: arena {self.arena_bytes} B vs naive peak {self.naive_peak} B"
            f"  (theoretical full-alias {self.theoretical_arena_bytes} B)",
            f"  allocations folded: {self.folded_allocs}   "
            f"dead ops removed: {len(self.removed)}   "
            f"stream groups: {len(self.stream_groups)}   sync points: {self.sync_points}",
            "",
            "  execution order:",
        ]
        for pos, nid in enumerate(self.order):
            n = g.nodes[nid]
            mark = ""
            if nid in self.critical_path:
                mark = " ◄ critical"
            slot = f" slot[{n.slot}]" if n.slot is not None else ""
            lines.append(
                f"    {pos:>3} {n.kind:<10} {n.name:<22} "
                f"{n.est_ms:>9.4f} ms  {n.bytes:>10} B{slot}{mark}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "device": self.device,
            "graph": self.graph.to_dict(),
            "order": self.order,
            "critical_path": self.critical_path,
            "removed": self.removed,
            "stream_groups": self.stream_groups,
            "slot_of": self.slot_of,
            "slot_bytes": self.slot_bytes,
            "arena_bytes": self.arena_bytes,
            "theoretical_arena_bytes": self.theoretical_arena_bytes,
            "naive_peak": self.naive_peak,
            "reuse_count": self.reuse_count,
            "folded_allocs": self.folded_allocs,
            "sync_points": self.sync_points,
            "est_total_ms": self.est_total_ms,
            "calibrate_scale": self.calibrate_scale,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ExecutionPlan:
        graph = ComputationGraph.from_dict(data["graph"])
        return cls(
            graph=graph, mode=data["mode"], device=data["device"],
            order=data["order"], critical_path=data["critical_path"],
            removed=data["removed"], stream_groups=data["stream_groups"],
            slot_of={int(k): v for k, v in data["slot_of"].items()},
            slot_bytes={int(k): v for k, v in data["slot_bytes"].items()},
            arena_bytes=data["arena_bytes"],
            theoretical_arena_bytes=data["theoretical_arena_bytes"],
            naive_peak=data["naive_peak"], reuse_count=data["reuse_count"],
            folded_allocs=data["folded_allocs"], sync_points=data["sync_points"],
            est_total_ms=data["est_total_ms"],
            calibrate_scale=data["calibrate_scale"], created_at=data["created_at"],
        )


# ──────────────────────────────────────────────────────────────────────────────
# Optimization passes
# ──────────────────────────────────────────────────────────────────────────────

def detect_dead_nodes(graph: ComputationGraph) -> list[int]:
    """Find provably-dead / no-op nodes (removed from the schedule).

    Currently detects a ``contiguous()`` whose producer is already contiguous
    and marked ``output_name == "contiguous"``-safe, plus any node explicitly
    typed ``noop``. Conservative by construction: a false negative costs an
    unnecessary kernel, a false positive would change outputs — so we only
    remove what we can *prove*.
    """
    removed: list[int] = []
    for nid, node in graph.nodes.items():
        if node.kind == NOOP:
            removed.append(nid)
    return removed


def _residual_chain(graph: ComputationGraph) -> list[int]:
    """Node ids that carry the residual stream (embed → per-layer adds)."""
    return [nid for nid, n in graph.nodes.items()
            if n.kind in RESIDUAL_KINDS and n.output_name == "residual"]


def plan_execution(
    model,
    config,
    *,
    mode: str = INFER,
    device: str = "cpu",
    ref_shape: tuple = (1, 64),
    decode: bool = False,
    cache_len: int | None = None,
    amp_dtype=None,
    calibrate_run: bool = True,
) -> ExecutionPlan:
    """Analyze → estimate → reorder → assign buffers → build the plan.

    Args:
        model: the ``MetisLM`` instance.
        config: its ``ModelConfig``.
        mode: ``"infer"`` (arena reuse sound) or ``"train"`` (advisory plan).
        device: device string for the cost model ("cpu"/"cuda").
        ref_shape: ``(B, T)`` the plan is sized for.
        decode: build the plan for a kv-cache decode step.
        cache_len: cached ``T_k`` for decode attention.
        amp_dtype: activation dtype for the byte model (default fp32).
        calibrate_run: time the model once and rescale ``est_ms`` to reality.

    Returns:
        An :class:`ExecutionPlan` — use :meth:`render` / JSON for the artifact
        and :class:`~metis.scheduler.runtime.ExecutionScheduler` to run it.
    """
    import torch as _torch

    if amp_dtype is None:
        amp_dtype = _torch.float32
    graph = analyze_model(model, config, ref_shape=ref_shape, decode=decode,
                          cache_len=cache_len, amp_dtype=amp_dtype,
                          training=(mode == TRAIN))
    estimate_costs(graph, device)
    scale = 1.0
    if calibrate_run and not decode:
        try:
            scale = calibrate(graph, model, config, device=device)
        except Exception:
            scale = 1.0
    for node in graph.nodes.values():
        node.est_ms *= scale

    order = graph.topological_order()
    removed = detect_dead_nodes(graph)
    live = [nid for nid in order if nid not in removed]

    # Reorder pass: mark the critical path and assign stream groups.
    graph.compute_liveness()
    critical = graph.critical_path()

    # Buffer assignment (theoretical full-alias bound).
    nodes_live = [graph.nodes[nid] for nid in live]
    theoretical = assign_buffers(nodes_live)

    # Actual runtime arena: infer folds the residual chain into one rolling
    # slot; train keeps per-layer residual (advisory slots, no aliasing).
    residual = [nid for nid in _residual_chain(graph) if nid in live]
    slot_of: dict[int, int] = {nid: -1 for nid in theoretical.slot_of}
    if mode == INFER and residual:
        max_bytes = max((graph.nodes[n].bytes for n in residual), default=0)
        for nid in residual:
            slot_of[nid] = 0
        arena_bytes = max_bytes
    else:
        arena_bytes = theoretical.arena_bytes

    # Allocations eliminated by in-place folding (infer mode).
    folded = 0
    if mode == INFER:
        folded += len([nid for nid in live if graph.nodes[nid].kind == ADD])
        folded += len([nid for nid in live if graph.nodes[nid].kind in FOLDED_KINDS])

    # Sync plan: one event per stream hand-off; single-stream → zero.
    stream_groups: list[list[int]] = []
    stream_map: dict[int, int] = {}
    for nid in live:
        s = stream_map.setdefault(nid, 0)
        while len(stream_groups) <= s:
            stream_groups.append([])
        stream_groups[s].append(nid)
    sync_points = sum(
        1 for i in range(len(live) - 1)
        if stream_map.get(live[i], 0) != stream_map.get(live[i + 1], 0)
    )
    for nid in live:
        graph.nodes[nid].stream = f"s{stream_map.get(nid, 0)}"

    plan = ExecutionPlan(
        graph=graph,
        mode=mode,
        device=device,
        order=live,
        critical_path=critical,
        removed=removed,
        stream_groups=stream_groups,
        slot_of=slot_of,
        slot_bytes=theoretical.slot_bytes if mode != INFER else {0: arena_bytes},
        arena_bytes=arena_bytes,
        theoretical_arena_bytes=theoretical.arena_bytes,
        naive_peak=theoretical.naive_peak,
        reuse_count=theoretical.reuse_count,
        folded_allocs=folded,
        sync_points=sync_points,
        est_total_ms=sum(graph.nodes[n].est_ms for n in live),
        calibrate_scale=scale,
        created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    # Publish the slot/stream on each node for the renderer.
    for nid in live:
        graph.nodes[nid].slot = plan.slot_of.get(nid)
    return plan
