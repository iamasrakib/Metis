"""
Μῆτις (Metis) — Liveness-based buffer allocation for the execution scheduler
============================================================================
Two pieces:

* ``ArenaAllocator`` — the **planner's** engine. After liveness analysis each
  node's output is a live interval ``[order, dead_at]``; the allocator runs a
  greedy interval-graph coloring that reuses a slot as soon as its previous
  occupant is dead. This is exactly what bounds the activation footprint: the
  naive per-layer allocation keeps every tensor alive until the *end* of the
  forward (autograd), while the planned arena holds only the simultaneously
  live tensors. In this model the residual stream collapses to **one rolling
  slot** (in-place ``add_``), which is the difference between ``~9·n_layers``
  activation tensors and a handful.

Correctness rule
----------------
Two tensors may share a slot only if their live intervals do not overlap. The
allocator enforces this on the *scheduled* order; the runtime additionally
only aliases under ``torch.inference_mode()`` (see ``runtime.py``) because
autograd needs forward activations to survive to the backward pass — the plan
therefore never aliases a training-mode tensor.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ──────────────────────────────────────────────────────────────────────────────
# Planner-side analysis
# ──────────────────────────────────────────────────────────────────────────────

def naive_peak_bytes(nodes) -> int:
    """Peak live output bytes if nothing is reused (per-op fresh allocations).

    ``nodes`` must have ``order`` and ``dead_at`` set (:meth:`ComputationGraph
    .compute_liveness`). This is the baseline the arena is compared against.
    """
    peak = 0
    positions = sorted({n.order for n in nodes})
    for pos in positions:
        live = sum(
            n.bytes for n in nodes
            if n.bytes > 0 and n.order <= pos <= n.dead_at
        )
        peak = max(peak, live)
    return peak


@dataclass
class BufferAssignment:
    """Result of the liveness-based slot assignment."""

    slot_of: dict[int, int] = field(default_factory=dict)   # node id → slot
    slot_bytes: dict[int, int] = field(default_factory=dict)  # slot → max bytes
    arena_bytes: int = 0          # Σ slot_bytes — the planned footprint
    naive_peak: int = 0           # baseline without reuse
    reuse_count: int = 0          # nodes that landed on a freed slot
    slots: int = 0


def assign(nodes) -> BufferAssignment:
    """Greedy interval-coloring: reuse a slot once its occupant is dead.

    ``nodes`` must have ``order``, ``dead_at``, ``bytes`` set. Metadata nodes
    (``bytes == 0``) are ignored — they produce views, not storage.
    """
    ordered = sorted(nodes, key=lambda n: n.order)
    slot_of: dict[int, int] = {}
    slot_bytes: dict[int, int] = {}
    slot_dead: dict[int, int] = {}   # slot → dead_at of current occupant
    next_slot = 0
    reuse = 0

    for node in ordered:
        if node.bytes <= 0:
            slot_of[node.id] = -1
            continue
        free = [s for s, d in slot_dead.items() if d < node.order]
        if free:
            slot = free[0]
            reuse += 1
        else:
            slot = next_slot
            next_slot += 1
        slot_of[node.id] = slot
        slot_dead[slot] = node.dead_at
        slot_bytes[slot] = max(slot_bytes.get(slot, 0), node.bytes)

    return BufferAssignment(
        slot_of=slot_of,
        slot_bytes=slot_bytes,
        arena_bytes=sum(slot_bytes.values()),
        naive_peak=naive_peak_bytes(nodes),
        reuse_count=reuse,
        slots=next_slot,
    )
