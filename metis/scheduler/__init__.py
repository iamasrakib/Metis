"""
Μῆτις (Metis) — Graph-based execution scheduler
================================================
At startup the scheduler analyzes the model's computation graph, estimates
operator cost, safely reorders, reuses buffers, and minimizes allocations and
synchronization — producing an :class:`ExecutionPlan` and a runtime that
executes it with **identical outputs** to the eager forward.

Pipeline::

    build_scheduler(model, config)          # analyze → estimate → plan
        └─ ExecutionScheduler.execute(idx, kv_cache=...)   # drop-in for model()

    from metis.scheduler import build_scheduler
    sched = build_scheduler(model, config, mode="infer", device=config.device)
    logits, _, new_cache = sched.execute(idx, kv_cache=kv_cache)

The plan is JSON-serializable (``plan.to_dict()``) and human-readable
(``plan.render()``) — the "optimized execution plan" artifact. The parity
suite (``benchmarks/verify_exec_plan_parity.py``) proves outputs are
bit-identical to the eager path.
"""

from .buffers import BufferAssignment, assign
from .graph import (
    ADD,
    ATTN,
    CAT_SINK,
    CONTIG,
    DROP,
    EMBED,
    EMBED_POS,
    GEMM,
    HEAD,
    KV_APPEND,
    MOE_GEMM,
    MOE_ROUTE,
    NOOP,
    NORM,
    ROPE,
    SILU_MUL,
    VIEW,
    ComputationGraph,
    GraphNode,
    analyze_model,
    is_metadata,
)
from .planner import INFER, TRAIN, ExecutionPlan, plan_execution
from .runtime import ExecutionScheduler, build_scheduler

__all__ = [
    # scheduler
    "ExecutionScheduler",
    "build_scheduler",
    "ExecutionPlan",
    "plan_execution",
    "INFER",
    "TRAIN",
    # graph
    "ComputationGraph",
    "GraphNode",
    "analyze_model",
    "is_metadata",
    "EMBED",
    "EMBED_POS",
    "DROP",
    "NORM",
    "GEMM",
    "VIEW",
    "ROPE",
    "CAT_SINK",
    "KV_APPEND",
    "ATTN",
    "CONTIG",
    "ADD",
    "SILU_MUL",
    "MOE_ROUTE",
    "MOE_GEMM",
    "HEAD",
    "NOOP",
    # buffers
    "BufferAssignment",
    "assign",
]
