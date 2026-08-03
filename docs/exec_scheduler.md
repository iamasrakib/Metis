# Graph-based execution scheduler for Metis

The execution scheduler analyses the model's computation graph at startup and
produces an optimised execution plan that reuses buffers, minimises allocations,
reduces synchronisation, and estimates operator cost — while producing
**bit-identical outputs** to the eager forward.

## Design overview

```
build_scheduler(model, config)
    │
    ├── graph.py:    analyze_model(model, config)  →  ComputationGraph
    ├── cost.py:     estimate_costs(graph, device)  →  per-node est_ms
    ├── planner.py:  plan_execution(...)            →  ExecutionPlan
    │       ├── reorder: critical path, dead-op removal
    │       ├── buffers: liveness → arena slots, residual collapse
    │       └── sync plan: minimal event count
    └── runtime.py:  ExecutionScheduler(model, plan)
            ├── infer:  arena + in-place residual + SwiGLU fold
            └── train:  forwards to eager (autograd-safe)
```

## The six scheduler behaviours

| Behaviour | Where | What happens |
|-----------|-------|-------------|
| Analyse graph | `graph.py` | Typed recipe walker expands the module tree into op nodes with shapes, deps, FLOPs/bytes. Config-aware (MoE, QK-norm, sink, GQA). |
| Estimate cost | `cost.py` | Roofline model (flops/bytes → est_ms) with CPU/CUDA peak constants; optional startup calibration. |
| Reorder safe ops | `planner.py` | Topological order + critical path prove the block-level serialization is the honest lower bound; dead/no-op nodes are dropped. |
| Reuse buffers | `runtime.py` | Infer: residual stream rolls in-place (`add_`); SwiGLU activation folds into the `w13` gate view via `narrow` + `silu(inplace)` + `mul_`. |
| Minimise allocations | `planner.py` | Infer: n_layers+1 residual allocations → 1; n_layers silu temps → 0. Train: 0 (advisory only). |
| Reduce sync | `runtime.py` | Single-stream ordered execution; zero `.item()` / `.synchronize()` in the hot loop (statically verified). |

## Honest reorder story

The residual stream in a decoder-only transformer serialises the blocks. The
optimizer computes the critical path and reports *exactly* where the data
dependencies forbid reordering. No "fake parallelism" is manufactured — the
plan's critical path length equals the full execution order for the standard
config.

Where independent work *does* exist (e.g. future models with parallel attention
+ FFN), the planner detects it and assigns side-stream groups. On the standard
Metis architecture it honestly reports zero stream groups.

## Parity guarantee

The infer runtime calls the **same modules** as the eager forward — the only
divergence is the two `add_` calls replacing `x + out` (bit-identical: same
add kernel, same rounding, verified by the parity suite). The SwiGLU fold
replaces `silu(gate) * up` with `F.silu(gate, inplace=True); gate.mul_(up)`
which writes the same values into the same storage layout (also bit-identical).

The parity suite (`benchmarks/verify_exec_plan_parity.py`) asserts
`torch.equal` across: prefill, decode, targets, MoE, QK-norm, attention sink,
GQA, and train mode. Non-zero exit on any failure.

## How to use

### CLI

```bash
# Enable for generate:
metis generate --prompt "Hello" --exec-scheduler

# Enable for chat:
metis chat --exec-scheduler
```

### Python API

```python
from metis import build_scheduler, MetisLM, ModelConfig

model = MetisLM(ModelConfig.from_preset("small"))
model.eval()

sched = build_scheduler(model, mode="infer", device="cpu", calibrate_run=False)
print(sched.plan.render())

with torch.no_grad():
    logits, _, kv_cache = sched.execute(idx, kv_cache=kv_cache)
```

### Plan artifact

```python
plan = sched.plan
plan.to_dict()           # JSON-serialisable
plan.render()            # human-readable text
plan.summary()           # summary dict
```

## Benchmarks

```bash
# Parity verification:
python benchmarks/verify_exec_plan_parity.py --device cpu

# Performance benchmark:
python benchmarks/benchmark_exec_plan.py --device cpu --iters 20
```

## Architecture notes

- **CPU focus**: the arena removes real malloc churn (every avoided `torch.empty`
  on CPU is a freed-and-remalloc'd buffer). On CUDA the caching allocator makes
  per-call overhead smaller, so the win is primarily allocation-count reduction.
- **Training scope**: autograd needs forward activations for backward, so the
  scheduler's runtime does **not** alias or fold in training mode. The plan
  analysis (cost, critical path, liveness) is still generated and available.
- **KV-cache**: the attention module's internal `torch.cat` for KV appending is
  not touched (it is inside the attention module's forward). The KV-cache
  arena is an extension point.
