# Grouped MoE execution engine in Metis

Metis executes its Mixture-of-Experts FFN through a grouped, dynamically
scheduled pipeline in ([`metis/moe.py`](../metis/moe.py)): **token sorting →
expert batching → expert grouping → dynamic capacity → grouped GEMM → grouped
SwiGLU → grouped output projection**. It replaces the legacy per-expert loop —
one gather / GEMM / activation / GEMM / scatter per expert — with a batched
execution that runs all active experts in a handful of kernel launches.

The routing stage (gate → softmax → top-k → weight normalization) is shared
and **bit-identical** across engines; only the execution of the selected
experts differs. The legacy loop is retained as `per_expert`, a
byte-identical reference for parity verification and deterministic debugging —
mirroring how `metis/attn.py` keeps the manual math attention path alongside
the fused kernels. The pre-redesign scheduler is preserved as
`forward_grouped_legacy` for before/after benchmarks.

```
┌──────────────────────────────────────────────────────────────────────┐
│ MoE.forward                                                          │
│   gate(x) → softmax → topk → normalize         (identical everywhere)│
└──────────────────────────────────────────────────────────────────────┘
                          │
            ┌─────────────┴─────────────┐
            ▼                           ▼
   ┌─────────────────────┐     ┌──────────────────────────┐
   │ grouped (default)   │     │ per_expert (reference)   │
   │ 1. token sorting    │     │ for each expert:         │
   │ 2. expert batching  │     │   x[mask] → w1 → SiLU →  │
   │ 3. expert grouping  │     │   w2 → output[mask] +=   │
   │ 4. grouped GEMM     │     └──────────────────────────┘
   │ 5. grouped SwiGLU   │
   │ 6. grouped output   │
   └─────────────────────┘
```

## Why the per-expert loop was slow

Each token is routed to `top_k` experts, so a forward dispatches `N·top_k`
(token, expert) entries. The legacy loop walked the experts one at a time and,
for each, did a boolean-mask gather, two GEMMs, an activation, and an
in-place masked scatter (`output[mask] += …`):

- **Many tiny launches** — 8–16 experts × 4 layers ⇒ hundreds of small GEMMs
  that under-utilize the SM and pay full launch latency each.
- **Scatter-bound backward** — `output[mask] +=` is `index_put_`, whose
  backward (`IndexBackward0`) plus the `nonzero`/`index` mask plumbing is
  disproportionately expensive on low-power GPUs.
- **Repeated memory movement** — every expert re-reads and re-writes its slice
  through separate gather/scatter kernels.

Profiling a real training step (RTX 2050, bf16 AMP) makes the scatter problem
concrete:

| engine | `_index_put_impl_` | `nonzero` | `index` | scatter family total |
|---|---:|---:|---:|---:|
| per_expert | 184 calls · 24.9 ms | 376 calls · 41.3 ms | 192 calls · 14.5 ms | **90.6 ms** |
| grouped | 16 calls · 3.1 ms | 0 | 0 | **6.3 ms** |

The grouped engine collapses the whole per-expert scatter machinery into a
single `index_add_` (and its backward), removing `nonzero` entirely.

## Design

### 1. Token sorting

The `N·top_k` dispatch entries are flattened and sorted by expert id with a
stable `torch.sort`, producing contiguous per-expert blocks. A stable sort
keeps each token's per-expert entries in their original relative order, which
makes the within-block layout exact.

```
flat_experts = top_k_indices.reshape(-1)               # (M,)   M = N·top_k
token_ids    = arange(N).repeat_interleave(top_k)      # (M,)
sorted_experts, sort_idx = torch.sort(flat_experts, stable=True)
sorted_tokens   = token_ids[sort_idx]
sorted_weights  = flat_weights[sort_idx]
```

### 2. Expert batching

`torch.bincount` gives the per-expert token counts; idle experts are dropped
(only `A` active experts are materialized, so empty experts cost nothing). A
cumulative sum yields each expert's block start:

```
counts   = bincount(sorted_experts, minlength=E)      # (E,)
active   = nonzero(counts)                            # (A,)  sorted, A ≤ E
offsets  = cumsum(counts) - counts                    # block start per expert
```

### 3. Expert grouping + dynamic capacity (the redesign)

The pre-redesign grouped engine sized *every* block by the busiest expert
(`max_m = counts[active].max()`). A skewed routing — one expert swamped,
several near-idle — therefore padded most of the block (and most of the
tensor-core M-tiles) with empty rows. The redesigned scheduler fixes this in
two steps:

**Expert grouping.** `_group_active_experts` bins the active experts by token
load using greedy first-fit-decreasing, so every group has a tight max/min
ratio (default `group_max_ratio = 2.0`, tunable via
`ModelConfig.moe_group_ratio`). Idle experts never enter a group.

**Dynamic capacity.** Each group pads to
`max(busiest member, ceil(group tokens / group size))` instead of the global
max — the tightest block that never drops a token. Similar-load experts share
a group (balanced execution); the padded capacity tracks the actual load.

```
groups = _group_active_experts(counts, active, group_max_ratio)   # list[list[int]]
for group in groups:
    group_act = tensor(sorted(group))                       # sorted ids
    max_m     = max(counts[group_act].max(),
                    ceil(counts[group_act].sum() / A_g))
    ...dispatch → bmm → silu → bmm → index_add...
```

`right=True` in `bucketize` is still required: with the default `right=False`,
`bucketize` returns the boundary *equal to* the input, which mis-ranks the
smallest expert to −1. Each group's ids are sorted so its padded-block rows
and stacked weights agree.

A single gather per group (`x[src_idx]`, padding rows read `x[0]`)
materializes the padded `(A_g, max_m, D)` dispatch; routing weights are padded
with zeros the same way so padding contributes exactly zero to every
downstream op.

### 5. Grouped GEMM (`w1`)

`torch._grouped_mm` is not available on the Windows torch build, so the
grouped GEMM is expressed as one **strided-batched `torch.bmm`**: all `A`
experts' `w1` weights are stacked into `(A, D, hidden)` and multiplied with
the padded dispatch in a single cuBLAS call. Under AMP the bmm runs in bf16 on
tensor cores — the same promotion `nn.Linear` applies, but packed into one
high-occupancy kernel instead of `A` small ones.

### 6. Grouped SwiGLU

The expert nonlinearity (these experts are `Linear → SiLU → Linear`, no
separate gate branch) is applied as one elementwise pass over the grouped
activation: `h = silu(h1)`. The primitive accepts an optional gate projection
`h3` for a full `silu(h1)·h3` SwiGLU if the expert layout ever gains one.

### 7. Grouped output projection (`w2`) + scatter-accumulate

A second `torch.bmm` computes every expert's `w2` projection. Routing weights
are broadcast over the padding axis, and a single `index_add_` reuses the
slot→token map to accumulate each token's `top_k` contributions — one kernel
instead of one masked scatter per expert.

### Gradient flow

Every op is differentiable, so gradients flow to every parameter exactly as
before: `torch.sort` (backward through `sort_idx`), `index_copy` (gather in
backward), `bmm`, and `index_add` (scatter in backward). The stacked expert
weights stay connected to the parameters because `torch.stack` is
differentiable — no detached copies.

## Requirements — preserved

| Requirement | How it is preserved |
|---|---|
| **Routing** | gate → softmax → `torch.topk` → normalize executed identically before dispatch; verified bit-identical (`torch.equal`) |
| **Top-k behavior** | `M = N·top_k` entries, one per (token, expert); each token still receives exactly `top_k` weighted contributions |
| **Gradients** | all grouped ops differentiable; grads match the per-expert loop to 1e-6 |
| **Model outputs** | same math, fused-kernel rounding only; fp32 outputs match to ~8e-8 |
| **Checkpoints** | experts stay `Sequential(Linear, SiLU, Linear)` ⇒ `experts.{i}.0.weight` / `.2.weight` keys unchanged; existing MoE checkpoints load strictly |

## Numerical parity

`python benchmarks/verify_moe_parity.py --device cuda` (18/18 PASS, group-vs-reference):

```
[routing]   top-k indices + weights bit-identical
[forward]   fp32  max_err=7.82e-08
[forward]   fp16  max_err=3.13e-05
[forward]   bf16  max_err=0.00e+00
[grad]      all expert/gate grads match  max_grad_diff=1.13e-06
[model]     logits  max_logit_err=4.77e-07   loss identical
[model]     model grads  max_grad_diff=4.47e-08
[model]     KV-cache decode == full forward  (max_pos_err=0)
[model]     gradient-checkpointed grads == full grads  (0.00e+00)
[edge]      idle experts / top-k=1 / single-expert crowding all match
```

On the saved trained MoE checkpoint (`checkpoints/final_model.pt`, d=256,
8 experts): the new engine loads the weights strictly, and grouped-vs-per-
expert logits differ by `8.6e-6` and expert grads by `1.8e-7` — i.e. the two
engines are numerically interchangeable on real weights.

## Benchmark results

`python benchmarks/benchmark_moe.py --device cuda` — RTX 2050, torch 2.6.0+cu124
(median of 10 runs; layer numbers are forward+backward under bf16 AMP).

### Layer-level (MoE forward + backward)

| d_model | N tokens | experts | per_expert | grouped | speedup | launches p→g |
|--------:|---------:|--------:|-----------:|--------:|--------:|:---:|
| 256 | 128 | 8 | 17.4 ms | 3.8 ms | **4.6x** | 2635→634 |
| 256 | 128 | 16 | 35.4 ms | 5.4 ms | **6.5x** | 5219→874 |
| 256 | 2048 | 8 | 18.3 ms | 5.0 ms | **3.7x** | 2635→634 |
| 512 | 128 | 8 | 17.5 ms | 11.5 ms | **1.5x** | 2635→634 |
| 512 | 2048 | 8 | 21.9 ms | 16.3 ms | **1.3x** | 2635→634 |

The win is largest for many small experts (E=16: up to 6.5x) and shrinks as
per-expert GEMMs get big enough to amortize their own launches (d=512: 1.3–1.5x).

### Model-level (full MetisLM train step, 4 layers, bf16 AMP)

| engine | train step | tokens/s | CUDA-launching ops |
|---|---:|---:|---:|
| per_expert | 97.3 ms | 2 630 | 14 893 |
| grouped | 33.2 ms | 7 707 | 5 697 |
| **Δ** | **3.1x faster** | **2.9x** | **2.6x fewer** |

### Memory

The grouped engine trades a little memory for speed — zero-padded blocks and
per-forward stacked weights add roughly 2–8 MB of transient activation:

| seq len T | per_expert peak | grouped peak | Δ |
|---:|---:|---:|---:|
| 128 | 59.8 MB | 61.7 MB | +1.9 MB |
| 256 | 63.1 MB | 67.5 MB | +4.4 MB |
| 512 | 73.3 MB | 78.8 MB | +5.5 MB |

(About +3–7.5% of total peak for the 1.3–6.5x speedup.)

## Kernel profiling

The torch profilers on this Windows CUDA build attribute device time to aten
ops rather than individual kernels, so the breakdown is op-level — sufficient
to localize where each engine spends its time.

`python benchmarks/profile_moe.py` (one train step each):

```
[per_expert]  scatter family = 90.6 ms  (nonzero 376×, index_put 184×, index 192×)
[grouped]     scatter family =  6.3 ms  (index_put 16×)
```

The per-expert engine burns most of its device time in boolean-mask scatter
plumbing (`nonzero` + `_index_put_impl_` + `IndexBackward0`); the grouped
engine replaces all of it with a single `index_add_` and spends its time
spread across batched GEMMs and the sorting/compaction kernels.

## Dynamic scheduling — before / after

`python benchmarks/benchmark_moe.py --mode schedule --device cuda` compares the
pre-redesign scheduler (`forward_grouped_legacy`: every block padded to the
busiest expert) with the redesigned grouped + dynamic-capacity scheduler, under
increasing routing skew (fraction of tokens forced onto expert 0). RTX 2050,
torch 2.6.0, 1024 tokens, forward+backward, median of 10.

| d | E | skew | groups | old (ms) | new (ms) | speedup | pad waste old→new |
|---:|--:|-----:|-------:|---------:|---------:|--------:|------------------:|
| 512 | 16 | 0.00 | 1 | 11.33 | 8.65 | **1.31x** | 10.2% → 10.2% |
| 512 | 16 | 0.25 | 2 | 15.75 | 10.67 | **1.48x** | 175.8% → 19.8% |
| 512 | 16 | 0.50 | 2 | 21.90 | 9.33 | **2.35x** | 353.9% → 11.1% |
| 512 | 8 | 0.50 | 2 | 11.84 | 7.62 | **1.55x** | 146.5% → 10.1% |
| 256 | 16 | 0.50 | 2 | 7.37 | 5.05 | **1.46x** | 353.9% → 11.1% |
| 256 | 8 | 0.50 | 2 | 5.99 | 5.32 | **1.13x** | 146.5% → 10.1% |

The redesign's wins are exactly the stated goals:

* **Reduce padding / maximize occupancy** — a skewed routing pads up to **354%
  waste** with the old global-max scheduler; the grouped scheduler cuts that to
  **~10%**, so the tensor-core M-tiles are filled with real tokens.
* **Reduce idle experts** — idle experts never enter a group (they already cost
  nothing); near-idle experts share a group so they batch into one GEMM.
* **Reduce tiny GEMMs** — similar-load experts execute in a single strided-
  batched `bmm` pair, and every group's block tracks its actual token load.

At large d the new scheduler is faster even under *uniform* routing (1.3x), and
it never regresses there. For small models (d=256) with *moderate* skew the
extra per-group dispatch can cost more than the recovered padding; raise
`moe_group_ratio` (e.g. `1e9` = one group) to trade waste back for fewer
launches. Parity is exact when block shapes match and within fp rounding
otherwise (`max_parity_diff` ~1e-4–1e-3 at high skew, `0.0` at low skew).

## Configuration

`ModelConfig.moe_engine` / `--moe-engine` / `METIS_MOE_ENGINE`:

| value | behavior |
|---|---|
| `auto` (default) | grouped |
| `grouped` | token sorting + expert grouping + dynamic-capacity bmm pipeline |
| `per_expert` | the exact legacy loop (reference / debugging) |

`ModelConfig.moe_group_ratio` / `--moe-group-ratio` tunes the scheduler's
grouping threshold — the max-to-min token ratio tolerated inside one expert
group (default `2.0`):

| value | behavior |
|---|---|
| `<= 1` | one group per expert (max batching granularity, min padding) |
| `2.0` (default) | balanced launch count vs padding |
| `1e9` | single group (fewest launches; legacy-like block with dynamic capacity) |

```bash
metis train --preset medium --use-moe --moe-engine grouped --moe-group-ratio 2.0
METIS_MOE_ENGINE=per_expert metis train ...
```

`MoE.last_engine` reports the concrete engine used by the most recent forward.

## Reproduce

```bash
# correctness + parity (tests + verify script)
python -m pytest tests/test_moe.py -q
python benchmarks/verify_moe_parity.py --device cuda

# timings, launch counts, memory, before/after schedule
# (JSON + Markdown in benchmarks/results/)
python benchmarks/benchmark_moe.py --device cuda
python benchmarks/benchmark_moe.py --mode schedule --device cuda

# op-level CUDA-time profile (JSON in benchmarks/results/)
python benchmarks/profile_moe.py
```
