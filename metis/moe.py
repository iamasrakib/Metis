"""
Μῆτις (Metis) — Grouped MoE execution engine with dynamic scheduling
=====================================================================
Replaces the legacy per-expert MoE loop (one gather / GEMM / SiLU / GEMM /
scatter per expert) with a batched, grouped, dynamically-scheduled execution
pipeline:

    1. token sorting     — stable argsort of the (token × top-k) dispatch
                           entries by expert id → contiguous per-expert blocks
    2. expert batching   — per-expert counts, prefix-sum boundaries, active
                           experts (idle experts are dropped)
    3. expert grouping   — active experts are binned by token load so each
                           group gets a tight, dynamically-sized block
    4. grouped GEMM      — one strided-batched ``torch.bmm`` per group
                           computes the group's ``w1`` projections in a single
                           launch (cublasGemmStridedBatched → fp16 tensor
                           cores on Ampere+)
    5. grouped SwiGLU    — the SiLU activation applied elementwise over the
                           grouped activations in one kernel
    6. grouped output projection — one ``torch.bmm`` per group for ``w2``,
                           then ``index_add_`` accumulates each token's top-k
                           contributions.

Scheduling (what changed in the redesign)
-----------------------------------------
The pre-redesign grouped engine padded *every* active expert up to the
busiest expert's token count (``max_m = max(counts[active])``). A skewed
routing — one expert swamped, several near-idle — therefore wasted most of
the padded block (and most of the tensor-core M-tiles) on empty rows. The
redesigned scheduler:

* **Token sorting** — unchanged (stable argsort).
* **Expert grouping** — :func:`_group_active_experts` bins active experts by
  token count (first-fit-decreasing) so every group has a tight max/min
  ratio. Idle experts never enter a group.
* **Dynamic capacity** — each group pads only to
  ``max(max group count, ceil(group tokens / group size))`` instead of the
  global max, slashing empty rows while never dropping a token.
* **Balanced execution** — a group of similarly-loaded experts gives every
  ``bmm`` a well-occupied M dimension (fewer tiny/empty GEMMs), and total
  padded capacity tracks the actual token load.

Routing (gate → softmax → top-k → normalization) is shared, unchanged, and
*bit-identical* across engines: only the execution of the selected experts
differs. ``per_expert`` is kept as the byte-identical legacy reference
(deterministic / debugging / parity verification), and
:func:`forward_grouped_legacy` preserves the pre-redesign scheduler for
before/after benchmarking.

Design notes
------------
- Routing, top-k behavior, gradients, and (up to fused-kernel fp rounding)
  model outputs are preserved. Gradients flow because every op is
  differentiable: ``torch.sort`` (backward through ``sort_idx``),
  ``index_copy`` (gather in backward), ``bmm``, ``index_add`` (scatter in
  backward), and the stacked expert weights stay connected to the parameters
  (``torch.stack`` is differentiable).
- ``torch._grouped_mm`` is unavailable on this torch build (2.6.0 Windows),
  so grouped GEMM is expressed as a padded strided-batched ``bmm`` — the
  canonical portable "grouped GEMM" primitive with full autograd. Padding
  rows read ``x[0]`` and are masked to zero by the grouped weights before the
  scatter-add, so they contribute exactly zero.
- Empty experts are dropped from the batched groups (``A`` = active count), so
  idle experts cost nothing beyond routing.

Engine names (user-facing → concrete):
    "auto"         → best available engine (grouped)
    "grouped"      → token sorting + expert grouping + grouped bmm pipeline
    "per_expert"   → the exact legacy per-expert loop (reference)
"""

import os
import warnings
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .expert_cache import ExpertCache

# ──────────────────────────────────────────────────────────────────────────────
# Engine constants
# ──────────────────────────────────────────────────────────────────────────────

AUTO = "auto"                    # best available engine (grouped)
GROUPED = "grouped"              # sorted / batched / grouped-bmm pipeline
PER_EXPERT = "per_expert"        # exact legacy per-expert loop (reference)

USER_FACING_ENGINES = frozenset({AUTO, GROUPED, PER_EXPERT})
CONCRETE_ENGINES = frozenset({GROUPED, PER_EXPERT})

__all__ = [
    "AUTO", "GROUPED", "PER_EXPERT", "USER_FACING_ENGINES", "CONCRETE_ENGINES",
    "normalize_engine", "resolve_engine", "detect_moe_engines", "MoE",
    "grouped_gemm", "grouped_swiglu", "grouped_output_projection",
    "forward_grouped", "forward_per_expert", "ExpertCache",
]


# ──────────────────────────────────────────────────────────────────────────────
# Engine resolution
# ──────────────────────────────────────────────────────────────────────────────

_warned = set()


def _warn_unavailable(requested: str, reason: str = "") -> None:
    """Warn once per (engine, reason) that a forced engine fell back."""
    key = (requested, reason)
    if key in _warned:
        return
    _warned.add(key)
    detail = f" ({reason})" if reason else ""
    warnings.warn(
        f"MoE engine {requested!r} is not available{detail}; "
        f"falling back to grouped.",
        stacklevel=3,
    )


def normalize_engine(requested: str | None) -> str:
    """Validate a user-facing engine name; resolve ``auto`` → concrete.

    Returns one of ``GROUPED`` / ``PER_EXPERT``. Raises ``ValueError`` on
    unknown names.
    """
    if requested is None or requested == AUTO:
        return GROUPED
    if requested in CONCRETE_ENGINES:
        return requested
    raise ValueError(
        f"Unknown MoE engine {requested!r}. Allowed: {sorted(USER_FACING_ENGINES)}"
    )


def resolve_engine(requested: str | None) -> str:
    """Resolve the concrete engine for one call.

    Precedence: ``METIS_MOE_ENGINE`` env var > explicit ``requested``
    (``auto`` / ``None`` → ``grouped``).
    """
    env = os.environ.get("METIS_MOE_ENGINE")
    if env:
        requested = normalize_engine(env)
    else:
        requested = normalize_engine(requested)
    return requested


def detect_moe_engines() -> dict:
    """Return a report of the engines this machine can run.

    Both engines are pure-PyTorch, so they are available everywhere; the report
    records the device / GPU for the benchmark header and the concrete default.
    """
    return {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "grouped": True,
        "per_expert": True,
        "recommended": GROUPED,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Grouped primitives
# ──────────────────────────────────────────────────────────────────────────────

def grouped_gemm(
    left: torch.Tensor, weights: torch.Tensor, name: str = ""
) -> torch.Tensor:
    """Grouped GEMM: ``left @ weights`` for every active expert in one launch.

    ``left`` is ``(A, M, K)`` (zero-padded per-expert blocks) and ``weights``
    is ``(A, K, N)``. A single strided-batched ``torch.bmm`` computes all
    ``A`` matmuls — cuBLAS dispatches this to fp16 tensor cores under AMP.
    """
    return torch.bmm(left, weights)


def grouped_swiglu(h1: torch.Tensor, h3: torch.Tensor | None = None) -> torch.Tensor:
    """Grouped SwiGLU activation over the batched hidden projections.

    These experts are ``Linear → SiLU → Linear`` (no separate gate branch),
    so the activation is ``silu(h1)``; if a gate projection ``h3`` is supplied
    the full SwiGLU product ``silu(h1) * h3`` is computed. One elementwise
    pass over the grouped tensor, no per-expert loop.
    """
    h = F.silu(h1)
    if h3 is not None:
        h = h * h3
    return h


def grouped_output_projection(
    h: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """Grouped output projection: ``h @ weights`` for every expert in one bmm."""
    return torch.bmm(h, weights)


# ──────────────────────────────────────────────────────────────────────────────
# Grouped execution pipeline (redesigned: grouping + dynamic capacity)
# ──────────────────────────────────────────────────────────────────────────────

def _group_active_experts(
    counts: torch.Tensor, active: torch.Tensor, group_max_ratio: float
) -> list[list[int]]:
    """Bin active experts into groups of similar token load.

    Greedy first-fit-decreasing: experts sorted by token count descending; each
    is placed into the first group whose largest member has at most
    ``group_max_ratio`` x the candidate's count. Every group then has a tight
    max/min ratio, so a per-group dynamic block size pads almost nothing.

    Args:
        counts: (E,) per-expert token counts.
        active: (A,) sorted active (nonzero-count) expert ids.
        group_max_ratio: max-to-min token ratio tolerated inside one group.
            ``<= 1`` -> every expert its own group (max batching granularity,
            minimal padding); ``>= 1e9`` -> one group (minimal launch count,
            max padding).

    Returns:
        List of expert-id groups, each sorted descending by count.
    """
    if group_max_ratio <= 1.0:
        return [[int(e)] for e in active.tolist()]
    if group_max_ratio >= 1e9:
        return [active.tolist()]

    groups: list[list[int]] = []
    for e in sorted(active.tolist(), key=lambda i: -int(counts[i])):
        c = int(counts[e])
        for g in groups:
            if int(counts[g[0]]) <= c * group_max_ratio:
                g.append(e)
                break
        else:
            groups.append([e])
    return groups


def _dispatch_group(
    x: torch.Tensor,
    sorted_experts: torch.Tensor,
    sorted_tokens: torch.Tensor,
    sorted_weights: torch.Tensor,
    offsets: torch.Tensor,
    group_act: torch.Tensor,
    max_m: int,
    is_full: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Zero-padded per-expert blocks for one expert group.

    Args:
        is_full: ``True`` when the group covers *every* active expert — skips
            the membership scan (``isin``) and uses the cheaper direct-slot
            path, which also makes the common low-skew (one-group) case as
            fast as the legacy scheduler.

    Returns ``(padded, padded_w, src_idx)``:
      * ``padded``  -- ``(A_g, max_m, D)``, padding rows read ``x[0]``;
      * ``padded_w`` -- aligned routing weights (padding -> 0);
      * ``src_idx`` -- slot -> token map for the final ``index_add_``.
    """
    device = x.device
    A_g = group_act.numel()
    M = sorted_experts.numel()

    if is_full:
        # Every dispatch entry belongs to this group — skip the membership
        # scan. Identical layout to the legacy direct-slot path.
        grp_experts, grp_tokens, grp_weights = sorted_experts, sorted_tokens, sorted_weights
        grp_rank = torch.bucketize(grp_experts, group_act, right=True) - 1
        grp_idx = torch.arange(M, device=device)
    else:
        in_group = torch.isin(sorted_experts, group_act)
        grp_experts = sorted_experts[in_group]
        grp_tokens = sorted_tokens[in_group]
        grp_weights = sorted_weights[in_group]
        grp_rank = torch.bucketize(grp_experts, group_act, right=True) - 1
        grp_idx = torch.arange(M, device=device)[in_group]

    # Global flat position of each group entry in the sorted dispatch array,
    # minus the expert's block offset gives its position inside the block.
    grp_pos = grp_idx - offsets.gather(0, grp_experts)
    slot = grp_rank * max_m + grp_pos  # (M_g,) unique slot ids

    src_idx = torch.zeros(A_g * max_m, dtype=torch.long, device=device)
    src_idx.index_copy_(0, slot, grp_tokens)
    padded = x[src_idx].view(A_g, max_m, x.shape[1])

    padded_w = sorted_weights.new_zeros(A_g * max_m)
    padded_w.index_copy_(0, slot, grp_weights)
    return padded, padded_w.view(A_g, max_m), src_idx


def forward_grouped(
    x: torch.Tensor,
    top_k_weights: torch.Tensor,
    top_k_indices: torch.Tensor,
    w1_views: list[torch.Tensor],
    w2_views: list[torch.Tensor],
    *,
    top_k: int,
    num_experts: int,
    group_max_ratio: float = 2.0,
    cache: ExpertCache | None = None,
    record: Callable | None = None,
) -> torch.Tensor:
    """Execute the routed experts with grouped, dynamically-scheduled blocks.

    Redesigned scheduling -- token sorting -> expert batching -> expert
    grouping -> per-group dynamic capacity -> grouped GEMM -> grouped SwiGLU ->
    grouped output projection:

    1. **Token sorting** -- stable argsort of the (token x top-k) dispatch
       entries by expert id -> contiguous per-expert blocks.
    2. **Expert grouping** -- active experts are binned by token count
       (:func:`_group_active_experts`); idle experts are skipped entirely.
    3. **Dynamic capacity** -- every group pads to ``max(busiest member,
       ceil(group tokens / group size))`` instead of the global max, so a
       skewed routing no longer pads *every* expert up to the outlier.
    4. **Balanced execution** -- one strided-batched ``bmm`` pair per group
       (well-occupied M-tiles, no tiny/empty GEMMs); results accumulate with
       one ``index_add_`` per group.

    Routing (gate -> softmax -> top-k -> normalization) is shared and
    unchanged; the schedule only rearranges *where* tokens are computed, never
    *what* is computed, so outputs and gradients match the per-expert
    reference up to fused-GEMM fp rounding.

    Args:
        x: (N, D) token activations (fp32 master dtype).
        top_k_weights: (N, top_k) normalized routing weights.
        top_k_indices: (N, top_k) int64 expert ids per token.
        w1_views: per-expert ``w1.weight.t()`` views -- (D, hidden) each.
        w2_views: per-expert ``w2.weight.t()`` views -- (hidden, D) each.
        top_k: experts selected per token.
        num_experts: total experts (for ``bincount`` length).
        group_max_ratio: max-to-min token ratio tolerated inside one expert
            group. Larger -> fewer, fatter groups; ``<= 1`` -> one group per
            expert. Default 2.0 balances launch count against padding.
        cache: optional :class:`ExpertCache` for persistent expert weight
            caching. ``None`` (the default) disables caching and is byte-
            identical to the original uncached path.

    Returns:
        (N, D) accumulated per-token expert output -- matches the per-expert
        loop up to fused-GEMM fp rounding.
    """
    N, D = x.shape
    device = x.device

    # 1. Token sorting ------------------------------------------------------
    flat_experts = top_k_indices.reshape(-1)        # (M,)
    flat_weights = top_k_weights.reshape(-1)        # (M,)

    token_ids = torch.arange(N, device=device).repeat_interleave(top_k)  # (M,)
    sorted_experts, sort_idx = torch.sort(flat_experts, stable=True)
    sorted_tokens = token_ids[sort_idx]
    sorted_weights = flat_weights[sort_idx]

    # 2. Expert batching ----------------------------------------------------
    counts = torch.bincount(sorted_experts, minlength=num_experts)  # (E,)
    active = torch.nonzero(counts, as_tuple=False).flatten()        # (A,)
    A = active.numel()
    if A == 0:
        return torch.zeros_like(x)
    offsets = torch.cumsum(counts, 0) - counts

    # 3. Expert grouping + per-group dynamic capacity ----------------------
    groups = _group_active_experts(counts, active, group_max_ratio)

    # Determine the cache/cast dtype.  Under AMP autocast the MoE receives
    # fp32 (RMSNorm / residual stream stay fp32), so x.dtype = fp32.  But
    # the intended cache dtype is the autocast compute dtype (bf16/fp16) so
    # the cached tensors are pre-cast, eliminating the per-bmm autocast
    # weight cast and halving resident memory.
    if torch.is_autocast_enabled(x.device.type):
        eff_dtype = torch.get_autocast_dtype(x.device.type)
    else:
        eff_dtype = x.dtype
    # The grouped-GEMM pipeline computes in eff_dtype under autocast (the
    # bmm outputs are bf16/fp16), so the accumulation buffer must match —
    # an fp32 accumulator would make index_add_ fail with a dtype mismatch.
    output = x.new_zeros(N, D, dtype=eff_dtype)

    # Report this layer's routed groups (ascending ids — the cache keys) so
    # the layer prefetcher can warm the *next* layer's cache during compute.
    if record is not None:
        record([tuple(sorted(g)) for g in groups], eff_dtype)

    for group in groups:
        group = sorted(group)  # ascending ids -> bucketize boundaries are sorted
        group_act = torch.tensor(group, dtype=torch.long, device=device)
        A_g = group_act.numel()
        group_tokens = int(counts[group_act].sum().item())
        # Tightest safe per-expert block: fits the busiest member and, on
        # average, every member (dynamic capacity -- no global-max padding).
        max_m = max(
            int(counts[group_act].max().item()),
            (group_tokens + A_g - 1) // A_g,
        )

        padded, padded_w, src_idx = _dispatch_group(
            x, sorted_experts, sorted_tokens, sorted_weights, offsets,
            group_act, max_m, is_full=(len(group) == A),
        )

        # Grouped weights: stack only this group's views (autograd-connected).
        if cache is not None:
            group_sources = (
                [w1_views[i] for i in group] + [w2_views[i] for i in group]
            )
            w1_group, w2_group = cache.get_or_build(
                group, eff_dtype, sources=group_sources,
                build=lambda: (
                    torch.stack([w1_views[i] for i in group]).to(eff_dtype),
                    torch.stack([w2_views[i] for i in group]).to(eff_dtype),
                ),
            )
        else:
            w1_group = torch.stack([w1_views[i] for i in group]).to(eff_dtype)
            w2_group = torch.stack([w2_views[i] for i in group]).to(eff_dtype)

        # 4. Balanced execution (one bmm pair per group) --------------------
        h1 = grouped_gemm(padded, w1_group)           # (A_g, max_m, hidden)
        h = grouped_swiglu(h1)                        # (A_g, max_m, hidden)
        out = grouped_output_projection(h, w2_group)  # (A_g, max_m, D)
        out = out * padded_w.unsqueeze(-1)            # zero the padding rows

        # 5. Accumulation ----------------------------------------------------
        # Cast to eff_dtype explicitly: out * padded_w can promote to fp32 if
        # padded_w is fp32 (non-autocast routing weights), which must not leak
        # into the eff_dtype accumulator.
        output.index_add_(0, src_idx, out.reshape(-1, D).to(eff_dtype))

    return output


def forward_grouped_legacy(
    x: torch.Tensor,
    top_k_weights: torch.Tensor,
    top_k_indices: torch.Tensor,
    w1_views: list[torch.Tensor],
    w2_views: list[torch.Tensor],
    *,
    top_k: int,
    num_experts: int,
) -> torch.Tensor:
    """Pre-redesign grouped engine -- single global-max-padded block.

    The OLD scheduler: every active expert is padded up to the busiest
    expert's token count (``max_m = max(counts[active])``), so a skewed
    routing pads many empty rows. Preserved verbatim for before/after
    benchmarks and as a reference for the redesigned scheduler.
    """
    N, D = x.shape
    device = x.device

    flat_experts = top_k_indices.reshape(-1)        # (M,)
    flat_weights = top_k_weights.reshape(-1)        # (M,)
    M = flat_experts.numel()

    token_ids = torch.arange(N, device=device).repeat_interleave(top_k)  # (M,)
    sorted_experts, sort_idx = torch.sort(flat_experts, stable=True)
    sorted_tokens = token_ids[sort_idx]
    sorted_weights = flat_weights[sort_idx]

    counts = torch.bincount(sorted_experts, minlength=num_experts)  # (E,)
    active = torch.nonzero(counts, as_tuple=False).flatten()        # (A,)
    A = active.numel()
    if A == 0:
        return torch.zeros_like(x)
    offsets = torch.cumsum(counts, 0) - counts
    max_m = int(counts[active].max().item())

    block_rank = torch.bucketize(sorted_experts, active, right=True) - 1
    pos_in_block = torch.arange(M, device=device) - offsets.gather(0, sorted_experts)
    slot = block_rank * max_m + pos_in_block                        # (M,)

    src_idx = torch.zeros(A * max_m, dtype=torch.long, device=device)
    src_idx.index_copy_(0, slot, sorted_tokens)
    padded = x[src_idx].view(A, max_m, D)                           # (A, max_m, D)

    padded_w = sorted_weights.new_zeros(A * max_m)
    padded_w.index_copy_(0, slot, sorted_weights)
    padded_w = padded_w.view(A, max_m)

    dtype = padded.dtype
    w1_group = torch.stack(w1_views)[active].to(dtype)  # (A, D, hidden)
    w2_group = torch.stack(w2_views)[active].to(dtype)  # (A, hidden, D)

    h1 = grouped_gemm(padded, w1_group)             # (A, max_m, hidden)
    h = grouped_swiglu(h1)                          # (A, max_m, hidden)
    out = grouped_output_projection(h, w2_group)    # (A, max_m, D)
    out = out * padded_w.unsqueeze(-1)              # zero padding rows

    output = x.new_zeros(N, D)
    output.index_add_(0, src_idx, out.reshape(-1, D))
    return output


def forward_per_expert(
    x: torch.Tensor,
    top_k_weights: torch.Tensor,
    top_k_indices: torch.Tensor,
    experts: nn.ModuleList,
    *,
    top_k: int,
) -> torch.Tensor:
    """Exact legacy per-expert loop — the reference execution engine.

    Byte-identical to the pre-redesign ``MoE.forward``: each expert gathers
    its tokens with a boolean mask, runs ``w2(silu(w1(·)))``, and scatters the
    weighted result back. Kept so ``grouped`` can be verified against the
    historical behavior and as a deterministic debugging path.
    """
    output = torch.zeros_like(x)
    for expert_idx, expert in enumerate(experts):
        mask = (top_k_indices == expert_idx).any(dim=-1)
        if not mask.any():
            continue
        expert_input = x[mask]
        expert_weight = top_k_weights[mask][(top_k_indices[mask] == expert_idx)]
        expert_output = expert(expert_input)
        output[mask] += expert_output * expert_weight.unsqueeze(-1)
    return output


# ──────────────────────────────────────────────────────────────────────────────
# MoE module
# ──────────────────────────────────────────────────────────────────────────────

class MoE(nn.Module):
    """Sparse Mixture of Experts layer with a grouped execution engine.

    Routes each token to the top-k experts out of ``num_experts`` via a learned
    router, then executes the selected experts either with the grouped
    pipeline (default; token sorting → expert batching → grouped GEMM → grouped
    SwiGLU → grouped output projection) or with the exact per-expert loop.

    Reference: Shazeer et al., "Outrageously Large Neural Networks" (2017)
    """

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.moe_num_experts
        self.top_k = config.moe_top_k
        self.d_model = config.d_model

        # Engine selection: "auto" | "grouped" | "per_expert"
        self.engine_request = getattr(config, "moe_engine", "auto")
        self.last_engine = None   # concrete engine used by the last forward
        # Layer-prefetch attachment (set by MetisLM after building layers).
        self._prefetch_idx = -1
        self._prefetcher = None
        # Expert-grouping threshold for the grouped scheduler (dynamic capacity).
        self.group_ratio = float(getattr(config, "moe_group_ratio", 2.0))

        # Expert weight cache: keeps active group tensors resident across
        # forwards to avoid repeated stack+cast of master fp32 weights.
        # Env vars override config; validate them (negatives warn, non-ints
        # fall back to config default).
        cfg_size = getattr(config, "moe_cache_size", 64)
        cfg_bytes = getattr(config, "moe_cache_bytes", 0)
        try:
            env_size = int(os.environ["METIS_MOE_CACHE_SIZE"])
            if env_size < 0:
                warnings.warn(
                    f"METIS_MOE_CACHE_SIZE={env_size} is negative; disabling cache",
                    stacklevel=3,
                )
            cache_size = env_size
        except (KeyError, ValueError):
            cache_size = cfg_size
        try:
            env_bytes = int(os.environ["METIS_MOE_CACHE_BYTES"])
            if env_bytes < 0:
                warnings.warn(
                    f"METIS_MOE_CACHE_BYTES={env_bytes} is negative; using 0",
                    stacklevel=3,
                )
            cache_bytes = env_bytes
        except (KeyError, ValueError):
            cache_bytes = cfg_bytes
        compile_on = getattr(config, "compile_model", False)
        if cache_size > 0 and not compile_on:
            self._cache = ExpertCache(cache_size, cache_bytes)
        else:
            self._cache = None
            if compile_on and cache_size > 0:
                warnings.warn(
                    "MoE expert cache disabled under torch.compile (not Dynamo-"
                    "traceable).  Set moe_cache_size=0 to suppress.",
                    stacklevel=3,
                )

        # Router / gate
        self.gate = nn.Linear(config.d_model, self.num_experts, bias=False)

        # Experts — each is a SwiGLU-style FFN kept as
        # ``Sequential(Linear, SiLU, Linear)`` so the ``state_dict`` keys
        # ``experts.{i}.0.weight`` / ``experts.{i}.2.weight`` are preserved
        # (existing MoE checkpoints load unchanged).
        hidden = int(4 * config.d_model * 2 / 3)
        hidden = ((hidden + 7) // 8) * 8

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.d_model, hidden, bias=False),
                nn.SiLU(),
                nn.Linear(hidden, config.d_model, bias=False),
            )
            for _ in range(self.num_experts)
        ])

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        x_flat = x.reshape(-1, D)  # (B*T, D)

        # Routing — shared by every engine, identical to the legacy path:
        # softmax logits → top-k → renormalize the kept weights.
        gate_logits = self.gate(x_flat)  # (B*T, n_experts)
        top_k_weights, top_k_indices = torch.topk(
            F.softmax(gate_logits, dim=-1), self.top_k, dim=-1
        )  # both (B*T, top_k)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        # Execute the selected experts with the resolved engine.
        engine = resolve_engine(self.engine_request)
        self.last_engine = engine
        if engine == PER_EXPERT:
            output = forward_per_expert(
                x_flat, top_k_weights, top_k_indices, self.experts,
                top_k=self.top_k,
            )
        else:
            output = forward_grouped(
                x_flat, top_k_weights, top_k_indices,
                w1_views=[e[0].weight.t() for e in self.experts],
                w2_views=[e[2].weight.t() for e in self.experts],
                top_k=self.top_k,
                num_experts=self.num_experts,
                group_max_ratio=self.group_ratio,
                cache=self._cache,
                record=self._record_routing if self._prefetcher is not None else None,
            )

        return self.dropout(output.view(B, T, D))

    # ── Expert cache API ──────────────────────────────────────────────────

    def invalidate_cache(self) -> None:
        """Drop all cached expert weight tensors.

        Call after every weight-mutating operation (optimizer step, load_state_dict,
        EMA apply/restore).  The framework training loop does this automatically;
        custom training loops should call ``model.invalidate_moe_caches()`` after
        each ``optimizer.step()``.
        """
        if self._cache is not None:
            self._cache.invalidate()

    def cache_stats(self) -> dict | None:
        """Return a snapshot of the expert cache statistics, or ``None``."""
        if self._cache is not None:
            return self._cache.stats()
        return None

    def _record_routing(self, groups, eff_dtype: torch.dtype) -> None:
        """Report this layer's routed groups to the layer prefetcher.

        Used as the ``record`` sink of ``forward_grouped``; the prefetcher
        warms the *next* layer's cache during this layer's compute.
        """
        if self._prefetcher is not None:
            self._prefetcher.record(self._prefetch_idx, groups, eff_dtype)

    def _apply(self, fn, recurse=True):
        """Reset the cache on device/dtype changes (``model.to(...)``)."""
        if self._cache is not None:
            self._cache.reset()
        return super()._apply(fn, recurse=recurse)
