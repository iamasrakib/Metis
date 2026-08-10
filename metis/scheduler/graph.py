"""
Μῆτις (Metis) — Computation-graph analysis for the execution scheduler
=======================================================================
The first stage of the scheduler. ``analyze_model`` walks the *real* Metis
module tree (``MetisLM`` → ``tok_emb`` → ``layers[]`` → ``attn`` / ``ffn`` →
``norm_f`` → ``lm_head``) and expands it into an operator-level DAG — one node
per kernel-launchable computation, with explicit data-dependency edges, output
shapes, and FLOPs/byte counts.

Why a typed recipe walker and not a tracer
------------------------------------------
The graph is derived from the *module structure* (per-module "recipes" that
mirror each module's ``forward``), not from tracing a live run. That keeps the
analysis:

* **Config-aware** — the walker reads the same flags the model's forward reads
  (``use_rope``, ``use_qk_norm``, ``use_attention_sink``, MoE vs SwiGLU vs MLP,
  GQA head counts), so the graph tracks the actual architecture.
* **Tracer-free** — MoE routing uses data-dependent shapes (``torch.nonzero``,
  ``bucketize``) that no static tracer handles cleanly; the recipe treats the
  routed expert block as a single ``moe_gemm`` node with a FLOP estimate.
* **Guaranteed not to drift** — ``tests/test_scheduler.py`` pins every recipe
  against the module source (node counts/kinds per config), and the runtime
  (``runtime.py``) never re-implements numerics: it calls the same modules, so
  a stale recipe can only mis-estimate cost, never change outputs.

Liveness
--------
``compute_liveness`` annotates every node with the position of its last
consumer, which turns the DAG into an interval graph: a node's output is *live*
from its own execution position until its last consumer has read it. This is
what the buffer assigner (``buffers.py`` / ``planner.py``) uses to prove which
buffers can be aliased and how the peak activation footprint is bounded.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

# ──────────────────────────────────────────────────────────────────────────────
# Node kinds
# ──────────────────────────────────────────────────────────────────────────────

EMBED = "embed"            # token embedding gather
EMBED_POS = "embed_pos"    # learned positional embedding add (use_rope=False)
DROP = "drop"              # dropout (training only)
NORM = "norm"              # RMSNorm / LayerNorm
GEMM = "gemm"              # linear projection (fused QKV / SwiGLU w13 etc.)
VIEW = "view"              # view / reshape / transpose / split / slice (no kernel)
ROPE = "rope"              # rotary position embedding (complex multiply)
CAT_SINK = "cat_sink"      # attention-sink token prepend
KV_APPEND = "kv_append"    # kv_cache cat
ATTN = "attn"              # causal attention kernel (flash / SDPA / math)
CONTIG = "contig"          # .contiguous() — zero FLOPs, may copy
ADD = "add"                # residual-stream add
SILU_MUL = "silu_mul"      # SwiGLU activation (silu(gate) * up)
ACT = "act"                # GELU (MLP fallback)
MOE_ROUTE = "moe_route"    # gate softmax → top-k routing
MOE_GEMM = "moe_gemm"      # grouped expert GEMM block (data-dependent shape)
HEAD = "head"              # lm_head projection
NOOP = "noop"              # provably-dead node removed by the planner

# Node kinds that produce a fresh, runtime-owned tensor (module outputs) as
# opposed to zero-kernel metadata views.
METADATA_KINDS = frozenset({VIEW, NOOP})


def is_metadata(kind: str) -> bool:
    """True for nodes that launch no kernel and allocate no storage."""
    return kind in METADATA_KINDS


@dataclass
class GraphNode:
    """One computation in the operator DAG.

    ``deps`` are the node ids whose tensors this node reads. ``out_shape`` is
    the shape of the tensor this node produces. ``output_name`` labels the
    tensor role in the model's dataflow (``"residual"``, ``"attn_out"``, …) —
    used by the planner to recognise the residual stream.
    """

    id: int
    name: str
    kind: str
    module_path: str
    out_shape: tuple | None
    deps: frozenset = frozenset()
    flops: int = 0
    bytes: int = 0                # output tensor bytes (allocation footprint)
    est_ms: float = 0.0           # filled by cost.py
    stream: str | None = None     # side-stream group, filled by the planner
    slot: int | None = None       # arena slot, filled by the planner
    output_name: str = ""
    order: int = -1               # position in the final execution order
    dead_at: int = -1             # last position that reads this output (liveness)

    @property
    def is_metadata(self) -> bool:
        return is_metadata(self.kind)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "module_path": self.module_path,
            "out_shape": None if self.out_shape is None else list(self.out_shape),
            "deps": sorted(self.deps),
            "flops": self.flops,
            "bytes": self.bytes,
            "est_ms": self.est_ms,
            "stream": self.stream,
            "slot": self.slot,
            "output_name": self.output_name,
        }


@dataclass
class ComputationGraph:
    """The operator DAG produced by :func:`analyze_model`."""

    nodes: dict[int, GraphNode] = field(default_factory=dict)
    ref_shape: tuple = (1, 64)     # (B, T) the plan is sized for
    decode: bool = False           # kv-cache decode step (T_q == 1) vs prefill
    cache_len: int | None = None   # T_k for decode attention (prefill: == T)

    def add(
        self,
        kind: str,
        name: str,
        module_path: str,
        out_shape: tuple | None,
        deps: frozenset | set | None = None,
        **kw,
    ) -> int:
        nid = len(self.nodes)
        self.nodes[nid] = GraphNode(
            id=nid, kind=kind, name=name, module_path=module_path,
            out_shape=out_shape, deps=frozenset(deps or ()), **kw,
        )
        return nid

    # ── traversal ────────────────────────────────────────────────────────

    def topological_order(self) -> list[int]:
        """Kahn's algorithm — a valid execution order (deps before users)."""
        order: list[int] = []
        indeg = {nid: len(n.deps) for nid, n in self.nodes.items()}
        ready = [nid for nid, d in indeg.items() if d == 0]
        while ready:
            nid = ready.pop(0)
            order.append(nid)
            for other in self.nodes:
                if nid in self.nodes[other].deps:
                    indeg[other] -= 1
                    if indeg[other] == 0:
                        ready.append(other)
        if len(order) != len(self.nodes):
            raise RuntimeError(
                f"cycle detected in computation graph "
                f"({len(self.nodes) - len(order)} nodes unreachable)"
            )
        return order

    def sources(self) -> list[int]:
        return [nid for nid, n in self.nodes.items() if not n.deps]

    def sinks(self) -> list[int]:
        consumers = {d for n in self.nodes.values() for d in n.deps}
        return [nid for nid in self.nodes if nid not in consumers]

    def critical_path(self) -> list[int]:
        """Longest est-time dependency chain (the schedule's lower bound).

        For a decoder-only transformer the residual stream serialises the
        blocks, so this is typically the whole block chain — the honest proof
        of *why* operations cannot be reordered further. Empty ``est_ms``
        values degrade gracefully to a longest-dependency-chain.
        """
        order = self.topological_order()
        longest: dict[int, float] = {nid: 0.0 for nid in self.nodes}
        parent: dict[int, int] = {nid: -1 for nid in self.nodes}
        for nid in order:
            node = self.nodes[nid]
            best, best_p = 0.0, -1
            for dep in node.deps:
                if longest[dep] > best:
                    best, best_p = longest[dep], dep
            longest[nid] = best + node.est_ms
            parent[nid] = best_p
        end = max(longest, key=longest.get)
        path = []
        cur = end
        while cur != -1:
            path.append(cur)
            cur = parent[cur]
        path.reverse()
        return path

    # ── liveness ─────────────────────────────────────────────────────────

    def compute_liveness(self) -> None:
        """Set ``dead_at`` for every node — the position of its last consumer.

        A node's output is live on the interval ``[order, dead_at]``. Sink
        outputs (the logits) live to the end of the schedule.
        """
        order = self.topological_order()
        for i, nid in enumerate(order):
            self.nodes[nid].order = i
        consumers: dict[int, list[int]] = {nid: [] for nid in self.nodes}
        for nid, node in self.nodes.items():
            for dep in node.deps:
                consumers[dep].append(nid)
        for nid, node in self.nodes.items():
            reads = consumers[nid]
            if not reads:
                node.dead_at = len(order) - 1  # kept to the end (sink/root)
            else:
                node.dead_at = max(self.nodes[c].order for c in reads)

    # ── serialization ────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "ref_shape": list(self.ref_shape),
            "decode": self.decode,
            "cache_len": self.cache_len,
            "nodes": [self.nodes[nid].to_dict() for nid in sorted(self.nodes)],
        }

    @classmethod
    def from_dict(cls, data: dict) -> ComputationGraph:
        g = cls(ref_shape=tuple(data["ref_shape"]), decode=data.get("decode", False),
                cache_len=data.get("cache_len"))
        for nd in data["nodes"]:
            g.nodes[nd["id"]] = GraphNode(
                id=nd["id"], name=nd["name"], kind=nd["kind"],
                module_path=nd["module_path"],
                out_shape=None if nd["out_shape"] is None else tuple(nd["out_shape"]),
                deps=frozenset(nd["deps"]), flops=nd["flops"], bytes=nd["bytes"],
                est_ms=nd["est_ms"], stream=nd["stream"], slot=nd["slot"],
                output_name=nd["output_name"],
            )
        return g


# ──────────────────────────────────────────────────────────────────────────────
# Analyzer
# ──────────────────────────────────────────────────────────────────────────────

def _swiglu_hidden(d_model: int) -> int:
    hidden = int(4 * d_model * 2 / 3)
    return ((hidden + 7) // 8) * 8


def analyze_model(
    model,
    config,
    ref_shape: tuple = (1, 64),
    *,
    decode: bool = False,
    cache_len: int | None = None,
    amp_dtype: torch.dtype = torch.float32,
    training: bool = False,
) -> ComputationGraph:
    """Expand the model's module tree into an operator DAG.

    Args:
        model: a ``MetisLM`` (or a module exposing the same tree).
        config: the model's ``ModelConfig``.
        ref_shape: ``(B, T)`` the plan is sized/estimated for. During decode
            pass ``(B, 1)``.
        decode: KV-cache decode step (``T_q == 1``, ``kv_append`` present).
        cache_len: cached ``T_k`` used to shape decode attention.
        amp_dtype: activation dtype (drives the byte footprint estimate).
        training: include dropout nodes (inference has none).

    Returns:
        A :class:`ComputationGraph` with shapes and FLOPs/bytes set; costs
        (``est_ms``) are filled by ``cost.py`` afterwards.
    """
    B, T = ref_shape
    d = config.d_model
    H = config.n_heads
    kvH = config.n_kv_heads
    hd = config.head_dim
    kv_dim = config.d_model * kvH // H
    vocab = config.vocab_size
    db = amp_dtype.itemsize
    drop = config.dropout > 0 and training

    g = ComputationGraph(ref_shape=(B, T), decode=decode,
                         cache_len=cache_len if decode else T)

    # ── token embedding ─────────────────────────────────────────────────
    x = g.add(EMBED, "tok_emb", "tok_emb", (B, T, d),
              flops=2 * B * T * d, bytes=B * T * d * db, output_name="residual")
    if not config.use_rope:
        x = g.add(EMBED_POS, "pos_emb", "pos_emb", (B, T, d), deps={x},
                  flops=B * T * d, bytes=B * T * d * db, output_name="residual")
    if drop:
        x = g.add(DROP, "drop", "drop", (B, T, d), deps={x},
                  flops=B * T * d, bytes=B * T * d * db, output_name="residual")

    # ── transformer blocks ──────────────────────────────────────────────
    t_q = 1 if decode else T
    t_k = (cache_len if cache_len is not None else T) if decode else T
    for i in range(config.n_layers):
        x = _layer(
            g, x, i, B=B, T=t_q, d=d, H=H, kvH=kvH, hd=hd, kv_dim=kv_dim,
            vocab=vocab, db=db, drop=drop,
            use_rope=config.use_rope, use_qk_norm=config.use_qk_norm,
            use_sink=config.use_attention_sink and not decode,
            decode=decode, t_k=t_k,
            use_moe=config.use_moe, use_swiglu=config.use_swiglu,
            moe_top_k=config.moe_top_k, moe_num_experts=config.moe_num_experts,
        )

    # ── final norm + head ───────────────────────────────────────────────
    x = g.add(NORM, "norm_f", "norm_f", (B, t_q, d), deps={x},
              flops=2 * B * t_q * d, bytes=B * t_q * d * db, output_name="final_norm")
    g.add(HEAD, "lm_head", "lm_head", (B, t_q, vocab), deps={x},
          flops=2 * B * t_q * vocab * d, bytes=B * t_q * vocab * db,
          output_name="logits")
    return g


def _layer(g: ComputationGraph, x: int, i: int, *, B, T, d, H, kvH, hd, kv_dim, vocab,
           db, drop, use_rope, use_qk_norm, use_sink, decode, t_k,
           use_moe, use_swiglu, moe_top_k, moe_num_experts) -> int:
    """Expand one TransformerBlock into nodes; return the new residual id."""
    prefix = f"layers.{i}"

    # ── Attention branch ────────────────────────────────────────────────
    ln1 = g.add(NORM, f"ln_1.{i}", f"{prefix}.ln_1", (B, T, d), deps={x},
                flops=2 * B * T * d, bytes=B * T * d * db)
    qkv_in = ln1
    if use_sink:
        qkv_in = g.add(CAT_SINK, f"sink.{i}", f"{prefix}.attn.sink_token",
                       (B, T, d), deps={ln1}, flops=B * d, bytes=B * T * d * db)
    qkv = g.add(GEMM, f"qkv.{i}", f"{prefix}.attn.qkv", (B, T, d + 2 * kv_dim),
                deps={qkv_in}, flops=2 * B * T * (d + 2 * kv_dim) * d,
                bytes=B * T * (d + 2 * kv_dim) * db)
    # q/k/v split + view/transpose — metadata (zero kernels).
    q = g.add(VIEW, f"q.view.{i}", f"{prefix}.attn", (B, H, T, hd), deps={qkv})
    k = g.add(VIEW, f"k.view.{i}", f"{prefix}.attn", (B, kvH, T, hd), deps={qkv})
    v = g.add(VIEW, f"v.view.{i}", f"{prefix}.attn", (B, kvH, T, hd), deps={qkv})
    if use_qk_norm:
        q = g.add(NORM, f"q_norm.{i}", f"{prefix}.attn.q_norm", (B, H, T, hd),
                  deps={q}, flops=2 * B * T * d, bytes=B * T * d * db)
        k = g.add(NORM, f"k_norm.{i}", f"{prefix}.attn.k_norm", (B, kvH, T, hd),
                  deps={k}, flops=2 * B * T * kv_dim, bytes=B * T * kv_dim * db)
    if use_rope:
        q = g.add(ROPE, f"rope_q.{i}", f"{prefix}.attn.rope_freqs", (B, H, T, hd),
                  deps={q}, flops=6 * B * T * d, bytes=B * T * d * db)
        k = g.add(ROPE, f"rope_k.{i}", f"{prefix}.attn.rope_freqs", (B, kvH, T, hd),
                  deps={k}, flops=6 * B * T * kv_dim, bytes=B * T * kv_dim * db)
    if decode:
        # KV append: k/v grow to the cached length; q stays 1 token.
        k = g.add(KV_APPEND, f"kv_append.k.{i}", f"{prefix}.attn.kv_cache",
                  (B, kvH, t_k, hd), deps={k}, flops=0, bytes=B * t_k * kvH * hd * db)
        v = g.add(KV_APPEND, f"kv_append.v.{i}", f"{prefix}.attn.kv_cache",
                  (B, kvH, t_k, hd), deps={v}, flops=0, bytes=B * t_k * kvH * hd * db)
    y = g.add(ATTN, f"attn.{i}", f"{prefix}.attn", (B, H, T, hd), deps={q, k, v},
              flops=4 * B * H * T * t_k * hd, bytes=B * T * d * db)
    y = g.add(CONTIG, f"attn.contig.{i}", f"{prefix}.attn", (B, T, d), deps={y},
              flops=0, bytes=B * T * d * db)
    o_proj = g.add(GEMM, f"o_proj.{i}", f"{prefix}.attn.o_proj", (B, T, d), deps={y},
                   flops=2 * B * T * d * d, bytes=B * T * d * db)
    attn_out = o_proj
    if drop:
        attn_out = g.add(DROP, f"attn.drop.{i}", f"{prefix}.attn.resid_dropout",
                         (B, T, d), deps={o_proj}, flops=B * T * d, bytes=B * T * d * db)
    x = g.add(ADD, f"res_add_attn.{i}", prefix, (B, T, d), deps={x, attn_out},
              flops=B * T * d, bytes=B * T * d * db, output_name="residual")

    # ── FFN branch ──────────────────────────────────────────────────────
    ln2 = g.add(NORM, f"ln_2.{i}", f"{prefix}.ln_2", (B, T, d), deps={x},
                flops=2 * B * T * d, bytes=B * T * d * db)
    if use_moe:
        hidden = _swiglu_hidden(d)
        gate = g.add(GEMM, f"moe.gate.{i}", f"{prefix}.ffn.gate",
                     (B, T, moe_num_experts), deps={ln2},
                     flops=2 * B * T * moe_num_experts * d,
                     bytes=B * T * moe_num_experts * db)
        route = g.add(MOE_ROUTE, f"moe.route.{i}", f"{prefix}.ffn",
                      (B, T, moe_top_k), deps={gate},
                      flops=8 * B * T * moe_num_experts,
                      bytes=B * T * moe_top_k * db)
        ffn_out = g.add(MOE_GEMM, f"moe.experts.{i}", f"{prefix}.ffn.experts",
                        (B, T, d), deps={route},
                        flops=4 * B * T * d * hidden * moe_top_k,
                        bytes=B * T * d * db)
    elif use_swiglu:
        hidden = _swiglu_hidden(d)
        w13 = g.add(GEMM, f"w13.{i}", f"{prefix}.ffn.w13", (B, T, 2 * hidden),
                    deps={ln2}, flops=2 * B * T * (2 * hidden) * d,
                    bytes=B * T * 2 * hidden * db)
        act = g.add(SILU_MUL, f"silu_mul.{i}", f"{prefix}.ffn", (B, T, hidden),
                    deps={w13}, flops=5 * B * T * hidden, bytes=B * T * hidden * db)
        ffn_out = g.add(GEMM, f"w2.{i}", f"{prefix}.ffn.w2", (B, T, d), deps={act},
                        flops=2 * B * T * d * hidden, bytes=B * T * d * db)
    else:
        hidden = 4 * d
        c_fc = g.add(GEMM, f"c_fc.{i}", f"{prefix}.ffn.c_fc", (B, T, hidden),
                     deps={ln2}, flops=2 * B * T * hidden * d,
                     bytes=B * T * hidden * db)
        act = g.add(ACT, f"gelu.{i}", f"{prefix}.ffn.gelu", (B, T, hidden), deps={c_fc},
                    flops=6 * B * T * hidden, bytes=B * T * hidden * db)
        ffn_out = g.add(GEMM, f"c_proj.{i}", f"{prefix}.ffn.c_proj", (B, T, d), deps={act},
                        flops=2 * B * T * d * hidden, bytes=B * T * d * db)
    if drop:
        ffn_out = g.add(DROP, f"ffn.drop.{i}", f"{prefix}.ffn.dropout",
                        (B, T, d), deps={ffn_out}, flops=B * T * d, bytes=B * T * d * db)
    return g.add(ADD, f"res_add_ffn.{i}", prefix, (B, T, d), deps={x, ffn_out},
                 flops=B * T * d, bytes=B * T * d * db, output_name="residual")
