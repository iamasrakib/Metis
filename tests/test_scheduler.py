"""
Tests for the graph-based execution scheduler (metis/scheduler/).

Covers: computation-graph analysis, cost model, planner, arena allocator, and
the execution scheduler — including bit-identical parity against the eager
forward across MoE on/off, QK-norm, attention sink, GQA, KV-cache decode, and
both infer and train modes. All tests are CPU-runnable.
"""

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from metis.config import ModelConfig
from metis.model import MetisLM, SwiGLU
from metis.scheduler import (
    INFER,
    NOOP,
    TRAIN,
    ComputationGraph,
    ExecutionPlan,
    analyze_model,
    assign,
    build_scheduler,
    is_metadata,
    plan_execution,
)

# ── Helpers ──────────────────────────────────────────────────────────────

def make_config(**overrides) -> ModelConfig:
    defaults = dict(
        vocab_size=256, d_model=64, n_heads=4, n_kv_heads=0, n_layers=2,
        max_seq_len=32, dropout=0.0, use_rmsnorm=True, use_swiglu=True,
        use_rope=True, tie_weights=True, use_moe=False, use_qk_norm=False,
        use_attention_sink=False, moe_num_experts=8, moe_top_k=2,
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


def make_model(**overrides) -> MetisLM:
    cfg = make_config(**overrides)
    return MetisLM(cfg)


# ── Graph tests ──────────────────────────────────────────────────────────

class TestComputationGraph:
    def test_basic_graph_build(self):
        model = make_model()
        model.eval()
        g = analyze_model(model, model.config, ref_shape=(1, 16))
        assert len(g.nodes) > 0
        # 2-layer SwiGLU: embed + 2 blocks × (norm+qkv+views+rope+attn+
        # contig+gemm_o+add+norm+gemm_w13+act+gemm_w2+add) + norm_f + head
        assert len(g.nodes) >= 30

    def test_topological_order_valid(self):
        model = make_model()
        model.eval()
        g = analyze_model(model, model.config, ref_shape=(1, 16))
        order = g.topological_order()
        # every node appears after all its deps
        pos = {nid: i for i, nid in enumerate(order)}
        for nid, node in g.nodes.items():
            for dep in node.deps:
                assert pos[dep] < pos[nid], f"node {nid} depends on {dep} but dep is after"

    def test_critical_path_ends_at_head(self):
        model = make_model()
        model.eval()
        g = analyze_model(model, model.config, ref_shape=(1, 16))
        for node in g.nodes.values():
            node.est_ms = 1.0  # uniform to test dep chain
        cp = g.critical_path()
        assert len(cp) >= 3
        last = g.nodes[cp[-1]]
        assert last.kind == "head"

    def test_liveness(self):
        model = make_model()
        model.eval()
        g = analyze_model(model, model.config, ref_shape=(1, 16))
        g.compute_liveness()
        for node in g.nodes.values():
            assert node.order >= 0
            assert node.dead_at >= node.order

    def test_moe_config(self):
        model = make_model(use_moe=True, use_swiglu=True)
        model.eval()
        g = analyze_model(model, model.config, ref_shape=(1, 16))
        kinds = [n.kind for n in g.nodes.values()]
        assert "moe_route" in kinds
        assert "moe_gemm" in kinds

    def test_mlp_config(self):
        model = make_model(use_swiglu=False)
        model.eval()
        g = analyze_model(model, model.config, ref_shape=(1, 16))
        kinds = [n.kind for n in g.nodes.values()]
        assert "act" in kinds  # GELU MLP uses ACT kind

    def test_qk_norm_config(self):
        model = make_model(use_qk_norm=True)
        model.eval()
        g = analyze_model(model, model.config, ref_shape=(1, 16))
        names = [n.name for n in g.nodes.values()]
        assert any("q_norm" in n for n in names)
        assert any("k_norm" in n for n in names)

    def test_sink_config(self):
        model = make_model(use_attention_sink=True)
        model.eval()
        g = analyze_model(model, model.config, ref_shape=(1, 16))
        names = [n.name for n in g.nodes.values()]
        assert any("sink" in n for n in names)

    def test_decode_graph(self):
        model = make_model()
        model.eval()
        g = analyze_model(model, model.config, ref_shape=(1, 1), decode=True, cache_len=16)
        assert g.decode
        kinds = [n.kind for n in g.nodes.values()]
        assert "kv_append" in kinds

    def test_gqa_config(self):
        model = make_model(n_kv_heads=2)
        model.eval()
        g = analyze_model(model, model.config, ref_shape=(1, 16))
        # qkv output dim = d + 2*kv_dim where kv_dim = d*kvH/H
        assert len(g.nodes) > 20

    def test_json_roundtrip(self):
        model = make_model()
        model.eval()
        g = analyze_model(model, model.config, ref_shape=(1, 16))
        data = g.to_dict()
        g2 = ComputationGraph.from_dict(data)
        assert len(g2.nodes) == len(g.nodes)
        for nid in g.nodes:
            assert g2.nodes[nid].kind == g.nodes[nid].kind

    def test_metadata_detection(self):
        assert is_metadata("view")
        assert is_metadata("noop")
        assert not is_metadata("gemm")
        assert not is_metadata("norm")

    def test_noop_node_detected(self):
        g = ComputationGraph()
        a = g.add("gemm", "a", "a", (4, 4), deps=None, bytes=64)
        g.add(NOOP, "b", "b", None, deps={a}, bytes=0)
        from metis.scheduler.planner import detect_dead_nodes
        removed = detect_dead_nodes(g)
        assert len(removed) == 1

    def test_zero_dep_node_has_no_deps(self):
        model = make_model()
        model.eval()
        g = analyze_model(model, model.config, ref_shape=(1, 16))
        srcs = g.sources()
        assert len(srcs) >= 1
        for s in srcs:
            assert not g.nodes[s].deps


# ── Cost model tests ─────────────────────────────────────────────────────

class TestCost:
    def test_estimate_costs_fills_est_ms(self):
        model = make_model()
        model.eval()
        g = analyze_model(model, model.config, ref_shape=(1, 16))
        from metis.scheduler.cost import estimate_costs
        estimate_costs(g, "cpu")
        for node in g.nodes.values():
            if not is_metadata(node.kind):
                assert node.est_ms >= 0

    def test_gemm_dominates_norm(self):
        model = make_model()
        model.eval()
        g = analyze_model(model, model.config, ref_shape=(1, 16))
        from metis.scheduler.cost import estimate_costs
        estimate_costs(g, "cpu")
        qkv0 = [n for n in g.nodes.values() if n.name == "qkv.0"][0]
        ln0 = [n for n in g.nodes.values() if n.name == "ln_1.0"][0]
        assert qkv0.est_ms > ln0.est_ms  # GEMM slower than norm

    def test_calibration_scale(self):
        model = make_model()
        model.eval()
        from metis.scheduler.cost import calibrate, estimate_costs
        g = analyze_model(model, model.config, ref_shape=(1, 16))
        estimate_costs(g, "cpu")
        scale = calibrate(g, model, model.config, device="cpu", iters=2)
        assert scale > 0


# ── Buffer allocator tests ───────────────────────────────────────────────

class TestBuffers:
    def test_assign_reuses_slots(self):
        g = ComputationGraph()
        # two non-overlapping nodes
        a = g.add("gemm", "a", "a", (1,), bytes=100)
        b = g.add("gemm", "b", "b", (1,), deps={a}, bytes=200)
        g.nodes[a].dead_at = 0
        g.nodes[b].order = 1
        g.nodes[b].dead_at = 1
        result = assign([g.nodes[a], g.nodes[b]])
        # b is born after a dies → it must reuse a's slot exactly.
        assert result.slots == 1
        assert result.reuse_count == 1

    def test_overlap_needs_two_slots(self):
        g = ComputationGraph()
        a = g.add("gemm", "a", "a", (1,), bytes=100)
        b = g.add("gemm", "b", "b", (1,), deps={a}, bytes=200)
        g.nodes[a].dead_at = 1   # a alive at position 1
        g.nodes[b].order = 0     # b at position 0 (overlap!)
        g.nodes[b].dead_at = 0
        # both live at position 0 simultaneously
        result = assign([g.nodes[a], g.nodes[b]])
        assert result.slots >= 2  # need 2 slots

    def test_arena_below_naive(self):
        g = ComputationGraph()
        a = g.add("gemm", "a", "a", (1,), bytes=100)
        b = g.add("gemm", "b", "b", (1,), deps={a}, bytes=200)
        g.nodes[a].dead_at = 0
        g.nodes[b].order = 1
        g.nodes[b].dead_at = 1
        result = assign([g.nodes[a], g.nodes[b]])
        assert result.arena_bytes <= result.naive_peak

    def test_naive_peak_sums_live(self):
        g = ComputationGraph()
        a = g.add("gemm", "a", "a", (1,), bytes=100)
        b = g.add("gemm", "b", "b", (1,), deps={a}, bytes=200)
        g.nodes[a].dead_at = 0
        g.nodes[b].order = 1
        g.nodes[b].dead_at = 1
        peak = assign([g.nodes[a], g.nodes[b]]).naive_peak
        # a: live at pos 0 (100), b: live at pos 1 (200) -> peak 200
        assert peak >= 200


# ── Planner tests ────────────────────────────────────────────────────────

class TestPlanner:
    def test_plan_builds(self):
        model = make_model()
        model.eval()
        plan = plan_execution(model, model.config, mode=INFER, device="cpu",
                              ref_shape=(1, 16), calibrate_run=False)
        assert isinstance(plan, ExecutionPlan)
        assert len(plan.order) > 0
        assert len(plan.critical_path) > 0

    def test_plan_arena_infer_smaller_than_naive(self):
        model = make_model()
        model.eval()
        plan = plan_execution(model, model.config, mode=INFER, device="cpu",
                              ref_shape=(1, 16), calibrate_run=False)
        assert plan.arena_bytes < plan.naive_peak

    def test_plan_train_no_folded_allocs(self):
        model = make_model()
        model.eval()
        plan = plan_execution(model, model.config, mode=TRAIN, device="cpu",
                              ref_shape=(1, 16), calibrate_run=False)
        # train: no in-place folding
        assert plan.folded_allocs == 0

    def test_plan_residual_slot_zero(self):
        model = make_model()
        model.eval()
        plan = plan_execution(model, model.config, mode=INFER, device="cpu",
                              ref_shape=(1, 16), calibrate_run=False)
        residual_nodes = [n for n in plan.graph.nodes.values()
                          if plan.slot_of.get(n.id) == 0]
        assert len(residual_nodes) >= 3  # embed + 2x (attn_add + ffn_add) = 5

    def test_plan_folded_allocs(self):
        model = make_model()
        model.eval()
        plan = plan_execution(model, model.config, mode=INFER, device="cpu",
                              ref_shape=(1, 16), calibrate_run=False)
        # 2-layer: 4 residual adds + 2 silu_muls = 6
        assert plan.folded_allocs == 6

    def test_plan_sync_points_zero(self):
        model = make_model()
        model.eval()
        plan = plan_execution(model, model.config, mode=INFER, device="cpu",
                              ref_shape=(1, 16), calibrate_run=False)
        assert plan.sync_points == 0

    def test_plan_critical_path_is_sequential(self):
        model = make_model()
        model.eval()
        plan = plan_execution(model, model.config, mode=INFER, device="cpu",
                              ref_shape=(1, 16), calibrate_run=False)
        # critical path should include both blocks
        assert len(plan.critical_path) >= 10

    def test_plan_json_roundtrip(self):
        model = make_model()
        model.eval()
        plan = plan_execution(model, model.config, mode=INFER, device="cpu",
                              ref_shape=(1, 16), calibrate_run=False)
        data = plan.to_dict()
        plan2 = ExecutionPlan.from_dict(data)
        assert len(plan2.order) == len(plan.order)
        assert plan2.arena_bytes == plan.arena_bytes

    def test_plan_render(self):
        model = make_model()
        model.eval()
        plan = plan_execution(model, model.config, mode=INFER, device="cpu",
                              ref_shape=(1, 16), calibrate_run=False)
        text = plan.render()
        assert "Metis exec plan" in text
        assert "tok_emb" in text

    def test_plan_summary(self):
        model = make_model()
        model.eval()
        plan = plan_execution(model, model.config, mode=INFER, device="cpu",
                              ref_shape=(1, 16), calibrate_run=False)
        s = plan.summary()
        assert "nodes" in s
        assert "arena_bytes" in s


# ── Runtime parity tests ────────────────────────────────────────────────

class TestInferParity:
    def test_prefill_match(self):
        model = make_model()
        model.eval()
        sched = build_scheduler(model, mode=INFER, calibrate_run=False,
                                ref_shape=(1, 16))
        idx = torch.randint(0, 256, (1, 16))
        with torch.no_grad():
            log_s, _, _ = sched.execute(idx)
            log_e, _, _ = model(idx)
        assert torch.equal(log_s, log_e)

    def test_decode_match(self):
        model = make_model()
        model.eval()
        sched = build_scheduler(model, mode=INFER, calibrate_run=False,
                                ref_shape=(1, 16))
        idx = torch.randint(0, 256, (1, 16))
        with torch.no_grad():
            _, _, cache = sched.execute(idx)
            _, _, cache_e = model(idx)
        for step in range(5):
            tok = torch.randint(0, 256, (1, 1))
            log_s, _, cache = sched.execute(tok, kv_cache=cache)
            log_e, _, cache_e = model(tok, kv_cache=cache_e)
            assert torch.equal(log_s, log_e), f"mismatch at decode step {step}"

    def test_decode_match_no_rope(self):
        """Non-RoPE positional path: decode positions advance via cache length.

        Regression for the scheduler runtime's non-RoPE branch — without RoPE
        the absolute position of a decode step comes from the KV cache length
        (``cached_len_of(kv_cache)``), which the eager model and the scheduler
        must agree on token-for-token.
        """
        model = make_model(use_rope=False)
        model.eval()
        sched = build_scheduler(model, mode=INFER, calibrate_run=False,
                                ref_shape=(1, 16))
        idx = torch.randint(0, 256, (1, 16))
        with torch.no_grad():
            _, _, cache = sched.execute(idx)
            _, _, cache_e = model(idx)
        for step in range(5):
            tok = torch.randint(0, 256, (1, 1))
            log_s, _, cache = sched.execute(tok, kv_cache=cache)
            log_e, _, cache_e = model(tok, kv_cache=cache_e)
            assert torch.equal(log_s, log_e), f"mismatch at decode step {step}"

    def test_prefill_with_targets(self):
        model = make_model()
        model.eval()
        sched = build_scheduler(model, mode=INFER, calibrate_run=False,
                                ref_shape=(1, 16))
        idx = torch.randint(1, 256, (1, 16))
        targets = torch.randint(1, 256, (1, 16))
        with torch.no_grad():
            log_s, loss_s, _ = sched.execute(idx, targets=targets)
            log_e, loss_e, _ = model(idx, targets=targets)
        assert torch.equal(log_s, log_e)
        assert loss_s.item() == loss_e.item()

    def test_moe_parity(self):
        model = make_model(use_moe=True, moe_num_experts=4, moe_top_k=2)
        model.eval()
        sched = build_scheduler(model, mode=INFER, calibrate_run=False,
                                ref_shape=(1, 16))
        idx = torch.randint(0, 256, (1, 16))
        with torch.no_grad():
            log_s, _, _ = sched.execute(idx)
            log_e, _, _ = model(idx)
        assert torch.equal(log_s, log_e)

    def test_qk_norm_parity(self):
        model = make_model(use_qk_norm=True)
        model.eval()
        sched = build_scheduler(model, mode=INFER, calibrate_run=False,
                                ref_shape=(1, 16))
        idx = torch.randint(0, 256, (1, 16))
        with torch.no_grad():
            log_s, _, _ = sched.execute(idx)
            log_e, _, _ = model(idx)
        assert torch.equal(log_s, log_e)

    def test_sink_parity(self):
        model = make_model(use_attention_sink=True)
        model.eval()
        sched = build_scheduler(model, mode=INFER, calibrate_run=False,
                                ref_shape=(1, 16))
        idx = torch.randint(0, 256, (1, 16))
        with torch.no_grad():
            log_s, _, _ = sched.execute(idx)
            log_e, _, _ = model(idx)
        assert torch.equal(log_s, log_e)

    def test_gqa_parity(self):
        model = make_model(n_kv_heads=2)
        model.eval()
        sched = build_scheduler(model, mode=INFER, calibrate_run=False,
                                ref_shape=(1, 16))
        idx = torch.randint(0, 256, (1, 16))
        with torch.no_grad():
            log_s, _, _ = sched.execute(idx)
            log_e, _, _ = model(idx)
        assert torch.equal(log_s, log_e)

    def test_inplace_add_bit_identical(self):
        x = torch.randn(3, 4)
        y = torch.randn(3, 4)
        a = x.clone()
        a.add_(y)
        b = x + y
        assert torch.equal(a, b)

    def test_silu_mul_fold_bit_identical(self):
        """SwiGLU in-place fold: silu + mul on narrow views must be bit-identical."""
        out = torch.randn(1, 4, 10)
        h = 5
        gate = out.narrow(-1, 0, h)
        up = out.narrow(-1, h, h)
        g = gate.clone()
        F.silu(g, inplace=True)
        g.mul_(up.clone())
        ref = F.silu(gate.clone()) * up.clone()
        assert torch.equal(g, ref)

    def test_counters(self):
        model = make_model()
        model.eval()
        sched = build_scheduler(model, mode=INFER, calibrate_run=False,
                                ref_shape=(1, 16))
        idx = torch.randint(0, 256, (1, 16))
        with torch.no_grad():
            sched.execute(idx)
        assert sched.counters["forwards"] == 1
        assert sched.counters["residual_allocs"] == 1
        assert sched.counters["syncs"] == 0

    def test_no_sync_in_source(self):
        """Static check: _execute_infer calls no .item()/.cpu()/.synchronize()."""
        import ast
        source = ast.parse(
            open(os.path.join(os.path.dirname(__file__),
                              "..", "metis", "scheduler", "runtime.py"),
                 encoding="utf-8").read()
        )
        # Docstrings are ``ast.Constant`` (str) nodes, so any Attribute hit here
        # is real code — a host-side sync in the hot loop.
        for node in ast.walk(source):
            assert not (
                isinstance(node, ast.Attribute)
                and node.attr in ("item", "cpu", "synchronize")
            ), f"host-sync call {node.attr!r} found in runtime source"


class TestTrainParity:
    def test_train_match(self):
        model = make_model()
        model.train()
        sched = build_scheduler(model, mode=TRAIN, calibrate_run=False,
                                ref_shape=(1, 16))
        idx = torch.randint(1, 256, (1, 16))
        targets = torch.randint(1, 256, (1, 16))
        log_s, loss_s, _ = sched.execute(idx, targets=targets)
        log_e, loss_e, _ = model(idx, targets=targets)
        assert torch.equal(log_s, log_e)
        assert loss_s.item() == loss_e.item()

    def test_train_eval_mode_match(self):
        model = make_model()
        model.eval()
        sched = build_scheduler(model, mode=TRAIN, calibrate_run=False,
                                ref_shape=(1, 16))
        idx = torch.randint(0, 256, (1, 16))
        with torch.no_grad():
            log_s, _, _ = sched.execute(idx)
            log_e, _, _ = model(idx)
        assert torch.equal(log_s, log_e)


# ── SwiGLU import check ─────────────────────────────────────────────────

class TestSwiGLUFold:
    def test_silu_inplace_preserves_output(self):
        """Confirm SwiGLU in-place fold and vanilla forward produce same output."""
        cfg = make_config()
        ffn = SwiGLU(cfg)
        x = torch.randn(1, 8, cfg.d_model)
        # vanilla
        out_vanilla = ffn(x)
        # manual fold using narrow (split disallows in-place)
        w13_out = ffn.w13(x)
        h = ffn.hidden
        gate = w13_out.narrow(-1, 0, h)
        up = w13_out.narrow(-1, h, h)
        F.silu(gate, inplace=True)
        gate.mul_(up)
        out_folded = ffn.dropout(ffn.w2(gate))
        assert torch.equal(out_vanilla, out_folded)
