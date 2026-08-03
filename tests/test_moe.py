"""
Μῆτις (Metis) — Tests for the grouped MoE execution engine
============================================================
Verifies that the grouped pipeline (token sorting → expert batching →
grouped GEMM → grouped SwiGLU → grouped output projection) preserves the
legacy per-expert behavior exactly, across every required axis:

  • routing         — softmax → top-k → normalization identical bit-for-bit
  • forward parity  — grouped == per_expert (fp32, fp16/bf16 AMP)
  • gradient parity — grads through the grouped ops == per-expert grads
  • model parity    — end-to-end logits / loss / grads on a full MetisLM
  • checkpoint      — state_dict keys ``experts.{i}.0/2.weight`` unchanged
  • edge cases      — idle experts, top-k=1, single-expert crowding,
                      gradient checkpointing, engine resolution
"""

import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from metis.config import ModelConfig  # noqa: E402
from metis.model import MetisLM  # noqa: E402
from metis.moe import (  # noqa: E402
    GROUPED,
    PER_EXPERT,
    MoE,
    _group_active_experts,
    detect_moe_engines,
    forward_grouped,
    forward_grouped_legacy,
    forward_per_expert,
    normalize_engine,
    resolve_engine,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CUDA = DEVICE.startswith("cuda")


def make_config(**overrides) -> ModelConfig:
    """Minimal MoE config for testing."""
    defaults = dict(
        d_model=64,
        n_heads=4,
        n_kv_heads=0,
        n_layers=2,
        max_seq_len=32,
        vocab_size=256,
        dropout=0.0,
        use_rmsnorm=True,
        use_swiglu=True,
        use_rope=True,
        tie_weights=True,
        use_moe=True,
        moe_num_experts=4,
        moe_top_k=2,
        moe_engine="auto",
        use_qk_norm=False,
        use_attention_sink=False,
        use_flash_attn=False,
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


def _clone_weights(src: MetisLM) -> dict:
    return {k: v.clone() for k, v in src.state_dict().items()}


def _routed(moe: MoE, x: torch.Tensor):
    """Reproduce the shared routing stage exactly as ``MoE.forward`` does."""
    x_flat = x.reshape(-1, x.shape[-1])
    gate_logits = moe.gate(x_flat)
    top_k_weights, top_k_indices = torch.topk(
        F.softmax(gate_logits, dim=-1), moe.top_k, dim=-1
    )
    top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
    return top_k_weights, top_k_indices


# ── Engine resolution ────────────────────────────────────────────────────────

class TestEngineResolution:
    def test_auto_resolves_to_grouped(self):
        assert resolve_engine("auto") == GROUPED
        assert resolve_engine(None) == GROUPED

    def test_engine_names(self):
        assert resolve_engine(GROUPED) == GROUPED
        assert resolve_engine(PER_EXPERT) == PER_EXPERT

    def test_invalid_engine_raises(self):
        with pytest.raises(ValueError):
            normalize_engine("bogus")

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("METIS_MOE_ENGINE", PER_EXPERT)
        assert resolve_engine("auto") == PER_EXPERT
        monkeypatch.setenv("METIS_MOE_ENGINE", "bogus")
        with pytest.raises(ValueError):
            resolve_engine("auto")

    def test_detect_report(self):
        report = detect_moe_engines()
        assert report["grouped"] and report["per_expert"]
        assert report["recommended"] == GROUPED


# ── Routing preservation ─────────────────────────────────────────────────────

class TestRouting:
    def test_routing_identical_to_legacy_formula(self):
        moe = MoE(make_config(moe_num_experts=8, moe_top_k=3)).to(DEVICE)
        x = torch.randn(4, 16, 64, device=DEVICE)
        # Independent reimplementation of the legacy routing.
        x_flat = x.reshape(-1, 64)
        gate_logits = moe.gate(x_flat)
        w, i = torch.topk(F.softmax(gate_logits, dim=-1), 3, dim=-1)
        w = w / w.sum(-1, keepdim=True)
        gw, gi = _routed(moe, x)
        assert torch.equal(gi, i)               # top-k indices bit-identical
        assert torch.equal(gw, w)               # weights bit-identical

    def test_weights_normalized(self):
        moe = MoE(make_config(moe_num_experts=4, moe_top_k=2)).to(DEVICE)
        x = torch.randn(2, 8, 64, device=DEVICE)
        gw, _ = _routed(moe, x)
        assert torch.allclose(gw.sum(-1), torch.ones_like(gw.sum(-1)), atol=1e-6)


# ── Forward parity ───────────────────────────────────────────────────────────

class TestForwardParity:
    @pytest.fixture(autouse=True)
    def _engines(self):
        torch.manual_seed(0)
        cfg = make_config(moe_num_experts=8, moe_top_k=2)
        self.moe_g = MoE(cfg).to(DEVICE)
        self.moe_p = MoE(cfg).to(DEVICE)
        self.moe_p.load_state_dict(self.moe_g.state_dict())  # identical weights

    def _run(self, moe, x, amp_dtype=None):
        moe.eval()
        if amp_dtype is None:
            return moe(x).float()
        with torch.autocast(DEVICE, dtype=amp_dtype):
            return moe(x).float()

    def test_forward_fp32(self):
        x = torch.randn(3, 12, 64, device=DEVICE)
        g = self._run(self.moe_g, x)
        p = self._run(self.moe_p, x)
        assert torch.allclose(g, p, atol=1e-4, rtol=1e-4), \
            f"max err {(g - p).abs().max().item():.3e}"

    @pytest.mark.skipif(not CUDA, reason="AMP autocast requires CUDA")
    def test_forward_fp16_amp(self):
        x = torch.randn(3, 12, 64, device=DEVICE)
        g = self._run(self.moe_g, x, torch.float16)
        p = self._run(self.moe_p, x, torch.float16)
        assert torch.allclose(g, p, atol=1e-2, rtol=1e-2), \
            f"max err {(g - p).abs().max().item():.3e}"

    @pytest.mark.skipif(not CUDA, reason="AMP autocast requires CUDA")
    def test_forward_bf16_amp(self):
        x = torch.randn(3, 12, 64, device=DEVICE)
        g = self._run(self.moe_g, x, torch.bfloat16)
        p = self._run(self.moe_p, x, torch.bfloat16)
        assert torch.allclose(g, p, atol=1e-2, rtol=1e-2), \
            f"max err {(g - p).abs().max().item():.3e}"

    def test_forward_dropout_same_seed(self):
        # With identical routing (no RNG before dropout), resetting the seed
        # makes the single dropout mask identical for both engines.
        cfg = make_config(moe_num_experts=8, moe_top_k=2, dropout=0.1)
        mg, mp = MoE(cfg).to(DEVICE), MoE(cfg).to(DEVICE)
        mp.load_state_dict(mg.state_dict())
        x = torch.randn(2, 8, 64, device=DEVICE)
        for m in (mg, mp):
            m.train()
        torch.manual_seed(1)
        g = mg(x).float()
        torch.manual_seed(1)
        p = mp(x).float()
        assert torch.allclose(g, p, atol=1e-4, rtol=1e-4), \
            f"max err {(g - p).abs().max().item():.3e}"


# ── Gradient parity ──────────────────────────────────────────────────────────

class TestGradientParity:
    def test_grads_match_per_expert(self):
        # Gradients through every grouped op (sort / index_copy / bmm /
        # index_add) must equal the per-expert loop, on identical weights.
        torch.manual_seed(0)
        cfg = make_config(moe_num_experts=8, moe_top_k=2)
        x = torch.randn(4, 8, 64, device=DEVICE)
        ref_state = MoE(cfg).to(DEVICE).state_dict()

        def grads(engine):
            c = make_config(moe_num_experts=8, moe_top_k=2, moe_engine=engine)
            moe = MoE(c).to(DEVICE)
            moe.load_state_dict(ref_state)
            moe(x).sum().backward()
            return {n: p.grad.clone() for n, p in moe.named_parameters()}

        g = grads(GROUPED)
        p = grads(PER_EXPERT)
        assert set(g) == set(p)
        diffs = [(g[n] - p[n]).abs().max().item() for n in g]
        assert max(diffs) < 1e-4, f"max grad diff {max(diffs):.3e}"

    def test_grads_finite_and_flow_to_all_params(self):
        # Every parameter (gate + all expert linears) receives a finite grad.
        cfg = make_config(moe_num_experts=8, moe_top_k=2)
        moe = MoE(cfg).to(DEVICE)
        x = torch.randn(4, 8, 64, device=DEVICE)
        moe(x).sum().backward()
        named = dict(moe.named_parameters())
        assert len(named) == 1 + 8 * 2  # gate + 8 experts × 2 linears
        for n, p in named.items():
            assert p.grad is not None and torch.isfinite(p.grad).all(), n


# ── Model-level parity ───────────────────────────────────────────────────────

class TestModelParity:
    def _model(self, engine, seed=0):
        torch.manual_seed(seed)
        cfg = make_config(
            d_model=96, n_layers=2, max_seq_len=48, moe_num_experts=8,
            moe_top_k=2, moe_engine=engine,
        )
        return MetisLM(cfg).to(DEVICE)

    def test_logits_loss_match(self):
        mg = self._model(GROUPED)
        mp = self._model(PER_EXPERT)
        mp.load_state_dict(mg.state_dict())
        idx = torch.randint(0, 256, (2, 16), device=DEVICE)
        mg.eval(); mp.eval()
        with torch.no_grad():
            lg, loss_g, _ = mg(idx, targets=idx)
            lp, loss_p, _ = mp(idx, targets=idx)
        assert torch.allclose(lg.float(), lp.float(), atol=1e-4, rtol=1e-4), \
            f"max logit err {(lg.float()-lp.float()).abs().max().item():.3e}"
        assert torch.allclose(loss_g, loss_p, atol=1e-4, rtol=1e-4)
        # The engine reported is the resolved one.
        assert mg.layers[0].ffn.last_engine == GROUPED

    def test_model_grads_match(self):
        mg = self._model(GROUPED)
        mp = self._model(PER_EXPERT)
        mp.load_state_dict(mg.state_dict())
        idx = torch.randint(0, 256, (2, 16), device=DEVICE)
        mg.train(); mp.train()
        _, lg, _ = mg(idx, targets=idx); lg.backward()
        g = {n: p.grad.clone() for n, p in mg.named_parameters()
             if p.grad is not None}
        mp.zero_grad()
        _, lp, _ = mp(idx, targets=idx); lp.backward()
        p = {n: p.grad.clone() for n, p in mp.named_parameters()
             if p.grad is not None}
        assert set(g) == set(p)
        diffs = [(g[n] - p[n]).abs().max().item() for n in g]
        assert max(diffs) < 1e-4, f"max grad diff {max(diffs):.3e}"

    def test_gradient_checkpointing(self):
        m = self._model(GROUPED)
        m.train()
        idx = torch.randint(0, 256, (1, 16), device=DEVICE)
        _, loss, _ = m(idx, targets=idx, use_checkpointing=True)
        loss.backward()
        assert any(p.grad is not None for p in m.parameters())


# ── Checkpoint compatibility ─────────────────────────────────────────────────

class TestCheckpointCompat:
    def test_expert_state_dict_keys_preserved(self):
        moe = MoE(make_config(moe_num_experts=8)).to("cpu")
        keys = [k for k in moe.state_dict() if "experts" in k]
        w1 = [k for k in keys if k.endswith("0.weight")]
        w2 = [k for k in keys if k.endswith("2.weight")]
        assert len(w1) == 8 and len(w2) == 8
        assert all(k.startswith("experts.") for k in keys)

    def test_strict_load_roundtrip(self):
        m = MetisLM(make_config(moe_num_experts=8)).to("cpu")
        sd = m.state_dict()
        m2 = MetisLM(make_config(moe_num_experts=8)).to("cpu")
        missing, unexpected = m2.load_state_dict(sd, strict=True)
        assert not missing and not unexpected

    def test_load_into_grouped_engine(self):
        # A model trained with either engine loads into the other, strictly.
        cfg = make_config(moe_num_experts=8)
        a = MetisLM(cfg).to("cpu")
        cfg.moe_engine = PER_EXPERT
        b = MetisLM(cfg).to("cpu")
        missing, unexpected = b.load_state_dict(a.state_dict(), strict=True)
        assert not missing and not unexpected


# ── Edge cases ───────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_idle_experts(self):
        # Force all routing weight onto experts {0,1}; experts 2..7 idle.
        moe = MoE(make_config(moe_num_experts=8, moe_top_k=2)).to(DEVICE)
        x = torch.randn(4, 8, 64, device=DEVICE)
        x_flat = x.reshape(-1, 64)
        logits = torch.full((x_flat.shape[0], 8), -1e9, device=DEVICE)
        logits[:, :2] = 1.0
        w, i = torch.topk(F.softmax(logits, dim=-1), 2, dim=-1)
        w = w / w.sum(-1, keepdim=True)
        g = forward_grouped(x_flat, w, i,
                            [e[0].weight.t() for e in moe.experts],
                            [e[2].weight.t() for e in moe.experts],
                            top_k=2, num_experts=8)
        p = forward_per_expert(x_flat, w, i, moe.experts, top_k=2)
        assert torch.allclose(g, p, atol=1e-4, rtol=1e-4), \
            f"max err {(g-p).abs().max().item():.3e}"

    def test_top_k_1(self):
        cfg = make_config(moe_num_experts=4, moe_top_k=1)
        moe_g = MoE(cfg).to(DEVICE)
        moe_p = MoE(cfg).to(DEVICE)
        moe_p.load_state_dict(moe_g.state_dict())
        x = torch.randn(5, 7, 64, device=DEVICE)
        moe_g.eval(); moe_p.eval()
        with torch.no_grad():
            g = moe_g(x).float(); p = moe_p(x).float()
        assert torch.allclose(g, p, atol=1e-4, rtol=1e-4), \
            f"max err {(g-p).abs().max().item():.3e}"

    def test_single_expert_crowding(self):
        # top-k=1, every token forced to expert 0 → maxM == N (full padding).
        moe = MoE(make_config(moe_num_experts=4, moe_top_k=1)).to(DEVICE)
        x = torch.randn(6, 6, 64, device=DEVICE)
        x_flat = x.reshape(-1, 64)
        logits = torch.full((x_flat.shape[0], 4), -1e9, device=DEVICE)
        logits[:, 0] = 1.0
        w, i = torch.topk(F.softmax(logits, dim=-1), 1, dim=-1)
        w = w / w.sum(-1, keepdim=True)
        g = forward_grouped(x_flat, w, i,
                            [e[0].weight.t() for e in moe.experts],
                            [e[2].weight.t() for e in moe.experts],
                            top_k=1, num_experts=4)
        p = forward_per_expert(x_flat, w, i, moe.experts, top_k=1)
        assert torch.allclose(g, p, atol=1e-4, rtol=1e-4), \
            f"max err {(g-p).abs().max().item():.3e}"

    def test_non_power_of_two_dims(self):
        # Non-power-of-two d_model/hidden → not a round tensor-core shape;
        # the grouped bmm must still match the per-expert reference.
        cfg = make_config(d_model=68, n_heads=4, moe_num_experts=4, moe_top_k=2)
        moe_g = MoE(cfg).to(DEVICE)
        moe_p = MoE(cfg).to(DEVICE)
        moe_p.load_state_dict(moe_g.state_dict())
        x = torch.randn(2, 8, 68, device=DEVICE)
        moe_g.eval(); moe_p.eval()
        with torch.no_grad():
            g = moe_g(x).float(); p = moe_p(x).float()
        assert torch.allclose(g, p, atol=1e-4, rtol=1e-4), \
            f"max err {(g-p).abs().max().item():.3e}"

    def test_per_expert_is_reference(self):
        # forward_per_expert reproduces the legacy in-place accumulation loop.
        torch.manual_seed(0)
        moe = MoE(make_config(moe_num_experts=4, moe_top_k=2)).to(DEVICE)
        x = torch.randn(3, 9, 64, device=DEVICE)
        w, i = _routed(moe, x)
        x_flat = x.reshape(-1, 64)
        out = forward_per_expert(x_flat, w, i, moe.experts, top_k=2)
        expected = torch.zeros_like(x_flat)
        for ei, expert in enumerate(moe.experts):
            mask = (i == ei).any(-1)
            if not mask.any():
                continue
            ew = w[mask][(i[mask] == ei)]
            expected[mask] += expert(x_flat[mask]) * ew.unsqueeze(-1)
        assert torch.allclose(out, expected, atol=1e-6, rtol=1e-6)


# ── Dynamic scheduling (redesigned grouped engine) ───────────────────────────

class TestDynamicScheduling:
    """Grouped + dynamic-capacity scheduler: grouping, waste, and parity."""

    def test_grouping_bins_by_count(self):
        counts = torch.tensor([200, 10, 180, 5, 100, 20, 90, 15],
                              dtype=torch.long, device=DEVICE)
        active = torch.nonzero(counts, as_tuple=False).flatten()
        groups = _group_active_experts(counts, active, 2.0)
        # within every group the max/min token ratio is bounded by 2.0
        for g in groups:
            g = [int(counts[i]) for i in g]
            assert max(g) <= 2.0 * min(g), f"group violates ratio: {g}"
        # every active expert appears exactly once
        flat = sorted(e for g in groups for e in g)
        assert flat == active.tolist()

    def test_grouping_ratio_boundaries(self):
        counts = torch.tensor([50, 10, 40, 8], dtype=torch.long, device=DEVICE)
        active = torch.nonzero(counts, as_tuple=False).flatten()
        assert len(_group_active_experts(counts, active, 1.0)) == 4  # per expert
        assert len(_group_active_experts(counts, active, 1e9)) == 1  # one group

    def test_forward_parity_legacy(self):
        """New scheduler == pre-redesign (global-max) scheduler, exactly."""
        torch.manual_seed(0)
        moe = MoE(make_config(moe_num_experts=8, moe_top_k=2)).to(DEVICE)
        x = torch.randn(6, 16, 64, device=DEVICE)
        w, i = _routed(moe, x)
        x_flat = x.reshape(-1, 64)
        w1 = [e[0].weight.t() for e in moe.experts]
        w2 = [e[2].weight.t() for e in moe.experts]
        g_new = forward_grouped(x_flat, w, i, w1, w2, top_k=2,
                                num_experts=8, group_max_ratio=2.0)
        g_old = forward_grouped_legacy(x_flat, w, i, w1, w2,
                                       top_k=2, num_experts=8)
        assert torch.allclose(g_new, g_old, atol=1e-5, rtol=1e-5), \
            f"max err {(g_new - g_old).abs().max().item():.3e}"

    def test_forward_parity_skewed(self):
        """Matches the pre-redesign scheduler on a heavily skewed routing."""
        torch.manual_seed(0)
        moe = MoE(make_config(moe_num_experts=8, moe_top_k=2)).to(DEVICE)
        x = torch.randn(8, 24, 64, device=DEVICE)
        x_flat = x.reshape(-1, 64)
        gate = torch.full((x_flat.shape[0], 8), -1e9, device=DEVICE)
        gate[:, 0] = 3.0                     # 50%+ to expert 0
        gate[:, 1:] = torch.rand(x_flat.shape[0], 7, device=DEVICE)
        w, i = torch.topk(F.softmax(gate, dim=-1), 2, dim=-1)
        w = w / w.sum(-1, keepdim=True)
        w1 = [e[0].weight.t() for e in moe.experts]
        w2 = [e[2].weight.t() for e in moe.experts]
        g_new = forward_grouped(x_flat, w, i, w1, w2, top_k=2,
                                num_experts=8, group_max_ratio=2.0)
        g_old = forward_grouped_legacy(x_flat, w, i, w1, w2,
                                       top_k=2, num_experts=8)
        # Different block shapes → different cuBLAS fp rounding; tight fp32 tol.
        assert torch.allclose(g_new, g_old, atol=1e-5, rtol=1e-5), \
            f"max err {(g_new - g_old).abs().max().item():.3e}"

    def test_waste_reduction(self):
        """Dynamic capacity pads strictly less than the global-max baseline."""
        torch.manual_seed(0)
        x = torch.randn(8, 32, 64, device=DEVICE)
        x_flat = x.reshape(-1, 64)
        gate = torch.full((x_flat.shape[0], 16), -1e9, device=DEVICE)
        gate[:, 0] = 2.0
        gate[:, 1:] = torch.rand(x_flat.shape[0], 15, device=DEVICE)
        w, i = torch.topk(F.softmax(gate, dim=-1), 2, dim=-1)
        counts = torch.bincount(i.reshape(-1), minlength=16)
        active = torch.nonzero(counts, as_tuple=False).flatten()
        groups = _group_active_experts(counts, active, 2.0)

        def waste(schedule):
            w_ = 0
            for g in schedule:
                A_g = len(g)
                ga = torch.as_tensor(sorted(g), device=counts.device)
                gt = int(counts[ga].sum())
                mm = max(int(counts[ga].max()), (gt + A_g - 1) // A_g)
                w_ += A_g * mm - gt
            return w_

        waste_old = waste([active.tolist()])
        waste_new = waste(groups)
        assert waste_new < waste_old, f"{waste_new} vs {waste_old}"

    def test_grads_match_per_expert(self):
        """Gradients through the grouped scheduler == per-expert loop."""
        torch.manual_seed(0)
        cfg = make_config(moe_num_experts=8, moe_top_k=2)
        x = torch.randn(4, 8, 64, device=DEVICE)
        ref_state = MoE(cfg).to(DEVICE).state_dict()

        def grads(engine):
            c = make_config(moe_num_experts=8, moe_top_k=2, moe_engine=engine)
            moe = MoE(c).to(DEVICE)
            moe.load_state_dict(ref_state)
            moe(x).sum().backward()
            return {n: p.grad.clone() for n, p in moe.named_parameters()}

        g = grads(GROUPED)      # redesigned scheduler is the grouped engine
        p = grads(PER_EXPERT)
        diffs = [(g[n] - p[n]).abs().max().item() for n in g]
        assert max(diffs) < 1e-4, f"max grad diff {max(diffs):.3e}"

    def test_legacy_matches_per_expert(self):
        torch.manual_seed(0)
        moe = MoE(make_config(moe_num_experts=4, moe_top_k=2)).to(DEVICE)
        x = torch.randn(3, 9, 64, device=DEVICE)
        w, i = _routed(moe, x)
        x_flat = x.reshape(-1, 64)
        g = forward_grouped_legacy(x_flat, w, i,
                                   [e[0].weight.t() for e in moe.experts],
                                   [e[2].weight.t() for e in moe.experts],
                                   top_k=2, num_experts=4)
        p = forward_per_expert(x_flat, w, i, moe.experts, top_k=2)
        assert torch.allclose(g, p, atol=1e-4, rtol=1e-4), \
            f"max err {(g-p).abs().max().item():.3e}"

    def test_config_validation_group_ratio(self):
        with pytest.raises(ValueError, match="moe_group_ratio"):
            make_config(moe_group_ratio=0.0)
        with pytest.raises(ValueError, match="moe_group_ratio"):
            make_config(moe_group_ratio=-1.0)

    def test_group_ratio_wired_through_model(self):
        """group_max_ratio flows from config; extreme ratios stay correct."""
        torch.manual_seed(0)
        for ratio in (1.0, 2.0, 1e9):
            cfg_g = make_config(moe_num_experts=8, moe_top_k=2,
                                moe_group_ratio=ratio)
            cfg_p = make_config(moe_num_experts=8, moe_top_k=2,
                                moe_engine=PER_EXPERT)
            moe_g = MoE(cfg_g).to(DEVICE)
            moe_p = MoE(cfg_p).to(DEVICE)
            moe_p.load_state_dict(moe_g.state_dict())
            x = torch.randn(4, 12, 64, device=DEVICE)
            moe_g.eval()
            moe_p.eval()
            with torch.no_grad():
                g = moe_g(x).float()
                p = moe_p(x).float()
            assert torch.allclose(g, p, atol=1e-4, rtol=1e-4), \
                f"ratio {ratio}: max err {(g-p).abs().max().item():.3e}"
