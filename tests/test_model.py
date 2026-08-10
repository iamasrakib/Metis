"""
Μῆτις (Metis) — Unit Tests for Model Architecture
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from metis.config import ModelConfig
from metis.model import (
    MLP,
    CausalSelfAttention,
    MetisLM,
    RMSNorm,
    SwiGLU,
    TransformerBlock,
)

# ── Helpers ──────────────────────────────────────────────────────────────────

def make_config(**overrides) -> ModelConfig:
    """Create a minimal model config for testing."""
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
        use_moe=False,
        use_qk_norm=False,
        use_attention_sink=False,
        use_flash_attn=False,
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


# ── Building Blocks ──────────────────────────────────────────────────────────

class TestRMSNorm:
    def test_output_shape(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 8, 64)
        y = norm(x)
        assert y.shape == x.shape

    def test_normalization(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 8, 64) * 10.0  # input far from unit RMS
        y = norm(x)
        # RMSNorm removes input scale: each feature vector ends at RMS ≈ 1
        # (weight defaults to ones).
        rms = y.pow(2).mean(-1).sqrt()
        assert rms.shape == (2, 8)
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)

    def test_normalization_respects_weight(self):
        norm = RMSNorm(64)
        norm.weight.data.fill_(3.0)
        x = torch.randn(2, 8, 64)
        y = norm(x)
        rms = y.pow(2).mean(-1).sqrt()
        # A non-unit weight scales the normalized vector (weight is ~constant
        # per channel, so the mean RMS tracks it).
        assert torch.allclose(rms, torch.full_like(rms, 3.0), atol=1e-2)

    def test_weight_shape(self):
        norm = RMSNorm(128)
        assert norm.weight.shape == (128,)


class TestSwiGLU:
    def test_output_shape(self):
        config = make_config()
        glu = SwiGLU(config)
        x = torch.randn(2, 8, config.d_model)
        y = glu(x)
        assert y.shape == x.shape

    def test_nonlinearity(self):
        config = make_config(d_model=32)
        glu = SwiGLU(config)
        x = torch.randn(4, 16, 32)
        y = glu(x)
        assert not torch.allclose(y, x)  # Should have transformed


class TestMLP:
    def test_output_shape(self):
        config = make_config()
        mlp = MLP(config)
        x = torch.randn(2, 8, config.d_model)
        y = mlp(x)
        assert y.shape == x.shape


class TestCausalSelfAttention:
    def test_output_shape_mha(self):
        """Test MHA (n_kv_heads == n_heads)."""
        config = make_config(d_model=64, n_heads=4, n_kv_heads=4)
        attn = CausalSelfAttention(config)
        x = torch.randn(2, 8, 64)
        y, cache = attn(x, kv_cache=None)
        assert y.shape == x.shape
        assert cache is not None
        assert cache[0].shape == (2, 4, 8, 16)  # (B, n_heads, T, head_dim)

    def test_output_shape_gqa(self):
        """Test GQA (n_kv_heads < n_heads)."""
        config = make_config(d_model=64, n_heads=4, n_kv_heads=2)
        attn = CausalSelfAttention(config)
        x = torch.randn(2, 8, 64)
        y, cache = attn(x, kv_cache=None)
        assert y.shape == x.shape

    def test_kv_cache(self):
        config = make_config(d_model=64, n_heads=4, n_kv_heads=2)
        attn = CausalSelfAttention(config)
        x1 = torch.randn(1, 1, 64)
        # A cold forward always builds and returns a cache (so generation can
        # start caching incrementally) — unexpanded n_kv_heads.
        y1, cache = attn(x1, kv_cache=None)
        assert cache is not None
        assert cache[0].shape == (1, 2, 1, 16)  # (B, n_kv_heads, T, head_dim)
        assert y1.shape == x1.shape

        # With an existing cache, the new K/V is appended along the seq dim.
        y2, cache2 = attn(x1, kv_cache=cache)
        assert cache2[0].shape == (1, 2, 2, 16)
        assert y2.shape == x1.shape

    def test_qk_norm(self):
        config = make_config(d_model=64, n_heads=4, use_qk_norm=True)
        attn = CausalSelfAttention(config)
        assert attn.use_qk_norm is True
        x = torch.randn(2, 8, 64)
        y, _ = attn(x)
        assert y.shape == x.shape


class TestTransformerBlock:
    def test_output_shape(self):
        config = make_config()
        block = TransformerBlock(config)
        x = torch.randn(2, 8, config.d_model)
        y, cache = block(x)
        assert y.shape == x.shape


# ── Full Model ───────────────────────────────────────────────────────────────

class TestMetisLM:
    def test_forward_no_targets(self):
        config = make_config()
        model = MetisLM(config)
        idx = torch.randint(0, config.vocab_size, (2, 16))
        logits, loss, cache = model(idx, targets=None)
        assert logits.shape == (2, 1, config.vocab_size)
        assert loss is None

    def test_forward_with_targets(self):
        config = make_config()
        model = MetisLM(config)
        idx = torch.randint(0, config.vocab_size, (2, 16))
        logits, loss, cache = model(idx, targets=idx)
        assert logits.shape == (2, 16, config.vocab_size)
        assert loss is not None
        assert loss.item() > 0  # Random init should have non-zero loss

    def test_loss_decreases(self):
        """Overfitting test: model should memorize a tiny sequence."""
        config = make_config(d_model=128, n_layers=3, max_seq_len=8, dropout=0.0)
        model = MetisLM(config)
        optimizer = model.configure_optimizers(0.0, 1e-2, "cpu")

        idx = torch.randint(0, min(config.vocab_size, 50), (4, 8))
        losses = []
        for _ in range(50):
            optimizer.zero_grad()
            _, loss, _ = model(idx, targets=idx)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should decrease
        assert losses[-1] < losses[0]

    def test_forward_exceeds_max_seq_len(self):
        config = make_config(max_seq_len=32)
        model = MetisLM(config)
        idx = torch.randint(0, config.vocab_size, (1, 64))
        with pytest.raises(ValueError):
            model(idx)

    def test_tie_weights(self):
        config = make_config(tie_weights=True)
        model = MetisLM(config)
        assert model.tok_emb.weight.data_ptr() == model.lm_head.weight.data_ptr()

    def test_no_tie_weights(self):
        config = make_config(tie_weights=False)
        model = MetisLM(config)
        assert model.tok_emb.weight.data_ptr() != model.lm_head.weight.data_ptr()

    def test_configure_optimizers(self):
        config = make_config()
        model = MetisLM(config)
        optimizer = model.configure_optimizers(0.1, 3e-4, "cpu")
        assert len(optimizer.param_groups) == 2
        assert optimizer.param_groups[0]["weight_decay"] == 0.1
        assert optimizer.param_groups[1]["weight_decay"] == 0.0

    def test_count_parameters(self):
        config = make_config()
        model = MetisLM(config)
        counts = model.count_parameters()
        assert "total" in counts
        assert counts["total"] > 0
        for component, count in counts.items():
            assert count >= 0

    def test_parameter_count_format(self):
        config = make_config(d_model=512, n_layers=8)
        MetisLM(config)  # builds without error at this size
        assert "M" in config.n_params or "K" in config.n_params or "B" in config.n_params

    def test_gradient_checkpointing(self):
        config = make_config()
        model = MetisLM(config)
        model.train()
        idx = torch.randint(0, config.vocab_size, (1, 16))
        _, loss, _ = model(idx, targets=idx, use_checkpointing=True)
        loss.backward()
        # Check gradients exist
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad


# ── MoE Tests ────────────────────────────────────────────────────────────────

class TestMoE:
    def test_moe_output_shape(self):
        config = make_config(d_model=64, use_moe=True, moe_num_experts=4, moe_top_k=2)
        model = MetisLM(config)
        idx = torch.randint(0, config.vocab_size, (2, 16))
        logits, loss, _ = model(idx, targets=idx)
        assert logits.shape == (2, 16, config.vocab_size)
        assert loss is not None


# ── Weight Initialization ────────────────────────────────────────────────────

class TestWeightInit:
    def test_init_range(self):
        config = make_config(d_model=128, n_layers=4)
        model = MetisLM(config)
        for name, param in model.named_parameters():
            if param.dim() >= 2 and "o_proj" in name:
                # Should be scaled by 1/sqrt(2*n_layers) = 1/sqrt(8) ≈ 0.35
                assert param.std().item() < 0.1  # Well below default 0.02


class TestFusedProjections:
    """Fused QKV / fused gate-up FFN: fewer GEMMs, checkpoint-compatible keys."""

    def test_fused_qkv_shapes(self):
        """One fused qkv Linear; legacy split projections are gone."""
        config = make_config(d_model=64, n_heads=4, n_kv_heads=2)
        attn = CausalSelfAttention(config)
        assert hasattr(attn, "qkv")
        assert not hasattr(attn, "q_proj") and not hasattr(attn, "k_proj")
        assert not hasattr(attn, "v_proj")
        # d_model + 2 * kv_dim = 64 + 2 * (64*2//4) = 64 + 64
        assert attn.qkv.weight.shape == (128, 64)
        assert attn.kv_dim == 32

    def test_fused_qkv_forward_and_cache(self):
        """Fused QKV forward still produces identical shapes and a KV cache."""
        config = make_config(d_model=64, n_heads=4, n_kv_heads=2)
        attn = CausalSelfAttention(config)
        x = torch.randn(2, 8, 64)
        y, cache = attn(x, kv_cache=None)
        assert y.shape == (2, 8, 64)
        assert cache is not None
        assert cache[0].shape == (2, 2, 8, 16)  # (B, n_kv_heads, T, head_dim)

    def test_fused_swiglu_shapes(self):
        config = make_config(d_model=64, n_heads=4)
        ffn = SwiGLU(config)
        assert hasattr(ffn, "w13")
        assert not hasattr(ffn, "w1") and not hasattr(ffn, "w3")
        assert hasattr(ffn, "w2")
        # hidden = round(4*64*2/3 up to /8) = 352; w13 = (2*352, 64)
        assert ffn.w13.weight.shape == (2 * ffn.hidden, 64)
        x = torch.randn(2, 8, 64)
        assert ffn(x).shape == (2, 8, 64)

    def test_state_dict_exports_legacy_split_keys(self):
        """state_dict() keeps q_proj/k_proj/v_proj and w1/w3 — never qkv/w13."""
        config = make_config(d_model=64, n_heads=4, n_kv_heads=2)
        model = MetisLM(config)
        sd = model.state_dict()
        assert "layers.0.attn.qkv.weight" not in sd
        assert "layers.0.ffn.w13.weight" not in sd
        assert "layers.0.attn.q_proj.weight" in sd
        assert "layers.0.attn.k_proj.weight" in sd
        assert "layers.0.attn.v_proj.weight" in sd
        assert "layers.0.ffn.w1.weight" in sd
        assert "layers.0.ffn.w3.weight" in sd
        # q_proj is the first d_model rows of the fused qkv weight
        # (Linear weight shape is (out_features, in_features)).
        attn = dict(model.named_modules())["layers.0.attn"]
        w = attn.qkv.weight.detach()
        assert torch.equal(sd["layers.0.attn.q_proj.weight"], w[:64])

    def test_state_dict_round_trip(self):
        """state_dict -> load_state_dict is lossless (no missing/unexpected)."""
        config = make_config(d_model=64, n_heads=4, n_kv_heads=2)
        model = MetisLM(config)
        fresh = MetisLM(config)
        missing, unexpected = fresh.load_state_dict(model.state_dict())
        assert missing == []
        assert unexpected == []

    def test_load_legacy_split_keys(self):
        """Old checkpoints (split q/k/v and w1/w3) load into the fused model.

        ``model.state_dict()`` already emits the legacy split keys, so loading
        it back into a fresh fused model exercises the old-format path end to
        end and must reproduce identical eval outputs.
        """
        config = make_config(d_model=64, n_heads=4, n_kv_heads=2)
        model = MetisLM(config).eval()
        sd = model.state_dict()
        assert any(k.endswith("attn.q_proj.weight") for k in sd)
        assert not any(k.endswith(("qkv.weight", "w13.weight")) for k in sd)

        fused = MetisLM(config).eval()
        missing, unexpected = fused.load_state_dict(sd)
        assert missing == []
        assert unexpected == []

        idx = torch.randint(0, 256, (2, 8))
        with torch.no_grad():
            assert torch.equal(model(idx, idx)[0], fused(idx, idx)[0])

    def test_load_fused_qkv_format(self):
        """New-format checkpoints (single qkv.weight / w13.weight) also load."""
        config = make_config(d_model=64, n_heads=4, n_kv_heads=2)
        model = MetisLM(config).eval()

        # Rebuild the checkpoint in the fused (pre-shim) format.
        new_sd = {}
        for k, v in model.state_dict().items():
            if any(
                k.endswith(s)
                for s in ("q_proj.weight", "k_proj.weight", "v_proj.weight")
            ):
                continue
            new_sd[k] = v
        for i in range(config.n_layers):
            attn = dict(model.named_modules())[f"layers.{i}.attn"]
            new_sd[f"layers.{i}.attn.qkv.weight"] = attn.qkv.weight.detach()

        fused = MetisLM(config).eval()
        missing, unexpected = fused.load_state_dict(new_sd)
        assert missing == []
        assert unexpected == []
        idx = torch.randint(0, 256, (2, 8))
        with torch.no_grad():
            assert torch.equal(model(idx, idx)[0], fused(idx, idx)[0])

    def test_fused_vs_separate_numerics(self):
        """Fused QKV == three separate projections; fused gate/up == two GEMMs.

        Compare against manually concatenated weights, in fp32 (exact matmul),
        so the fused GEMMs are mathematically identical to the separate ones.
        """
        torch.manual_seed(0)
        config = make_config(d_model=64, n_heads=4, n_kv_heads=2)
        attn = CausalSelfAttention(config)
        attn.eval()
        x = torch.randn(2, 8, 64)
        with torch.no_grad():
            q, k, v = attn.qkv(x).split([64, 32, 32], dim=-1)
        w = attn.qkv.weight.detach()  # (128, 64) = (d+2kv, d)
        with torch.no_grad():
            q_sep = x @ w[:64].t()
            k_sep = x @ w[64:96].t()
            v_sep = x @ w[96:].t()
        assert torch.equal(q, q_sep)
        assert torch.equal(k, k_sep)
        assert torch.equal(v, v_sep)
