"""
Metis — Tests for the persistent expert weight cache
======================================================
Verifies that the ExpertCache produces bit-identical outputs to the uncached
path, correctly detects staleness, evicts within capacity, and provides
accurate statistics. Covers both inference and training (autograd) scenarios.

  • forward parity       — cache-on vs cache-off: bit-identical outputs
  • cache-off unchanged  — moe_cache_size=0 produces the exact old path
  • hit rate             — repeated forwards with same input accumulate hits
  • LRU / eviction       — capacity 1, two groups → oldest evicted
  • staleness detection  — param.copy_() forces rebuild
  new weights reflected
  • byte accounting      — bytes_built + bytes_saved = total remat bytes
  • model-level          — MetisLM with cache on/off → same logits
  • config validation    — negative sizes raise
  • env override         — METIS_MOE_CACHE_SIZE respected
  • device transfer      — .to() resets cache
  • stats API            — all fields present and consistent
"""

import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from metis.config import ModelConfig  # noqa: E402
from metis.expert_cache import ExpertCache, _remat_bytes  # noqa: E402
from metis.model import MetisLM  # noqa: E402
from metis.moe import (  # noqa: E402
    MoE,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CUDA = DEVICE.startswith("cuda")


def make_config(**overrides) -> ModelConfig:
    defaults = dict(
        d_model=64, n_heads=4, n_kv_heads=0, n_layers=2, max_seq_len=32,
        vocab_size=256, dropout=0.0, use_rmsnorm=True, use_swiglu=True,
        use_rope=True, tie_weights=True, use_moe=True, moe_num_experts=4,
        moe_top_k=2, moe_engine="auto", use_qk_norm=False,
        use_attention_sink=False, use_flash_attn=False, moe_cache_size=16,
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


def _routed(moe, x_flat):
    gate_logits = moe.gate(x_flat)
    top_k_w, top_k_i = torch.topk(
        F.softmax(gate_logits, dim=-1), moe.top_k, dim=-1
    )
    top_k_w = top_k_w / top_k_w.sum(dim=-1, keepdim=True)
    return top_k_w, top_k_i


# ── ExpertCache unit tests ──────────────────────────────────────────────


class TestExpertCacheUnit:
    def test_disabled_cache_passthrough(self):
        c = ExpertCache(entry_capacity=0)
        w1 = torch.randn(2, 4)
        sources = [w1]
        result = c.get_or_build((0,), torch.float32, sources, lambda: (w1, w1))
        assert result[0] is w1
        assert c.misses == 1 and c.hits == 0

    def test_hit_and_miss(self):
        c = ExpertCache(entry_capacity=16)
        w1 = torch.randn(2, 4, requires_grad=True)
        w2 = torch.randn(4, 2, requires_grad=True)
        sources = [w1, w2]
        r1 = c.get_or_build((0, 1), torch.float32, sources,
                            lambda: (w1.clone(), w2.clone()))
        r2 = c.get_or_build((0, 1), torch.float32, sources,
                            lambda: (w1.clone(), w2.clone()))
        assert c.hits == 1 and c.misses == 1
        assert torch.equal(r1[0], r2[0])

    def test_stale_entry_replaced(self):
        c = ExpertCache(entry_capacity=16)
        w1 = torch.randn(2, 4, requires_grad=True)
        w2 = torch.randn(4, 2, requires_grad=True)
        sources = [w1, w2]
        c.get_or_build((0,), torch.float32, sources,
                       lambda: (w1.clone(), w2.clone()))
        # Simulate weight mutation via copy_ (bumps version)
        with torch.no_grad():
            w1.copy_(torch.randn_like(w1))
        c.get_or_build((0,), torch.float32, sources,
                       lambda: (w1.clone(), w2.clone()))
        assert c.misses == 2  # rebuilt after staleness

    def test_byte_accounting(self):
        c = ExpertCache(entry_capacity=64)
        w1 = torch.randn(3, 5, requires_grad=True)
        w2 = torch.randn(5, 3, requires_grad=True)
        sources = [w1, w2]
        built = (w1.clone().to(torch.bfloat16), w2.clone().to(torch.bfloat16))
        expected_remat = _remat_bytes(sources, built[0])
        read_per = (built[0].numel() * built[0].element_size()
                    + built[1].numel() * built[1].element_size())
        c.get_or_build((0, 1, 2), torch.bfloat16, sources, lambda: built)
        c.get_or_build((0, 1, 2), torch.bfloat16, sources, lambda: built)
        assert c.bytes_built == expected_remat
        assert c.bytes_saved == expected_remat
        assert c.bytes_read == read_per * 2  # 2 lookups
        remat_total = expected_remat * 2
        assert c.bytes_built + c.bytes_saved == remat_total
        sc = 100.0 * expected_remat / remat_total
        assert abs(c.stats()["stackcast_avoided_pct"] - sc) < 1e-6
        bw = 100.0 * expected_remat / (remat_total + read_per * 2)
        assert abs(c.stats()["bandwidth_reduction_pct"] - bw) < 1e-6

    def test_lru_eviction(self):
        c = ExpertCache(entry_capacity=2)
        w = torch.randn(2, 2)
        for i in range(3):
            c.get_or_build((i,), torch.float32, [w], lambda w=w: (w.clone(), w.clone()))
        assert len(c._entries) == 2
        assert c.evictions == 1

    def test_invalidate_keeps_stats(self):
        c = ExpertCache(entry_capacity=4)
        w = torch.randn(2, 2)
        c.get_or_build((0,), torch.float32, [w], lambda: (w.clone(), w.clone()))
        c.invalidate()
        assert len(c._entries) == 0
        assert c.hits == 0 and c.misses == 1  # stats preserved

    def test_reset_clears_everything(self):
        c = ExpertCache(entry_capacity=4)
        w = torch.randn(2, 2)
        c.get_or_build((0,), torch.float32, [w], lambda: (w.clone(), w.clone()))
        c.reset()
        assert len(c._entries) == 0
        assert c.hits == 0 and c.misses == 0

    def test_stats_completeness(self):
        c = ExpertCache(entry_capacity=8)
        s = c.stats()
        for k in ("enabled", "entry_capacity", "entries", "hits", "misses",
                   "hit_rate", "evictions", "bytes_saved", "bytes_built",
                   "bytes_read", "bandwidth_reduction_pct", "stackcast_avoided_pct",
                   "resident_bytes", "byte_capacity", "oversized_skips"):
            assert k in s, f"missing key: {k}"

    def test_requires_grad_keying(self):
        """no_grad build (inference) and grad build (training) are separate entries."""
        c = ExpertCache(entry_capacity=16)
        w1 = torch.randn(2, 4, requires_grad=True)
        w2 = torch.randn(4, 2, requires_grad=True)
        sources = [w1, w2]
        # Inference (no_grad)
        with torch.no_grad():
            c.get_or_build((0,), torch.float32, sources,
                           lambda: (w1.clone(), w2.clone()))
        # Training (grad enabled)
        c.get_or_build((0,), torch.float32, sources,
                       lambda: (w1.clone(), w2.clone()))
        assert c.misses == 2  # two separate entries
        assert len(c._entries) == 2

    def test_repr(self):
        c = ExpertCache(entry_capacity=4)
        r = repr(c)
        assert "ExpertCache" in r and "entries=0/4" in r


# ── Forward parity: cache-on vs cache-off ────────────────────────────────


class TestForwardParity:
    @pytest.fixture(autouse=True)
    def _engines(self):
        torch.manual_seed(0)
        cfg_cache = make_config(moe_num_experts=8, moe_top_k=2, moe_cache_size=16)
        self.moe_cache = MoE(cfg_cache).to(DEVICE)
        cfg_nocache = make_config(moe_num_experts=8, moe_top_k=2, moe_cache_size=0)
        self.moe_nocache = MoE(cfg_nocache).to(DEVICE)
        self.moe_nocache.load_state_dict(self.moe_cache.state_dict())
        # The reference must be genuinely cache-off (cache None), else the
        # 'cache-off' side would itself be a cached model.
        assert self.moe_nocache._cache is None

    def test_fp32_bit_identical(self):
        """Cache-on (with a real hit) must equal a genuinely cache-off model."""
        x = torch.randn(3, 12, 64, device=DEVICE)
        self.moe_cache.eval()
        self.moe_nocache.eval()
        with torch.no_grad():
            r_nocache = self.moe_nocache(x)   # reference: cache-off
            self.moe_cache(x)                  # warm up (miss)
            r_cache = self.moe_cache(x)        # cache hit
        # Sanity: the cached side really did hit
        stats = self.moe_cache.cache_stats()
        assert stats["hits"] >= 1, f"expected a cache hit, got {stats}"
        assert torch.equal(r_cache, r_nocache), \
            f"max err {(r_cache - r_nocache).abs().max().item():.3e}"

    def test_repeated_forward_bit_identical(self):
        x = torch.randn(2, 8, 64, device=DEVICE)
        self.moe_cache.eval()
        with torch.no_grad():
            o1 = self.moe_cache(x)
            o2 = self.moe_cache(x)  # should hit → identical
        assert torch.equal(o1, o2)

    @pytest.mark.skipif(not CUDA, reason="AMP requires CUDA")
    def test_fp16_amp(self):
        x = torch.randn(3, 12, 64, device=DEVICE)
        self.moe_cache.eval()
        self.moe_nocache.eval()
        with torch.no_grad(), torch.autocast(DEVICE, dtype=torch.float16):
            r_nocache = self.moe_nocache(x).float()   # reference: cache-off
            self.moe_cache(x)                          # warm up (miss)
            r_cache = self.moe_cache(x).float()        # cache hit
        assert torch.allclose(r_cache, r_nocache, atol=1e-2, rtol=1e-2)

    @pytest.mark.skipif(not CUDA, reason="AMP requires CUDA")
    def test_bf16_amp(self):
        x = torch.randn(3, 12, 64, device=DEVICE)
        self.moe_cache.eval()
        self.moe_nocache.eval()
        with torch.no_grad(), torch.autocast(DEVICE, dtype=torch.bfloat16):
            r_nocache = self.moe_nocache(x).float()   # reference: cache-off
            self.moe_cache(x)                          # warm up (miss)
            r_cache = self.moe_cache(x).float()        # cache hit
        assert torch.allclose(r_cache, r_nocache, atol=1e-2, rtol=1e-2)


# ── Staleness detection ─────────────────────────────────────────────────


class TestStaleness:
    def test_copy_forces_rebuild(self):
        torch.manual_seed(0)
        cfg = make_config(moe_num_experts=4, moe_top_k=2, moe_cache_size=16)
        m = MetisLM(cfg).to(DEVICE)
        m.eval()
        idx = torch.randint(0, 256, (2, 16), device=DEVICE)
        with torch.no_grad():
            lg1, _, _ = m(idx)
        # Mutate expert weight in-place via .data (fused-optimizer pattern)
        m.layers[0].ffn.experts[0][0].weight.data.copy_(
            torch.randn_like(m.layers[0].ffn.experts[0][0].weight)
        )
        # Framework invalidation (as training.py does after optimizer.step)
        m.invalidate_moe_caches()
        with torch.no_grad():
            lg2, _, _ = m(idx)
        # Outputs must differ after mutation + invalidation
        assert not torch.equal(lg1, lg2)


# ── Model-level parity ──────────────────────────────────────────────────


class TestModelParity:
    def _model(self, cache_size):
        torch.manual_seed(42)
        cfg = make_config(
            d_model=96, n_layers=2, max_seq_len=48,
            moe_num_experts=8, moe_top_k=2, moe_cache_size=cache_size,
        )
        return MetisLM(cfg).to(DEVICE)

    def test_logits_bit_identical(self):
        m_cache = self._model(16)
        m_no = self._model(0)
        m_no.load_state_dict(m_cache.state_dict())
        idx = torch.randint(0, 256, (2, 16), device=DEVICE)
        m_cache.eval()
        m_no.eval()
        with torch.no_grad():
            lg_c, _, _ = m_cache(idx)
            lg_n, _, _ = m_no(idx)
        assert torch.equal(lg_c, lg_n)

    def test_get_moe_cache_stats(self):
        m = self._model(16)
        stats = m.get_moe_cache_stats()
        assert len(stats) == m.config.n_layers
        # Non-MoE layers (if any) return None, MoE layers return dicts
        moe_stats = [s for s in stats if s is not None]
        assert len(moe_stats) > 0
        for s in moe_stats:
            assert "hit_rate" in s and "entries" in s

    def test_invalidate_moe_caches(self):
        m = self._model(16)
        m.eval()
        idx = torch.randint(0, 256, (2, 16), device=DEVICE)
        with torch.no_grad():
            m(idx)
        stats_before = m.get_moe_cache_stats()
        any_entries = any(
            s["entries"] > 0 for s in stats_before if s
        )
        assert any_entries
        m.invalidate_moe_caches()
        stats_after = m.get_moe_cache_stats()
        for s in stats_after:
            if s is not None:
                assert s["entries"] == 0


# ── Gradient parity (training path) ─────────────────────────────────────


class TestGradientParity:
    def test_grads_match_nocache(self):
        torch.manual_seed(0)
        cfg = make_config(moe_num_experts=8, moe_top_k=2, moe_cache_size=16)
        m_c = MetisLM(cfg).to(DEVICE)
        m_n = MetisLM(cfg).to(DEVICE)
        m_n.load_state_dict(m_c.state_dict())
        idx = torch.randint(0, 256, (2, 16), device=DEVICE)
        m_c.train()
        m_n.train()
        _, loss_c, _ = m_c(idx, targets=idx)
        loss_c.backward()
        # Disable cache for reference
        for layer in m_n.layers:
            layer.ffn._cache = None if hasattr(layer.ffn, '_cache') else None
        _, loss_n, _ = m_n(idx, targets=idx)
        loss_n.backward()
        gc = {n: p.grad.clone() for n, p in m_c.named_parameters() if p.grad is not None}
        gn = {n: p.grad.clone() for n, p in m_n.named_parameters() if p.grad is not None}
        diffs = [(gc[n] - gn[n]).abs().max().item() for n in gc if n in gn]
        assert max(diffs) < 1e-5, f"max grad diff {max(diffs):.3e}"


# ── Config / env / device ───────────────────────────────────────────────


class TestConfigAndDevice:
    def test_negative_size_raises(self):
        with pytest.raises(ValueError, match="moe_cache_size"):
            make_config(moe_cache_size=-1)

    def test_negative_bytes_raises(self):
        with pytest.raises(ValueError, match="moe_cache_bytes"):
            make_config(moe_cache_bytes=-5)

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("METIS_MOE_CACHE_SIZE", "32")
        cfg = make_config()
        m = MoE(cfg).to(DEVICE)
        assert m._cache is not None
        assert m._cache.entry_capacity == 32

    def test_env_disable(self, monkeypatch):
        monkeypatch.setenv("METIS_MOE_CACHE_SIZE", "0")
        cfg = make_config(moe_cache_size=16)
        m = MoE(cfg).to(DEVICE)
        assert m._cache is None

    def test_to_device_resets_cache(self):
        cfg = make_config(moe_num_experts=4, moe_cache_size=16)
        m = MoE(cfg).to(DEVICE)
        x = torch.randn(2, 8, 64, device=DEVICE)
        m.eval()
        with torch.no_grad():
            m(x)
        assert m._cache._resident > 0
        m.to(DEVICE)  # same device → _apply called → reset
        assert m._cache._resident == 0


# ── LRU capacity ────────────────────────────────────────────────────────


class TestLRUCapacity:
    def test_entries_bounded(self):
        cfg = make_config(moe_num_experts=8, moe_top_k=2, moe_cache_size=2)
        m = MoE(cfg).to(DEVICE)
        m.eval()
        # Force different groups by routing
        with torch.no_grad():
            for _ in range(10):
                x = torch.randn(1, 8, 64, device=DEVICE)
                m(x)
        stats = m.cache_stats()
        assert stats["entries"] <= 2
        assert stats["evictions"] >= 0


# ── Checkpoint round-trip (cache not in state_dict) ─────────────────────


class TestCheckpointCompat:
    def test_state_dict_unchanged(self):
        cfg = make_config(moe_num_experts=4, moe_cache_size=16)
        m = MetisLM(cfg).to("cpu")
        sd_before = set(m.state_dict().keys())
        # Populate cache
        m.eval()
        with torch.no_grad():
            m(torch.randint(0, 256, (1, 16)))
        sd_after = set(m.state_dict().keys())
        assert sd_before == sd_after  # cache not in state_dict


# ── New tests: hardening, AMP, gradient-hit, oversize, shape, thread-safety ──


class TestOversizeEntrySkip:
    """Oversize entry (resident > byte_capacity) must be skipped, not cached."""

    def test_oversize_skipped(self):
        c = ExpertCache(entry_capacity=100, byte_capacity=2000)
        # Small entry fits: 2×2×4 = 16 bytes per tensor = 32 bytes resident
        w = torch.randn(2, 2)
        c.get_or_build((0,), torch.float32, [w], lambda: (w.clone(), w.clone()))
        assert len(c._entries) == 1
        # Large entry: 2×20×4 = 320 bytes per tensor = 640 bytes resident > 2000? No.
        # Need actual large: 2×100×4 = 800 per tensor → 1600 resident < 2000.
        # Use a smaller budget to force eviction.
        c2 = ExpertCache(entry_capacity=100, byte_capacity=500)
        c2 = ExpertCache(entry_capacity=100, byte_capacity=500)
        w1 = torch.randn(2, 2)
        c2.get_or_build((0,), torch.float32, [w1], lambda: (w1.clone(), w1.clone()))
        assert len(c2._entries) == 1
        w2 = torch.randn(16, 16)  # 16×16×4×2 = 2048 resident > 500
        c2.get_or_build((1,), torch.float32, [w2], lambda: (w2.clone(), w2.clone()))
        assert c2.oversized_skips == 1
        assert len(c2._entries) == 1  # small entry still resident


class TestShapeStaleness:
    """Same data_ptr + _version but different shape must be detected as stale."""

    def test_reshape_triggers_miss(self):
        c = ExpertCache(entry_capacity=16)
        w = torch.randn(2, 3, requires_grad=True)
        sources = [w]
        c.get_or_build((0,), torch.float32, sources, lambda: (w.clone(), w.clone()))
        # Reshape in-place: new view, same storage, same version
        w2 = w.view(6, 1)
        sources2 = [w2]
        c.get_or_build((0,), torch.float32, sources2, lambda: (w2.clone(), w2.clone()))
        assert c.misses == 2  # stale because shape differs


class TestGradHitPath:
    """Gradient parity on the CACHE HIT path: micro-batch 2 hits cached tensor."""

    def test_grads_match_nocache_on_hit(self):
        torch.manual_seed(0)
        cfg = make_config(moe_num_experts=4, moe_top_k=2, moe_cache_size=16)
        m_c = MetisLM(cfg).to(DEVICE)
        m_n = MetisLM(cfg).to(DEVICE)
        m_n.load_state_dict(m_c.state_dict())
        idx1 = torch.randint(0, 256, (2, 16), device=DEVICE)
        idx2 = torch.randint(0, 256, (2, 16), device=DEVICE)
        m_c.train()
        m_n.train()
        # Two micro-batches: second should hit the cache built by the first
        _, loss1, _ = m_c(idx1, targets=idx1)
        loss1 = loss1 / 2
        loss1.backward()
        _, loss2, _ = m_c(idx2, targets=idx2)
        loss2 = loss2 / 2
        loss2.backward()
        torch.nn.utils.clip_grad_norm_(m_c.parameters(), 1.0)
        # Verify cache was hit
        stats = m_c.get_moe_cache_stats()
        total_hits = sum(s["hits"] for s in stats if s)
        assert total_hits > 0, f"expected cache hits, got {total_hits}"
        # Disable cache for reference
        for layer in m_n.layers:
            if hasattr(layer.ffn, "_cache"):
                layer.ffn._cache = None
        _, loss_n1, _ = m_n(idx1, targets=idx1)
        loss_n1 = loss_n1 / 2
        loss_n1.backward()
        _, loss_n2, _ = m_n(idx2, targets=idx2)
        loss_n2 = loss_n2 / 2
        loss_n2.backward()
        torch.nn.utils.clip_grad_norm_(m_n.parameters(), 1.0)
        gc = {n: p.grad.clone() for n, p in m_c.named_parameters() if p.grad is not None}
        gn = {n: p.grad.clone() for n, p in m_n.named_parameters() if p.grad is not None}
        diffs = [(gc[n] - gn[n]).abs().max().item() for n in gc if n in gn]
        assert max(diffs) < 1e-5, f"max grad diff {max(diffs):.3e}"


class TestThreadSafety:
    """Concurrent get_or_build must not corrupt LRU or counters."""

    def test_concurrent_no_crash(self):
        import concurrent.futures

        c = ExpertCache(entry_capacity=4)
        w = torch.randn(2, 2)

        def access(i):
            return c.get_or_build(
                (i % 3,), torch.float32, [w],
                lambda w=w: (w.clone(), w.clone()),
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(access, range(20)))

        assert len(c._entries) <= 4
        assert c._resident >= 0


class TestBytesRead:
    """Verify bytes_read and both bandwidth metrics are consistent."""

    def test_bytes_read_tracking(self):
        c = ExpertCache(entry_capacity=16)
        w1 = torch.randn(2, 4, requires_grad=True)
        w2 = torch.randn(4, 2, requires_grad=True)
        sources = [w1, w2]
        built = (w1.clone(), w2.clone())
        read_per = (built[0].numel() * built[0].element_size()
                    + built[1].numel() * built[1].element_size())
        # 3 different groups → 3 misses
        for i in range(3):
            c.get_or_build((i,), torch.float32, sources, lambda: built)
        s = c.stats()
        assert s["bytes_read"] == read_per * 3
        assert s["stackcast_avoided_pct"] == 0.0  # no hits → no avoided
        assert s["bandwidth_reduction_pct"] == 0.0
        # Now one hit: same group → bytes_saved > 0
        c.get_or_build((0,), torch.float32, sources, lambda: built)
        s2 = c.stats()
        assert s2["stackcast_avoided_pct"] > 0
        assert s2["bandwidth_reduction_pct"] > 0
        # stackcast_avoided_pct > bandwidth_reduction_pct (reads dilute the latter)
        assert s2["stackcast_avoided_pct"] > s2["bandwidth_reduction_pct"]
