"""
Μῆτις (Metis) — Tests for layer prefetching
============================================
Layer prefetching speculatively warms layer ``N+1``'s MoE expert cache during
layer ``N``'s compute (on a dedicated CUDA stream when available), so the next
layer's ``get_or_build`` calls hit and never stall on a synchronous
``torch.stack(views).to(dtype)``.

Covered here:
  • correctness  — prefetch ON vs OFF is bit-identical (logits, loss, grads)
  • cache API    — ``ExpertCache.prefetch`` never touches hit/miss counters;
                   a prefetched entry is served as a hit; staleness still
                   rebuilds
  • record flow  — MoE layers report their routed groups to the prefetcher
  • warm-up      — with per-step invalidation (the training loop), prefetch
                   raises the hit rate
  • gating       — prefetcher is CUDA-gated by default; ``use_layer_prefetch``
                   config flag disables it

On CPU the prefetcher is a no-op by default; tests that need it force it via
the ``METIS_LAYER_PREFETCH_FORCE_CPU`` env var.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from metis.config import ModelConfig  # noqa: E402
from metis.expert_cache import ExpertCache  # noqa: E402
from metis.model import MetisLM  # noqa: E402

CUDA = torch.cuda.is_available()


def _make(prefetch: bool, cache_size: int = 0, **kw) -> MetisLM:
    cfg = ModelConfig.from_preset(
        "tiny", vocab_size=256, device="cuda" if CUDA else "cpu", dropout=0.0,
        use_moe=True, moe_num_experts=8, moe_top_k=2,
        moe_cache_size=cache_size or 64,
        use_layer_prefetch=kw.pop("use_layer_prefetch", prefetch), seed=0,
        **kw,
    )
    return MetisLM(cfg)


@pytest.fixture(scope="module", autouse=True)
def _force_cpu_prefetch():
    """Allow the prefetcher on CPU for the tests that need it."""
    prev = os.environ.get("METIS_LAYER_PREFETCH_FORCE_CPU")
    os.environ["METIS_LAYER_PREFETCH_FORCE_CPU"] = "1"
    yield
    if prev is None:
        os.environ.pop("METIS_LAYER_PREFETCH_FORCE_CPU", None)
    else:
        os.environ["METIS_LAYER_PREFETCH_FORCE_CPU"] = prev


def _inputs(device: str, bs=2, T=16):
    torch.manual_seed(3)
    x = torch.randint(1, 256, (bs, T), device=device)
    y = torch.randint(1, 256, (bs, T), device=device)
    return x, y


# ── correctness ──────────────────────────────────────────────────────────────

class TestCorrectness:
    def test_eval_forward_bit_identical(self):
        torch.manual_seed(0)
        m_off = _make(False)
        torch.manual_seed(0)
        m_on = _make(True)
        m_on.load_state_dict(m_off.state_dict())
        assert m_on._layer_prefetch is not None and m_off._layer_prefetch is None
        x, y = _inputs(m_off.config.device)
        m_off.eval()
        m_on.eval()
        with torch.no_grad():
            for _ in range(4):  # steps 1+ exercise the prefetch path
                l_off, loss_off, _ = m_off(x, y)
                l_on, loss_on, _ = m_on(x, y)
                assert torch.equal(l_off, l_on)
                assert loss_off == loss_on

    def test_train_forward_and_grads_match(self):
        """Prefetched requires_grad tensors flow through backward identically."""
        torch.manual_seed(0)
        m_off = _make(False)
        torch.manual_seed(0)
        m_on = _make(True)
        m_on.load_state_dict(m_off.state_dict())
        x, y = _inputs(m_off.config.device)
        m_off.train()
        m_on.train()
        for m in (m_off, m_on):
            for _ in range(2):  # warm: populate records + caches
                m(x, y)[0].sum().backward()
                m.zero_grad()
            m.invalidate_moe_caches()
        torch.manual_seed(7)
        l_off, loss_off, _ = m_off(x, y)
        l_off.sum().backward()
        torch.manual_seed(7)
        l_on, loss_on, _ = m_on(x, y)
        l_on.sum().backward()
        g_off = [p.grad.clone() for p in m_off.parameters() if p.grad is not None]
        g_on = [p.grad.clone() for p in m_on.parameters() if p.grad is not None]
        assert loss_off == loss_on
        assert len(g_off) == len(g_on)
        assert max((a - b).abs().max().item() for a, b in zip(g_off, g_on)) == 0.0

    def test_prefetch_does_not_change_output_determinism(self):
        """Two prefetch-ON models with the same seed produce the same output."""
        torch.manual_seed(0)
        m1 = _make(True)
        torch.manual_seed(0)
        m2 = _make(True)
        x, y = _inputs(m1.config.device)
        m1.eval()
        m2.eval()
        with torch.no_grad():
            for _ in range(3):
                l1, _, _ = m1(x, y)
                l2, _, _ = m2(x, y)
                assert torch.equal(l1, l2)


# ── ExpertCache.prefetch API ─────────────────────────────────────────────────

class TestCachePrefetch:
    def test_prefetch_does_not_touch_counters(self):
        c = ExpertCache(64)
        w1 = torch.randn(8, 16)
        w2 = torch.randn(16, 8)
        src = [w1, w2]
        c.prefetch((0,), torch.float32, src,
                   build=lambda: (w1.clone(), w2.clone()))
        assert c.prefetched == 1
        assert c.hits == 0 and c.misses == 0 and c.bytes_built == 0

    def test_prefetched_entry_served_as_hit(self):
        c = ExpertCache(64)
        w1 = torch.randn(8, 16)
        w2 = torch.randn(16, 8)
        src = [w1, w2]
        c.prefetch((0,), torch.float32, src,
                   build=lambda: (w1.clone(), w2.clone()))
        a, b = c.get_or_build((0,), torch.float32, src,
                              build=lambda: (w1.clone(), w2.clone()))
        assert c.hits == 1 and c.misses == 0
        assert c.prefetch_useful == 1
        assert torch.equal(a, w1) and torch.equal(b, w2)

    def test_prefetch_rebuilds_on_staleness(self):
        """A weight mutation between prefetch and use forces a rebuild."""
        c = ExpertCache(64)
        w1 = torch.randn(8, 16)
        w2 = torch.randn(16, 8)
        src = [w1, w2]
        c.prefetch((0,), torch.float32, src,
                   build=lambda: (w1.clone(), w2.clone()))
        w1.add_(1.0)  # in-place mutation bumps _version → stale signature
        a, b = c.get_or_build((0,), torch.float32, [w1, w2],
                              build=lambda: (w1.clone(), w2.clone()))
        assert c.misses == 1  # rebuilt — the stale prefetch was discarded
        assert torch.equal(a, w1)

    def test_prefetch_noop_when_disabled(self):
        c = ExpertCache(0)  # disabled
        w1 = torch.randn(8, 16)
        w2 = torch.randn(16, 8)
        c.prefetch((0,), torch.float32, [w1, w2],
                   build=lambda: (w1, w2))
        assert c.prefetched == 0

    def test_reprefetch_noop_when_fresh(self):
        c = ExpertCache(64)
        w1 = torch.randn(8, 16)
        w2 = torch.randn(16, 8)
        src = [w1, w2]
        c.prefetch((0,), torch.float32, src, build=lambda: (w1, w2))
        c.prefetch((0,), torch.float32, src, build=lambda: (w1, w2))
        assert c.prefetched == 1  # second prefetch found it resident


# ── record + prefetch flow through the model ─────────────────────────────────

class TestPrefetchFlow:
    def test_moe_records_routing(self):
        m = _make(True)
        pf = m._layer_prefetch
        x, y = _inputs(m.config.device)
        m.eval()
        with torch.no_grad():
            m(x, y)
        assert pf is not None
        assert 0 in pf._prev_groups and len(pf._prev_groups) == len(m.layers)
        for groups in pf._prev_groups.values():
            assert all(isinstance(g, tuple) for g, _ in groups)

    def test_prefetch_issues_builds_after_warmup(self):
        m = _make(True)
        pf = m._layer_prefetch
        x, y = _inputs(m.config.device)
        m.eval()
        with torch.no_grad():
            m(x, y)          # step 0: records populate, no prefetch yet
            assert pf.prefetch_calls == 0
            m(x, y)          # step 1: prefetches using step-0 records
        assert pf.prefetch_calls > 0
        assert pf.predictions > 0

    def test_prefetch_raises_hit_rate_under_cold_start(self):
        """With the training loop's per-step invalidation, prefetch warms the
        cache so layers hit instead of re-building every step."""
        torch.manual_seed(0)
        m_off = _make(False, cache_size=16)
        torch.manual_seed(0)
        m_on = _make(True, cache_size=16)
        m_on.load_state_dict(m_off.state_dict())
        x, y = _inputs(m_off.config.device)
        m_off.eval()
        m_on.eval()
        with torch.no_grad():
            for _ in range(4):
                m_off(x, y)
                m_on(x, y)  # populate records
            for _ in range(4):  # measured steps, cold each time
                m_off.invalidate_moe_caches()
                m_off(x, y)
                m_on.invalidate_moe_caches()
                m_on(x, y)
        hr_off = sum(s["hits"] for s in m_off.get_moe_cache_stats() if s)
        hr_on = sum(s["hits"] for s in m_on.get_moe_cache_stats() if s)
        assert hr_on > hr_off  # prefetch converts cold misses into hits


# ── gating / config ──────────────────────────────────────────────────────────

class TestGating:
    def test_prefetcher_cuda_gated_on_cpu(self, monkeypatch):
        if CUDA:
            pytest.skip("CUDA machine — the CUDA gate doesn't apply")
        monkeypatch.delenv("METIS_LAYER_PREFETCH_FORCE_CPU", raising=False)
        m = _make(True)
        assert m._layer_prefetch is None  # CPU, no force env → no-op

    def test_config_flag_disables_prefetch(self):
        torch.manual_seed(0)
        m = _make(True, use_layer_prefetch=False)
        assert m._layer_prefetch is None

    def test_forward_grouped_record_sink(self):
        """forward_grouped calls record() with ascending-sorted group tuples."""
        from metis.moe import forward_grouped
        torch.manual_seed(0)
        N, E, D = 16, 4, 32
        x = torch.randn(N, D)
        w = torch.randn(N, E)
        idx = torch.randint(0, E, (N, 2))
        w1v = [torch.randn(D, 2 * D) for _ in range(E)]
        w2v = [torch.randn(2 * D, D) for _ in range(E)]
        seen = []
        forward_grouped(x, w, idx, w1v, w2v, top_k=2, num_experts=E,
                        record=lambda g, d: seen.append((g, d)))
        assert seen
        groups, dtype = seen[0]
        assert all(g == tuple(sorted(g)) for g in groups)  # ascending keys
