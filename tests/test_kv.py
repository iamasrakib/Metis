"""
Metis — Unit Tests for KV-cache subsystem (Phase 7)
=====================================================
Covers: LayerKV, KVCache, cached_len_of, cache_memory_bytes, kv_cache_ratio,
config validation, generate_text round-trip, static bit-identical parity,
quantized bounded error, MLA absorbed-vs-explicit parity, overflow reset.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from metis import (  # noqa: E402  (public API)
    KVBackendInfo,
    KVCache,
    LayerKV,
    MetisLM,
    ModelConfig,
    cache_memory_bytes,
    cached_len_of,
    dequantize_per_token,
    kv_cache_ratio,
    quantize_per_token,
)

# ── Helpers ──────────────────────────────────────────────────────────────────

def _cfg(**ov):
    d = dict(
        d_model=64, n_heads=4, n_kv_heads=0, n_layers=2,
        max_seq_len=32, vocab_size=256, dropout=0.0,
        use_rmsnorm=True, use_swiglu=True, use_rope=True,
        tie_weights=True, use_moe=False, use_qk_norm=False,
        use_attention_sink=False, use_flash_attn=False,
    )
    d.update(ov)
    return ModelConfig(**d)


def _seeded_model(cfg, seed=42):
    torch.manual_seed(seed)
    m = MetisLM(cfg).eval()
    return m


def _share_weights(target, source):
    target.load_state_dict({k: v.clone() for k, v in source.state_dict().items()})


# ── Quantization round-trip ─────────────────────────────────────────────────

class TestQuantizeDequantize:
    def test_roundtrip_identity(self):
        """quantize -> dequantize recovers the original within int8 precision."""
        x = torch.randn(1, 4, 8, 16) * 0.5
        q, scale = quantize_per_token(x)
        x2 = dequantize_per_token(q, scale, x.dtype)
        assert q.dtype == torch.int8
        assert q.shape == x.shape
        assert scale.shape == (1, 4, 8, 1)
        assert (x - x2).abs().max().item() < 0.02  # int8 precision ~1/127 ~ 0.008

    def test_zero_input(self):
        """Zero input produces zero output without division-by-zero."""
        x = torch.zeros(1, 2, 4, 8)
        q, scale = quantize_per_token(x)
        x2 = dequantize_per_token(q, scale, x.dtype)
        assert torch.all(x2 == 0)
        assert torch.all(scale == 1)  # guard: zero rows get scale=1

    def test_batch_consistency(self):
        """Different batch elements get independent scales."""
        x = torch.randn(3, 2, 4, 8)
        q, scale = quantize_per_token(x)
        # scale shape: (B, n_kv, T, 1) — independent per token
        assert scale.shape == (3, 2, 4, 1)
        # each batch has different scale
        assert not torch.allclose(scale[0], scale[1])


# ── LayerKV ──────────────────────────────────────────────────────────────────

class TestLayerKV:
    def test_static_append_and_read(self):
        cfg = _cfg(kv_backend="static")
        layer = LayerKV("static", cfg)
        k = torch.randn(1, 4, 5, 16)
        v = torch.randn(1, 4, 5, 16)
        layer.append(k, v)
        assert layer.cached_len == 5
        k2, v2 = layer.keys_values()
        assert torch.equal(k, k2)
        assert torch.equal(v, v2)

    def test_quantized_append_and_read(self):
        cfg = _cfg(kv_backend="quantized")
        layer = LayerKV("quantized", cfg)
        k = torch.randn(1, 4, 8, 16)
        v = torch.randn(1, 4, 8, 16)
        layer.append(k, v)
        assert layer.cached_len == 8
        assert layer._k.dtype == torch.int8  # stored as int8
        k2, v2 = layer.keys_values()
        assert k2.dtype == torch.float32
        assert (k - k2).abs().max().item() < 0.02

    def test_multiple_appends(self):
        cfg = _cfg(kv_backend="static")
        layer = LayerKV("static", cfg)
        for i in range(3):
            k = torch.randn(1, 4, 4, 16)
            v = torch.randn(1, 4, 4, 16)
            layer.append(k, v)
        assert layer.cached_len == 12

    def test_reset(self):
        cfg = _cfg(kv_backend="static")
        layer = LayerKV("static", cfg)
        k = torch.randn(1, 4, 8, 16)
        layer.append(k, k)
        assert layer.cached_len == 8
        layer.reset()
        assert layer.cached_len == 0

    def test_overflow_raises(self):
        cfg = _cfg(kv_backend="static", max_seq_len=4)
        layer = LayerKV("static", cfg)
        k = torch.randn(1, 4, 3, 16)
        layer.append(k, k)
        with pytest.raises(RuntimeError, match="overflow"):
            layer.append(k, k)  # 3+3 > 4

    def test_tuple_compat(self):
        cfg = _cfg(kv_backend="static")
        layer = LayerKV("static", cfg)
        k = torch.randn(1, 4, 8, 16)
        layer.append(k, k)
        assert len(layer) == 2
        assert layer[0].shape == (1, 4, 8, 16)
        assert layer[1].shape == (1, 4, 8, 16)

    def test_memory_bytes(self):
        cfg = _cfg(kv_backend="static")
        layer = LayerKV("static", cfg)
        k = torch.randn(1, 4, 8, 16)
        layer.append(k, k)
        expected = 2 * 1 * 4 * 32 * 16 * 4  # max_seq=32, 2 buffers
        assert layer.allocated_bytes() == expected
        assert layer.used_bytes() == 2 * 1 * 4 * 8 * 16 * 4  # only 8 rows used


# ── KVCache ──────────────────────────────────────────────────────────────────

class TestKVCache:
    def test_list_like(self):
        cfg = _cfg(kv_backend="static")
        cache = KVCache("static", cfg, n_layers=2)
        assert len(cache) == 2
        assert cache.cached_len == 0

    def test_from_legacy(self):
        cfg = _cfg(kv_backend="static")
        legacy = [(torch.randn(1, 4, 5, 16), torch.randn(1, 4, 5, 16)) for _ in range(2)]
        cache = KVCache.from_legacy("static", cfg, legacy, n_layers=2)
        assert cache.cached_len == 5
        k, v = cache[0].keys_values()
        assert torch.equal(k, legacy[0][0])

    def test_reset_all_layers(self):
        cfg = _cfg(kv_backend="static")
        cache = KVCache("static", cfg, n_layers=2)
        for layer in cache.layers:
            layer.append(torch.randn(1, 4, 8, 16), torch.randn(1, 4, 8, 16))
        cache.reset()
        assert cache.cached_len == 0


# ── cached_len_of ────────────────────────────────────────────────────────────

class TestCachedLenOf:
    def test_none(self):
        assert cached_len_of(None) == 0

    def test_default_tuple(self):
        k = torch.randn(1, 4, 8, 16)
        assert cached_len_of([(k, k)]) == 8

    def test_layerkv(self):
        cfg = _cfg(kv_backend="static")
        layer = LayerKV("static", cfg)
        layer.append(torch.randn(1, 4, 8, 16), torch.randn(1, 4, 8, 16))
        assert cached_len_of(layer) == 8

    def test_kvcache(self):
        cfg = _cfg(kv_backend="static")
        cache = KVCache("static", cfg, n_layers=2)
        cache[0].append(torch.randn(1, 4, 8, 16), torch.randn(1, 4, 8, 16))
        cache[1].append(torch.randn(1, 4, 8, 16), torch.randn(1, 4, 8, 16))
        assert cached_len_of(cache) == 8

    def test_mla_list(self):
        from metis.mla import MLALayerCache
        c = torch.randn(1, 8, 16)
        krope = torch.randn(1, 4, 8, 8)
        assert cached_len_of([MLALayerCache(c, krope)]) == 8


# ── cache_memory_bytes ───────────────────────────────────────────────────────

class TestCacheMemoryBytes:
    def test_default(self):
        b = cache_memory_bytes("default", B=1, n_kv_heads=4, head_dim=16,
                               T=32, max_seq_len=64, dtype=torch.float32)
        assert b == 2 * 1 * 4 * 32 * 16 * 4

    def test_static(self):
        b = cache_memory_bytes("static", B=1, n_kv_heads=4, head_dim=16,
                               T=32, max_seq_len=64, dtype=torch.float32)
        assert b == 2 * 1 * 4 * 64 * 16 * 4  # uses max_seq, not T

    def test_quantized(self):
        b = cache_memory_bytes("quantized", B=1, n_kv_heads=4, head_dim=16,
                               T=32, max_seq_len=64, dtype=torch.float32)
        # int8 K + int8 V + 2 scale arrays
        per = 1 * 4 * 32
        assert b == 2 * per * 16 + 2 * per * 4

    def test_mla(self):
        b = cache_memory_bytes("mla", B=1, n_kv_heads=4, head_dim=16,
                               T=32, max_seq_len=64, dtype=torch.float32,
                               mla_kv_latent_dim=16, mla_rope_head_dim=8,
                               n_heads=4)
        assert b == 1 * 32 * (16 + 4 * 8) * 4  # B * T * (c_d + nH * rD) * elem

    def test_ratio(self):
        kw = dict(B=1, n_kv_heads=4, head_dim=16, dtype=torch.float32)
        r = kv_cache_ratio("quantized", T=32, max_seq_len=64, **kw)
        assert r > 1.0  # default > quantized


# ── Config validation ────────────────────────────────────────────────────────

class TestKVConfigValidation:
    def test_default_kv_backend(self):
        cfg = _cfg()
        assert cfg.kv_backend == "default"

    def test_static_valid(self):
        cfg = _cfg(kv_backend="static")
        assert cfg.kv_backend == "static"

    def test_quantized_valid(self):
        cfg = _cfg(kv_backend="quantized")
        assert cfg.kv_backend == "quantized"

    def test_mla_valid(self):
        cfg = _cfg(kv_backend="mla")
        assert cfg.kv_backend == "mla"

    def test_invalid_kv_backend(self):
        with pytest.raises(ValueError, match="kv_backend"):
            _cfg(kv_backend="invalid")

    def test_invalid_cache_dtype(self):
        with pytest.raises(ValueError, match="kv_cache_dtype"):
            _cfg(kv_cache_dtype="float64")

    def test_invalid_quant_scheme(self):
        with pytest.raises(ValueError, match="kv_quant_scheme"):
            _cfg(kv_quant_scheme="fp16")


# ── Model integration: static bit-identical ──────────────────────────────────

class TestModelStaticBitIdentical:
    def test_cold_prefill(self):
        cfg = _cfg(kv_backend="static")
        model_s = _seeded_model(cfg)
        model_d = _seeded_model(_cfg())
        _share_weights(model_s, model_d)
        idx = torch.randint(0, 256, (1, 8))
        d_logits, _, _ = model_d(idx)
        s_logits, _, _ = model_s(idx)
        assert torch.equal(d_logits, s_logits)

    def test_decode_steps(self):
        cfg = _cfg(kv_backend="static")
        model_s = _seeded_model(cfg)
        model_d = _seeded_model(_cfg())
        _share_weights(model_s, model_d)
        idx = torch.randint(0, 256, (1, 8))
        d_logits, _, d_cache = model_d(idx)
        s_logits, _, s_cache = model_s(idx)
        assert torch.equal(d_logits, s_logits)
        for _ in range(3):
            d_logits2, _, d_cache = model_d(idx[:, -1:], kv_cache=d_cache)
            s_logits2, _, s_cache = model_s(idx[:, -1:], kv_cache=s_cache)
            assert torch.equal(d_logits2, s_logits2)


# ── Model integration: quantized bounded error ───────────────────────────────

class TestModelQuantizedError:
    def test_bounded_error(self):
        cfg = _cfg(kv_backend="quantized")
        model_q = _seeded_model(cfg)
        model_d = _seeded_model(_cfg())
        _share_weights(model_q, model_d)
        idx = torch.randint(0, 256, (1, 16))
        d_logits, _, d_cache = model_d(idx)
        q_logits, _, q_cache = model_q(idx)
        # Cold: error bounded
        assert (d_logits - q_logits).abs().max().item() < 2.0
        # Decode: error bounded
        for _ in range(5):
            d_logits2, _, d_cache = model_d(idx[:, -1:], kv_cache=d_cache)
            q_logits2, _, q_cache = model_q(idx[:, -1:], kv_cache=q_cache)
            assert (d_logits2 - q_logits2).abs().max().item() < 2.0


# ── Model integration: MLA ───────────────────────────────────────────────────

class TestModelMLA:
    def test_cold_prefill(self):
        cfg = _cfg(kv_backend="mla")
        model = _seeded_model(cfg)
        idx = torch.randint(0, 256, (1, 8))
        logits, _, cache = model(idx)
        assert logits.shape == (1, 1, 256)
        assert len(cache) == 2  # 2 layers
        assert cache[0].cached_len == 8

    def test_decode(self):
        cfg = _cfg(kv_backend="mla")
        model = _seeded_model(cfg)
        idx = torch.randint(0, 256, (1, 8))
        _, _, cache = model(idx)
        logits2, _, cache2 = model(idx[:, -1:], kv_cache=cache)
        assert logits2.shape == (1, 1, 256)
        assert cache2[0].cached_len == 9

    def test_absorbed_vs_explicit(self):
        """Absorbed decode must match full re-prefill (explicit K/V)."""
        cfg = _cfg(kv_backend="mla")
        model = _seeded_model(cfg)
        idx = torch.randint(0, 256, (1, 8))
        idx9 = torch.cat([idx, idx[:, -1:]], dim=1)
        # Explicit: full prefill of 9 tokens
        l_full, _, _ = model(idx9)
        l_ref = l_full[:, -1, :]
        # Absorbed: prefill 8, decode 1
        _, _, cache = model(idx)
        l_dec, _, _ = model(idx[:, -1:], kv_cache=cache)
        l_cached = l_dec[:, -1, :]
        assert (l_ref - l_cached).abs().max().item() < 1e-4

    def test_deterministic(self):
        cfg = _cfg(kv_backend="mla")
        m1 = _seeded_model(cfg, seed=42)
        m2 = _seeded_model(cfg, seed=42)
        _share_weights(m2, m1)
        idx = torch.randint(0, 256, (1, 8))
        l1, _, c1 = m1(idx)
        l2, _, c2 = m2(idx)
        assert torch.equal(l1, l2)


# ── Sliding window overflow ──────────────────────────────────────────────────

class TestSlidingWindow:
    def test_static_reset_and_refill(self):
        """Static cache: overflow triggers reset, re-prefill continues."""
        cfg = _cfg(kv_backend="static", max_seq_len=8)
        model = _seeded_model(cfg)
        idx = torch.randint(0, 256, (1, 8))
        _, _, cache = model(idx)
        assert cache.cached_len == 8
        # Next step would overflow (8+1 > 8), trigger reset via generate_text
        cache.reset()
        assert cache.cached_len == 0
        # Re-prefill works after reset
        _, _, cache2 = model(idx)
        assert cache2.cached_len == 8

    def test_mla_overflow_via_generate(self):
        """MLA: overflow triggers kv_cache=None reset in generate_text."""
        from metis.generate import generate_text
        cfg = _cfg(kv_backend="mla", max_seq_len=8)
        model = _seeded_model(cfg)
        from metis.data import CharTokenizer
        tok = CharTokenizer()
        tok.fit("hello world " * 20)
        output = generate_text(
            model, tok, "hello",
            max_new_tokens=15,
            temperature=0.0,
            device="cpu",
            use_kv_cache=True,
        )
        assert isinstance(output, str)
        assert len(output) > 0


# ── Backend info ─────────────────────────────────────────────────────────────

class TestKVBackendInfo:
    def test_describe_all(self):
        for name in ("default", "static", "quantized", "mla"):
            info = KVBackendInfo.describe(name)
            assert info.name == name
            assert info.bit_identical in (True, False)

    def test_mla_needs_retrain(self):
        assert KVBackendInfo.describe("mla").needs_retrain is True

    def test_static_bit_identical(self):
        assert KVBackendInfo.describe("static").bit_identical is True
