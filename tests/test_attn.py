"""
Μῆτις (Metis) — Tests for the FlashAttention dispatch layer
=============================================================
Covers backend detection, automatic fallback, numerical equivalence against
the exact manual math reference, KV-cache decode consistency, AMP, gradient
checkpointing, attention sink, and GQA — for both the fused path and the
manual (use_flash_attn=False) reference path.

CPU-safe by default; CUDA-only cases are marked with ``@gpu``.
"""

import math
import os
import sys
import warnings

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from metis.attn import (
    AUTO,
    FLASH_ATTN,
    MATH,
    SDPA_FLASH,
    SDPA_MEM_EFFICIENT,
    SDPA_MATH,
    causal_attention,
    detect_attention_backends,
    fused_attention_supported,
    math_attention,
    normalize_backend,
)
from metis.model import MetisLM
from tests.test_model import make_config

CUDA = torch.cuda.is_available()
gpu = pytest.mark.skipif(not CUDA, reason="CUDA required")


FUSED = {FLASH_ATTN, SDPA_FLASH, SDPA_MEM_EFFICIENT}


def bf16_fused_ok() -> bool:
    """True if some fused kernel accepts bf16 on this build/GPU.

    On Turing (e.g. Colab T4) torch's fused SDPA kernels are fp16-only, so
    bf16 calls legitimately degrade to the SDPA math kernel instead.
    """
    r = detect_attention_backends()
    return (
        r["flash_attn"] is not None
        or r["torch_flash_bf16"]
        or r["torch_mem_efficient_bf16"]
    )


def allowed_backends(dtype: torch.dtype) -> set:
    """Backends a call of ``dtype`` may legitimately land on (auto dispatch).

    On machines with no fused kernel at all every dtype falls back to the
    SDPA math kernel; on machines whose fused kernels are fp16-only, bf16
    falls back to it too.
    """
    allowed = set(FUSED)
    report = detect_attention_backends()
    if not report["fused_available"]:
        allowed.add(SDPA_MATH)
    elif dtype == torch.bfloat16 and not bf16_fused_ok():
        allowed.add(SDPA_MATH)
    return allowed


# ── Reference: exact legacy manual attention ─────────────────────────────────

def legacy_manual(q, k, v, dropout_p=0.0, is_causal=True, scale=None):
    """Byte-identical copy of the original CausalSelfAttention manual path
    (mask built from the sliced tril(max_seq_len) buffer)."""
    if scale is None:
        scale = 1.0 / math.sqrt(q.size(-1))
    att = (q @ k.transpose(-2, -1)) * scale
    T_q, T_k = q.size(2), k.size(2)
    maxlen = max(T_q, T_k)
    buf = torch.tril(torch.ones(maxlen, maxlen, device=q.device))
    mask = buf[T_k - T_q: T_k, :T_k].view(1, 1, T_q, T_k)
    att = att.masked_fill(mask == 0, float("-inf"))
    att = F.softmax(att, dim=-1)
    att = F.dropout(att, p=dropout_p, training=dropout_p > 0)
    return att @ v


def _repeat_kv(x, n_groups):
    if n_groups == 1:
        return x
    B, n_kv, T, D = x.size()
    x = x[:, :, None, :, :].expand(B, n_kv, n_groups, T, D)
    return x.reshape(B, n_kv * n_groups, T, D)


# ── Detection & normalization ────────────────────────────────────────────────

class TestDetection:
    def test_detect_backends_cpu(self):
        if CUDA:
            pytest.skip("CPU-only assertion")
        report = detect_attention_backends()
        assert report["torch_math"] is True
        assert report["torch_flash"] is False
        assert report["torch_mem_efficient"] is False
        assert report["device"] == "cpu"
        assert report["recommended"] == MATH
        assert report["fused_available"] is False
        assert report["gpu_name"] is None
        assert report["compute_capability"] is None

    def test_detect_backends_shape(self):
        report = detect_attention_backends()
        for key in ("device", "torch", "flash_attn", "flash_attn_gqa",
                    "torch_flash", "torch_mem_efficient", "torch_math",
                    "torch_flash_bf16", "torch_mem_efficient_bf16",
                    "fused_gqa", "recommended", "gpu_name",
                    "compute_capability", "fused_available"):
            assert key in report
        assert report["recommended"] in (
            FLASH_ATTN, SDPA_FLASH, SDPA_MEM_EFFICIENT, SDPA_MATH, MATH
        )
        # fused_available must be consistent with the auto recommendation:
        # "recommended" is a fused kernel iff one is available (the SDPA math
        # fallback is not fused).
        assert report["fused_available"] == (report["recommended"] in FUSED)
        if CUDA:
            # GPU capability detection: device name + compute capability tuple
            assert isinstance(report["gpu_name"], str) and report["gpu_name"]
            assert isinstance(report["compute_capability"], tuple)
            assert len(report["compute_capability"]) == 2
            assert all(isinstance(v, int) for v in report["compute_capability"])
        else:
            assert report["gpu_name"] is None
            assert report["compute_capability"] is None

    def test_fused_attention_supported(self):
        """Public helper agrees with the machine capability snapshot."""
        assert fused_attention_supported() is detect_attention_backends()["fused_available"]

    def test_normalize_backend(self):
        assert normalize_backend("auto") == AUTO
        assert normalize_backend("flash") == SDPA_FLASH
        assert normalize_backend("mem_efficient") == SDPA_MEM_EFFICIENT
        assert normalize_backend("math") == MATH
        assert normalize_backend(None) == AUTO
        with pytest.raises(ValueError):
            normalize_backend("bogus")

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("METIS_ATTN_BACKEND", "math")
        q = torch.randn(1, 4, 8, 16)
        from metis.attn import resolve_backend
        assert resolve_backend("auto", q) == MATH


# ── math_attention is the exact legacy reference ─────────────────────────────

class TestMathAttentionReference:
    def test_prefill_mha_bit_identical(self):
        torch.manual_seed(0)
        q = torch.randn(2, 4, 16, 16)
        k = torch.randn(2, 4, 16, 16)
        v = torch.randn(2, 4, 16, 16)
        y = math_attention(q, k, v, is_causal=True)
        ref = legacy_manual(q, k, v, is_causal=True)
        assert torch.equal(y, ref)  # bit-identical

    def test_decode_tq1_matches_legacy_allkeys(self):
        """T_q=1 decode attends to ALL cached keys (the legacy buffer slice),
        not just the first key (torch SDPA's is_causal tril mis-masks)."""
        torch.manual_seed(0)
        q = torch.randn(1, 4, 1, 16)
        k = torch.randn(1, 4, 8, 16)
        v = torch.randn(1, 4, 8, 16)
        y = math_attention(q, k, v, is_causal=True)
        ref = legacy_manual(q, k, v, is_causal=True)
        assert torch.equal(y, ref)

        # Sanity: attending to all keys is NOT the same as tril(1, T_k)
        tril_mask = torch.tril(torch.ones(1, 8, device=q.device)).view(1, 1, 1, 8)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(16)
        att_tril = att.masked_fill(tril_mask == 0, float("-inf"))
        y_tril = F.softmax(att_tril, dim=-1) @ v
        assert not torch.allclose(y, y_tril, atol=1e-3)

    def test_gqa_expansion_correct(self):
        """Dispatcher GQA (unexpanded KV) == math_attention on expanded KV."""
        torch.manual_seed(0)
        q = torch.randn(2, 4, 16, 16)
        k = torch.randn(2, 2, 16, 16)
        v = torch.randn(2, 2, 16, 16)
        y = causal_attention(q, k, v, n_heads=4, n_kv_heads=2, backend=MATH)
        ref = math_attention(q, _repeat_kv(k, 2), _repeat_kv(v, 2))
        assert torch.equal(y, ref)
        # MQA (n_kv_heads == 1) also broadcasts correctly.
        k1 = torch.randn(2, 1, 16, 16)
        v1 = torch.randn(2, 1, 16, 16)
        y1 = causal_attention(q, k1, v1, n_heads=4, n_kv_heads=1, backend=MATH)
        ref1 = math_attention(q, _repeat_kv(k1, 4), _repeat_kv(v1, 4))
        assert torch.equal(y1, ref1)


# ── Dispatcher: math path & CPU fallback ─────────────────────────────────────

class TestDispatcherCPU:
    def test_manual_matches_math_attention(self):
        torch.manual_seed(0)
        q = torch.randn(2, 4, 16, 16)
        k = torch.randn(2, 4, 16, 16)
        v = torch.randn(2, 4, 16, 16)
        bl = []
        y = causal_attention(q, k, v, n_heads=4, n_kv_heads=4,
                             backend=MATH, out_backend=bl)
        assert bl[0] == MATH
        assert torch.equal(y, math_attention(q, k, v))

    def test_cpu_fallback_to_math(self):
        torch.manual_seed(0)
        q = torch.randn(2, 4, 16, 16)
        k = torch.randn(2, 4, 16, 16)
        v = torch.randn(2, 4, 16, 16)
        bl = []
        y = causal_attention(q, k, v, n_heads=4, n_kv_heads=4,
                             use_flash_attn=True, out_backend=bl)
        assert bl[0] == MATH  # CPU has no fused kernels
        assert torch.allclose(y, math_attention(q, k, v), atol=1e-5)

    def test_cpu_prefill_and_decode_shapes(self):
        torch.manual_seed(0)
        for kv in (4, 2):
            q = torch.randn(2, 4, 32, 16)
            k = torch.randn(2, kv, 32, 16)
            v = torch.randn(2, kv, 32, 16)
            bl = []
            y = causal_attention(q, k, v, n_heads=4, n_kv_heads=kv,
                                 out_backend=bl)
            assert y.shape == q.shape
        q1 = torch.randn(2, 4, 1, 16)
        k8 = torch.randn(2, 4, 8, 16)
        v8 = torch.randn(2, 4, 8, 16)
        y = causal_attention(q1, k8, v8, n_heads=4, n_kv_heads=4)
        assert y.shape == (2, 4, 1, 16)

    def test_model_kv_cache_decode_consistency_cpu(self):
        """Cached step-by-step decode == full-context forward (manual path)."""
        for use_flash in (False, True):
            cfg = make_config(d_model=64, n_heads=4, n_kv_heads=2, n_layers=2,
                              max_seq_len=64, dropout=0.0,
                              use_flash_attn=use_flash)
            m = MetisLM(cfg)
            m.eval()
            torch.manual_seed(1)
            toks = torch.randint(0, cfg.vocab_size, (1, 32))
            cache = None
            outs = []
            with torch.no_grad():
                for i in range(toks.size(1)):
                    lg, _, cache = m(toks[:, i:i + 1], kv_cache=cache)
                    outs.append(lg[:, -1, :])
                full, _, _ = m(toks, targets=toks)
            for i in range(toks.size(1)):
                assert torch.allclose(outs[i], full[:, i], atol=1e-4), (
                    f"use_flash_attn={use_flash} position {i}"
                )

    def test_gradient_checkpointing_works(self):
        cfg = make_config(use_flash_attn=True)
        m = MetisLM(cfg)
        m.train()
        idx = torch.randint(0, cfg.vocab_size, (1, 16))
        _, loss, _ = m(idx, targets=idx, use_checkpointing=True)
        loss.backward()
        assert any(p.grad is not None for p in m.parameters())


# ── GPU: fused-kernel equivalence (mem-efficient / flash) ────────────────────

@gpu
class TestFusedEquivalence:
    def test_prefill_equivalence(self):
        """Fused prefill == math reference within fp16/bf16 tolerance."""
        report = detect_attention_backends()
        if not report["fused_available"]:
            pytest.skip("no fused kernel available on this build")
        torch.manual_seed(0)
        B, H, T, D = 2, 4, 128, 64
        for dt in (torch.float16, torch.bfloat16):
            for kv in (4, 2):
                q = torch.randn(B, H, T, D, device="cuda", dtype=dt)
                k = torch.randn(B, kv, T, D, device="cuda", dtype=dt)
                v = torch.randn(B, kv, T, D, device="cuda", dtype=dt)
                bl = []
                with warnings.catch_warnings(record=True) as record:
                    warnings.simplefilter("always")
                    y = causal_attention(q, k, v, n_heads=H, n_kv_heads=kv,
                                         out_backend=bl)
                # No fallback warnings in the plain auto path (degradation to
                # the SDPA math kernel is a silent selection, not a fallback)
                assert not [w for w in record if issubclass(w.category, UserWarning)]
                assert bl[0] in allowed_backends(dt)
                ref = math_attention(q, _repeat_kv(k, H // kv),
                                     _repeat_kv(v, H // kv))
                # fp16/bf16 fused kernels accumulate in fp32; residual
                # differences are rounding-order noise, bounded empirically.
                assert torch.allclose(y.float(), ref.float(),
                                      rtol=2e-2, atol=2e-2), (
                    f"dtype={dt} kv={kv} backend={bl[0]}"
                )

    def test_bf16_mem_efficient_no_kernel_fixed(self):
        """Regression: bf16 auto dispatch must never raise "No available kernel".

        On Turing (Colab T4) the fused kernels are fp16-only, so a bf16 call
        that resolved to ``SDPA_MEM_EFFICIENT`` used to be pinned to a kernel
        that rejected bf16 — leaving no kernel and crashing. It must now
        degrade to the SDPA math kernel (or MATH) and stay numerically correct.
        This mirrors the failing packed-training path (block-diagonal mask).
        """
        report = detect_attention_backends()
        if report["device"] != "cuda":
            pytest.skip("CUDA required")
        if not report["fused_available"]:
            pytest.skip("no fused kernel available")
        torch.manual_seed(0)
        B, H, T, D, n_kv = 1, 4, 64, 48, 1  # head_dim 48 like the 100M config
        q = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, n_kv, T, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, n_kv, T, D, device="cuda", dtype=torch.bfloat16)
        seg = torch.tril(torch.ones(T // 2, T // 2))
        mask = torch.block_diag(seg, seg).bool() \
            .unsqueeze(0).unsqueeze(0).cuda()  # (1, 1, T, T) packed causal
        bl = []
        y = causal_attention(
            q, k, v, n_heads=H, n_kv_heads=n_kv,
            attention_mask=mask, out_backend=bl,
        )
        assert y.shape == q.shape
        assert bl[0] in allowed_backends(torch.bfloat16) | {MATH}, bl[0]
        ref = math_attention(
            q, _repeat_kv(k, H // n_kv), _repeat_kv(v, H // n_kv),
            attention_mask=mask,
        )
        assert torch.allclose(y.float(), ref.float(), rtol=2e-2, atol=2e-2)

    def test_decode_equivalence(self):
        report = detect_attention_backends()
        if not report["fused_available"]:
            pytest.skip("no fused kernel available")
        torch.manual_seed(0)
        q = torch.randn(2, 4, 1, 64, device="cuda", dtype=torch.float16)
        k = torch.randn(2, 2, 256, 64, device="cuda", dtype=torch.float16)
        v = torch.randn(2, 2, 256, 64, device="cuda", dtype=torch.float16)
        bl = []
        y = causal_attention(q, k, v, n_heads=4, n_kv_heads=2, out_backend=bl)
        ref = math_attention(q, _repeat_kv(k, 2), _repeat_kv(v, 2))
        assert torch.allclose(y.float(), ref.float(), rtol=2e-2, atol=2e-2)

    def test_kv_cache_decode_consistency_cuda(self):
        """Cached step-by-step decode == full-context under AMP autocast."""
        for use_flash in (False, True):
            cfg = make_config(d_model=64, n_heads=4, n_kv_heads=2, n_layers=2,
                              max_seq_len=128, dropout=0.0,
                              use_flash_attn=use_flash)
            m = MetisLM(cfg).cuda()
            m.eval()
            torch.manual_seed(2)
            toks = torch.randint(0, cfg.vocab_size, (1, 48), device="cuda")
            cache = None
            outs = []
            with torch.no_grad(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                for i in range(toks.size(1)):
                    lg, _, cache = m(toks[:, i:i + 1], kv_cache=cache)
                    outs.append(lg[:, -1, :].float())
                full, _, _ = m(toks, targets=toks)
            assert m.layers[0].attn.last_backend is not None
            for i in range(toks.size(1)):
                assert torch.allclose(outs[i], full[:, i].float(),
                                      rtol=2e-2, atol=2e-2), (
                    f"use_flash_attn={use_flash} position {i}"
                )

    def test_amp_train_step_fused(self):
        """fp16 GradScaler train step with fused attention: finite grads."""
        cfg = make_config(d_model=64, n_heads=4, n_kv_heads=2, n_layers=2,
                          max_seq_len=64, dropout=0.0, use_flash_attn=True)
        m = MetisLM(cfg).cuda()
        m.train()
        opt = m.configure_optimizers(0.1, 1e-3, "cuda")
        scaler = torch.amp.GradScaler("cuda")
        idx = torch.randint(0, cfg.vocab_size, (2, 32), device="cuda")
        for _ in range(2):
            opt.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _, loss, _ = m(idx, targets=idx, use_checkpointing=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        assert m.layers[0].attn.last_backend in allowed_backends(torch.float16)
        for p in m.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), "non-finite gradient"

    def test_attention_sink_fused(self):
        cfg = make_config(d_model=64, n_heads=4, n_kv_heads=2, n_layers=1,
                          max_seq_len=64, dropout=0.0, use_flash_attn=True,
                          use_attention_sink=True)
        m = MetisLM(cfg).cuda()
        m.train()
        idx = torch.randint(0, cfg.vocab_size, (2, 16), device="cuda")
        logits, loss, _ = m(idx, targets=idx)
        assert logits.shape == (2, 16, cfg.vocab_size)
        loss.backward()
        m.eval()
        with torch.no_grad(), torch.autocast(device_type="cuda",
                                             dtype=torch.bfloat16):
            lg, _, cache = m(idx[:, :1], kv_cache=None)
        assert lg.shape == (2, 1, cfg.vocab_size)
        assert m.layers[0].attn.last_backend is not None

    def test_moe_with_fused(self):
        cfg = make_config(d_model=64, n_heads=4, n_kv_heads=2, n_layers=1,
                          max_seq_len=64, dropout=0.0, use_flash_attn=True,
                          use_moe=True, moe_num_experts=4, moe_top_k=2)
        m = MetisLM(cfg).cuda()
        m.train()
        idx = torch.randint(0, cfg.vocab_size, (2, 16), device="cuda")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss, _ = m(idx, targets=idx)
        loss.backward()
        assert m.layers[0].attn.last_backend in allowed_backends(torch.bfloat16)

    def test_forced_backend_fallback(self):
        """Requesting an unavailable kernel must degrade, not crash."""
        import metis.attn as attn_mod

        attn_mod._warned.clear()  # ensure the warn-once guard fires here
        torch.manual_seed(0)
        q = torch.randn(2, 4, 32, 64, device="cuda", dtype=torch.float16)
        k = torch.randn(2, 2, 32, 64, device="cuda", dtype=torch.float16)
        v = torch.randn(2, 2, 32, 64, device="cuda", dtype=torch.float16)
        warned = []
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            for forced in ("flash", "flash_attn"):
                bl = []
                y = causal_attention(q, k, v, n_heads=4, n_kv_heads=2,
                                     backend=forced, out_backend=bl)
                assert y.shape == q.shape
                assert bl[0] in (SDPA_FLASH, SDPA_MEM_EFFICIENT, SDPA_MATH, MATH)
            warned = [w for w in record
                      if "not available" in str(w.message)]
        # This build lacks the flash kernel, so at least one fallback warning
        # must have been emitted; on a flash-capable build "flash_attn" still
        # warns because the package is absent.
        assert warned, "expected a fallback warning for an unavailable backend"

    def test_qk_norm_uses_fused_backend(self):
        """Regression: use_qk_norm must not force the fp32 math fallback.

        QKNorm's RMSNorm ended with ``bf16 x fp32 weight``, which AMP autocast
        promotes to fp32, so the normalized q/k reached the dispatcher as fp32
        and were excluded from the fused fp16/bf16 kernels. Under AMP the
        QK-norm path must engage the same fused kernel as the plain path.
        """
        for use_qk_norm in (False, True):
            cfg = make_config(d_model=64, n_heads=4, n_kv_heads=2, n_layers=1,
                              max_seq_len=64, dropout=0.0, use_flash_attn=True,
                              use_qk_norm=use_qk_norm)
            m = MetisLM(cfg).cuda().train()
            torch.manual_seed(0)
            idx = torch.randint(0, cfg.vocab_size, (2, 16), device="cuda")
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss, _ = m(idx, targets=idx)
            assert m.layers[0].attn.last_backend in allowed_backends(torch.bfloat16), (
                f"use_qk_norm={use_qk_norm} should engage a fast kernel (fused "
                f"where bf16 fused is available, else the SDPA math fallback), "
                f"got {m.layers[0].attn.last_backend}"
            )

    def test_kv_cache_decode_uses_fused_backend(self):
        """Regression: KV-cache decode must run a fused kernel, not math.

        The decode RoPE branch used to cast q/k back to the layer-input dtype
        (fp32 embedding output under AMP), which silently excluded them from
        the fused fp16/bf16 kernels and forced a slow math fallback in
        inference. Prefill was unaffected (apply_rope preserves the projected
        dtype), so only cached decode regressed.
        """
        if not detect_attention_backends()["fused_available"]:
            pytest.skip("no fused kernel available")
        fused = allowed_backends(torch.bfloat16)
        cfg = make_config(d_model=64, n_heads=4, n_kv_heads=2, n_layers=2,
                          max_seq_len=128, dropout=0.0, use_flash_attn=True)
        m = MetisLM(cfg).cuda().eval()
        torch.manual_seed(3)
        toks = torch.randint(0, cfg.vocab_size, (1, 48), device="cuda")

        with torch.no_grad(), torch.autocast(device_type="cuda",
                                             dtype=torch.bfloat16):
            # Prefill
            _, _, cache = m(toks, targets=toks)
            assert m.layers[0].attn.last_backend in fused, (
                f"prefill should use a fused kernel (or the SDPA math fallback "
                f"where bf16 fused is unsupported), got "
                f"{m.layers[0].attn.last_backend}"
            )
            # Incremental decode — the regression was here
            for _ in range(3):
                _, _, cache = m(toks[:, -1:], kv_cache=cache)
            assert m.layers[0].attn.last_backend in fused, (
                f"decode should use a fused kernel (or the SDPA math fallback "
                f"where bf16 fused is unsupported), got "
                f"{m.layers[0].attn.last_backend}"
            )

        # And cached step-by-step decode still matches the full forward
        with torch.no_grad(), torch.autocast(device_type="cuda",
                                             dtype=torch.bfloat16):
            full, _, _ = m(toks, targets=toks)
        cache2 = None
        outs = []
        with torch.no_grad(), torch.autocast(device_type="cuda",
                                             dtype=torch.bfloat16):
            for i in range(toks.size(1)):
                lg, _, cache2 = m(toks[:, i:i + 1], kv_cache=cache2)
                outs.append(lg[:, -1, :].float())
        for i in range(toks.size(1)):
            assert torch.allclose(outs[i], full[:, i].float(),
                                  rtol=2e-2, atol=2e-2), (
                f"use_flash_attn=True position {i}"
            )


# ── End-to-end: forced math vs fused agree ───────────────────────────────────

class TestModelBackends:
    def test_forced_math_reports_math(self):
        cfg = make_config(use_flash_attn=True, attn_backend="math")
        m = MetisLM(cfg)
        m.eval()
        idx = torch.randint(0, cfg.vocab_size, (1, 16))
        with torch.no_grad():
            m(idx)
        assert m.layers[0].attn.last_backend == MATH

    def test_invalid_backend_rejected(self):
        with pytest.raises(ValueError):
            make_config(attn_backend="bogus")
