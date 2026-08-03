#!/usr/bin/env python3
"""
Metis — FlashAttention numerical-parity verification
=====================================================
Directly compares the OLD attention path (exact manual math) against the NEW
FlashAttention dispatch path across every requirement axis:

  prefill (MHA/GQA), decode (T_q=1 vs cache), KV-cache decode == full forward,
  AMP (fp16 + bf16 autocast), gradient checkpointing, QK-norm, attention sink,
  forced backend fallback, and torch.compile.

Each check prints a PASS/FAIL line; the script exits non-zero on any failure.
CPU-safe by default; the fused checks auto-skip when no fused kernel exists.

Usage:
    python benchmarks/verify_parity.py                 # auto device
    python benchmarks/verify_parity.py --device cuda
    python benchmarks/verify_parity.py --seed 0 --tol 2e-2
"""

import argparse
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from metis.attn import (  # noqa: E402
    FLASH_ATTN,
    MATH,
    SDPA_FLASH,
    SDPA_MEM_EFFICIENT,
    causal_attention,
    detect_attention_backends,
    math_attention,
    set_backend_flags,
)
from metis.config import ModelConfig  # noqa: E402
from metis.model import MetisLM  # noqa: E402


def _configure_stdio() -> None:
    """Make console output encoding-safe on Windows (cp1252 vs UTF-8)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


FUSED = {SDPA_FLASH, SDPA_MEM_EFFICIENT, FLASH_ATTN}


def _repeat_kv(x, groups):
    if groups == 1:
        return x
    B, n_kv, T, D = x.size()
    x = x[:, :, None, :, :].expand(B, n_kv, groups, T, D)
    return x.reshape(B, n_kv * groups, T, D)


def _make_cfg(**kw):
    cfg = dict(
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        n_layers=2,
        max_seq_len=256,
        vocab_size=256,
        dropout=0.0,
        use_flash_attn=True,
    )
    cfg.update(kw)
    return ModelConfig(**cfg)


class Result:
    def __init__(self, tol):
        self.tol = tol
        self.pass_ = 0
        self.fail = 0
        self.skips = 0

    def check(self, name, cond, detail=""):
        if cond:
            self.pass_ += 1
            print(f"  PASS  {name}" + (f"  ({detail})" if detail else ""))
        else:
            self.fail += 1
            print(f"  FAIL  {name}" + (f"  ({detail})" if detail else ""))

    def skip(self, name, why):
        self.skips += 1
        print(f"  SKIP  {name}  ({why})")

    def summary(self):
        print(f"\n  {self.pass_} passed, {self.fail} failed, {self.skips} skipped")
        return self.fail == 0


def verify_kernel_parity(r: Result, device: str, seed: int, tol: float):
    """Raw-kernel parity: dispatched fused == exact math reference."""
    report = detect_attention_backends()
    if not report["fused_available"]:
        r.skip("kernel parity", "no fused kernel")
        return
    torch.manual_seed(seed)
    print("\n[kernel] fused dispatch vs exact math reference")
    for dtype in (torch.float16, torch.bfloat16):
        for label, H, H_kv, T_q, T_k in (
            ("prefill MHA", 8, 8, 128, 128),
            ("prefill GQA", 8, 4, 128, 128),
            ("decode  MHA", 8, 8, 1, 256),
            ("decode  GQA", 8, 4, 1, 256),
        ):
            q = torch.randn(2, H, T_q, 64, device=device, dtype=dtype)
            k = torch.randn(2, H_kv, T_k, 64, device=device, dtype=dtype)
            v = torch.randn(2, H_kv, T_k, 64, device=device, dtype=dtype)
            groups = H // H_kv
            bl = []
            y = causal_attention(q, k, v, n_heads=H, n_kv_heads=H_kv, out_backend=bl)
            ref = math_attention(q, _repeat_kv(k, groups), _repeat_kv(v, groups))
            used = bl[0] if bl else "?"
            ok = torch.allclose(y.float(), ref.float(), rtol=tol, atol=tol)
            r.check(
                f"{label} {dtype} ({used})",
                ok,
                f"max_err={float((y.float() - ref.float()).abs().max()):.2e}",
            )


def verify_kv_cache_decode(r: Result, device: str, seed: int, tol: float):
    """Cached step-by-step decode == full-context forward (model level)."""
    print("\n[model] KV-cache decode == full forward (AMP bf16)")
    amp = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    for use_flash in (False, True):
        cfg = _make_cfg(max_seq_len=128, use_flash_attn=use_flash)
        m = MetisLM(cfg).to(device).eval()
        torch.manual_seed(seed)
        toks = torch.randint(0, cfg.vocab_size, (1, 48), device=device)
        cache = None
        outs = []
        with torch.no_grad(), torch.autocast(device_type=device, dtype=amp):
            for i in range(toks.size(1)):
                lg, _, cache = m(toks[:, i : i + 1], kv_cache=cache)
                outs.append(lg[:, -1, :].float())
            full, _, _ = m(toks, targets=toks)
        used = m.layers[0].attn.last_backend
        bad = [
            i
            for i in range(toks.size(1))
            if not torch.allclose(outs[i], full[:, i].float(), rtol=tol, atol=tol)
        ]
        r.check(
            f"KV-cache decode use_flash={use_flash} ({used})",
            not bad,
            f"max_pos_err={max(bad) if bad else 0}",
        )


def verify_amp_train(r: Result, device: str, seed: int):
    """fp16 + bf16 AMP train steps with fused attention: finite grads & loss."""
    print("\n[model] AMP train step (fp16 + bf16, gradient checkpointing)")
    for amp_dtype in (torch.float16, torch.bfloat16):
        cfg = _make_cfg(max_seq_len=64, use_flash_attn=True)
        m = MetisLM(cfg).to(device).train()
        opt = m.configure_optimizers(0.1, 1e-3, device)
        scaler = torch.amp.GradScaler(device, enabled=True)
        torch.manual_seed(seed)
        idx = torch.randint(0, cfg.vocab_size, (2, 32), device=device)
        losses = []
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=amp_dtype):
                _, loss, _ = m(idx, targets=idx, use_checkpointing=True)
            losses.append(loss.item())
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        used = m.layers[0].attn.last_backend
        finite = all(math.isfinite(loss_val) for loss_val in losses)
        grads_ok = all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in m.parameters())
        r.check(
            f"AMP {amp_dtype} fused train ({used})", finite and grads_ok, f"loss={losses[-1]:.4f}"
        )


def verify_grad_checkpointing(r: Result, device: str, seed: int):
    """Gradient checkpointing == no-checkpointing gradients (fused path).

    The same seed is re-applied before *each* model construction so both runs
    compare identical weights — without this, the comparison would be of two
    differently-initialized models, not of two training strategies.
    """
    print("\n[model] gradient checkpointing equivalence (fused)")
    cfg = _make_cfg(max_seq_len=64, use_flash_attn=True)

    def grads_with(ckpt):
        torch.manual_seed(seed)  # identical weights for both strategies
        m = MetisLM(cfg).to(device).train()
        idx = torch.randint(0, cfg.vocab_size, (2, 32), device=device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            _, loss, _ = m(idx, targets=idx, use_checkpointing=ckpt)
        loss.backward()
        return loss.item(), {
            n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None
        }

    l1, g1 = grads_with(False)
    l2, g2 = grads_with(True)
    names = set(g1) & set(g2)
    ok = all(torch.allclose(g1[n].float(), g2[n].float(), atol=1e-2) for n in names)
    r.check(
        "checkpointed grads == full grads",
        ok,
        f"max_grad_diff={max(float((g1[n] - g2[n]).float().abs().max()) for n in names):.2e}",
    )


def verify_forced_backends(r: Result, device: str, seed: int, tol: float):
    """Forcing math vs forced fused on the model agree (same weights)."""
    print("\n[model] forced-math vs forced-fused agreement (end-to-end)")
    report = detect_attention_backends()
    fused = report["recommended"]
    if fused == MATH:
        r.skip("forced backends", "no fused kernel")
        return
    # Map the concrete recommended kernel back to a user-facing config value.
    user_facing = {
        SDPA_FLASH: "flash",
        SDPA_MEM_EFFICIENT: "mem_efficient",
        FLASH_ATTN: "flash_attn",
        MATH: "math",
    }[fused]
    outs = {}
    for backend in ("math", user_facing):
        # Re-seed before each model so both backends compare IDENTICAL weights;
        # otherwise the diff would be between two random initializations.
        torch.manual_seed(seed)
        cfg = _make_cfg(max_seq_len=64, attn_backend=backend)
        m = MetisLM(cfg).to(device).eval()
        idx = torch.randint(0, 256, (1, 32), device=device)
        with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.bfloat16):
            lg, _, _ = m(idx, targets=idx)
        outs[backend] = lg.float()
    err = float((outs["math"] - outs[user_facing]).abs().max())
    r.check(f"forced {user_facing} == forced math", err < tol, f"max_logit_err={err:.2e}")


def verify_qk_norm_sink_moe(r: Result, device: str, seed: int):
    """Fused path with QK-norm, attention sink, and MoE all run correctly."""
    print("\n[model] QK-norm + attention sink + MoE with fused attention")
    cfg = _make_cfg(
        max_seq_len=64,
        use_flash_attn=True,
        use_qk_norm=True,
        use_attention_sink=True,
        use_moe=True,
        moe_num_experts=4,
        moe_top_k=2,
    )
    m = MetisLM(cfg).to(device).train()
    torch.manual_seed(seed)
    idx = torch.randint(0, cfg.vocab_size, (2, 16), device=device)
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss, _ = m(idx, targets=idx)
    loss.backward()
    used = m.layers[0].attn.last_backend
    ok = logits.shape == (2, 16, cfg.vocab_size) and math.isfinite(loss.item())
    r.check(f"QK-norm+sink+MoE fused ({used})", ok, f"loss={loss.item():.4f}")


def verify_compile(r: Result, device: str, seed: int, tol: float):
    """torch.compile produces matching outputs on the fused path.

    Only runnable where Triton is installed (e.g. Linux CUDA wheels); Windows
    torch builds do not ship Triton, so the check is skipped there rather than
    reported as a failure — that absence is a torch-platform limitation
    orthogonal to the attention integration.
    """
    if device.startswith("cpu") or not hasattr(torch, "compile"):
        r.skip("torch.compile", "no CUDA / no torch.compile")
        return
    try:
        import triton  # noqa: F401
    except ImportError:
        r.skip("torch.compile", "triton not installed (Windows torch wheel)")
        return
    print("\n[model] torch.compile fused-path equivalence")
    torch.manual_seed(seed)
    idx = torch.randint(0, 256, (1, 32), device=device)
    cfg = _make_cfg(max_seq_len=64, use_flash_attn=True)
    m = MetisLM(cfg).to(device).eval()
    mc = torch.compile(MetisLM(cfg).to(device).eval())
    with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.bfloat16):
        a, _, _ = m(idx, targets=idx)
        b, _, _ = mc(idx, targets=idx)
    err = float((a.float() - b.float()).abs().max())
    r.check("compile output ≈ eager", err < 1e-1, f"max_err={err:.2e}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="FlashAttention parity verification")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=2e-2)
    args = ap.parse_args(argv)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda"):
        set_backend_flags("auto")
    _configure_stdio()

    print(
        f"Metis FlashAttention parity verification — device={device} "
        f"seed={args.seed} tol={args.tol}"
    )
    cap = detect_attention_backends()
    print(f"Backends: recommended={cap['recommended']} fused_available={cap['fused_available']}")

    r = Result(args.tol)
    verify_kernel_parity(r, device, args.seed, args.tol)
    verify_kv_cache_decode(r, device, args.seed, args.tol)
    if device.startswith("cuda"):
        verify_amp_train(r, device, args.seed)
    verify_grad_checkpointing(r, device, args.seed)
    verify_forced_backends(r, device, args.seed, args.tol)
    if device.startswith("cuda"):
        verify_qk_norm_sink_moe(r, device, args.seed)
        verify_compile(r, device, args.seed, args.tol)

    ok = r.summary()
    print("\nOVERALL:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
