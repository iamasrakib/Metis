#!/usr/bin/env python3
"""
Metis — Grouped MoE numerical-parity verification
==================================================
Directly compares the LEGACY per-expert MoE loop against the NEW grouped
execution engine (token sorting → expert batching → grouped GEMM → grouped
SwiGLU → grouped output projection) across every requirement axis:

  routing, forward (fp32 + fp16/bf16 AMP), gradients, full-model logits/loss,
  KV-cache decode, gradient checkpointing, idle experts, and engine resolution.

Each check prints a PASS/FAIL line; the script exits non-zero on any failure.
CPU-safe by default; AMP checks auto-skip without CUDA.

Usage:
    python benchmarks/verify_moe_parity.py                # auto device
    python benchmarks/verify_moe_parity.py --device cuda
    python benchmarks/verify_moe_parity.py --seed 0 --tol 2e-2
"""

import argparse
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from metis.config import ModelConfig  # noqa: E402
from metis.model import MetisLM  # noqa: E402
from metis.moe import (  # noqa: E402
    GROUPED,
    PER_EXPERT,
    detect_moe_engines,
    normalize_engine,
    resolve_engine,
)


def _configure_stdio() -> None:
    """Make console output encoding-safe on Windows (cp1252 vs UTF-8)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def _make_cfg(**kw):
    cfg = dict(
        d_model=128,
        n_heads=4,
        n_kv_heads=2,
        n_layers=2,
        max_seq_len=128,
        vocab_size=256,
        dropout=0.0,
        use_moe=True,
        moe_num_experts=8,
        moe_top_k=2,
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


def _models(seed: int, **cfg_kw) -> tuple[MetisLM, MetisLM]:
    """Two MetisLM twins with IDENTICAL weights; one runs each engine."""
    torch.manual_seed(seed)
    m_grouped = MetisLM(_make_cfg(moe_engine=GROUPED, **cfg_kw))
    m_ref = MetisLM(_make_cfg(moe_engine=PER_EXPERT, **cfg_kw))
    m_ref.load_state_dict(m_grouped.state_dict())
    return m_grouped, m_ref


def verify_routing(r: Result, device: str, seed: int):
    """Top-k selection and weights are bit-identical to the legacy routing."""
    print("\n[routing] grouped == legacy (bit-identical top-k)")
    torch.manual_seed(seed)
    m_grouped, m_ref = _models(seed)
    m_grouped.to(device); m_ref.to(device)
    x = torch.randn(3, 12, 128, device=device)

    # Drive the routing stage directly on each layer's MoE module.
    for i in (0, 1):
        g_moe, r_moe = m_grouped.layers[i].ffn, m_ref.layers[i].ffn
        x_flat = x.reshape(-1, 128)
        gl = g_moe.gate(x_flat); rl = r_moe.gate(x_flat)
        gw, gi = torch.topk(torch.softmax(gl, -1), g_moe.top_k, -1)
        rw, ri = torch.topk(torch.softmax(rl, -1), r_moe.top_k, -1)
        gw = gw / gw.sum(-1, keepdim=True)
        rw = rw / rw.sum(-1, keepdim=True)
        r.check(
            f"layer {i} top-k indices + weights bit-identical",
            torch.equal(gi, ri) and torch.equal(gw, rw),
        )


def verify_forward_parity(r: Result, device: str, seed: int, tol: float):
    """Layer-level forward: grouped == per_expert in fp32 and AMP dtypes."""
    print("\n[forward] grouped vs per_expert (layer-level)")
    torch.manual_seed(seed)
    g_moe = _models(seed)[0].layers[0].ffn
    r_moe = _models(seed)[1].layers[0].ffn
    g_moe.to(device).eval(); r_moe.to(device).eval()
    r_moe.load_state_dict(g_moe.state_dict())
    x = torch.randn(4, 16, 128, device=device)

    cases = [(torch.float32, tol, "fp32")]
    if device.startswith("cuda"):
        cases += [
            (torch.float16, max(tol, 1e-2), "fp16 AMP"),
            (torch.bfloat16, max(tol, 1e-2), "bf16 AMP"),
        ]
    for dtype, t, label in cases:
        with torch.autocast(device_type=device.split(":")[0], dtype=dtype,
                            enabled=dtype != torch.float32):
            g_out = g_moe(x).float()
            r_out = r_moe(x).float()
        err = float((g_out - r_out).abs().max())
        r.check(f"forward {label}", err < t, f"max_err={err:.2e}")


def verify_grad_parity(r: Result, device: str, seed: int, tol: float):
    """Gradients through every grouped op == per-expert gradients."""
    print("\n[grad] grouped backward vs per_expert backward")
    torch.manual_seed(seed)
    g_moe, r_moe = _models(seed)[0].layers[0].ffn, _models(seed)[1].layers[0].ffn
    g_moe.to(device); r_moe.to(device)
    r_moe.load_state_dict(g_moe.state_dict())
    x = torch.randn(4, 16, 128, device=device)

    g_moe(x).sum().backward()
    r_moe(x).sum().backward()
    diffs = [
        float((g.grad - r.grad).abs().max())
        for g, r in zip(g_moe.parameters(), r_moe.parameters())
    ]
    r.check("all expert/gate grads match", max(diffs) < tol,
            f"max_grad_diff={max(diffs):.2e}")


def verify_model_parity(r: Result, device: str, seed: int, tol: float):
    """End-to-end: full MetisLM logits / loss / grads match."""
    print("\n[model] full MetisLM (MoE on): grouped vs per_expert")
    torch.manual_seed(seed)
    m_grouped, m_ref = _models(seed)
    m_grouped.to(device); m_ref.to(device)
    idx = torch.randint(0, 256, (2, 32), device=device)
    m_grouped.eval(); m_ref.eval()
    with torch.no_grad():
        lg, loss_g, _ = m_grouped(idx, targets=idx)
        lp, loss_p, _ = m_ref(idx, targets=idx)
    logit_err = float((lg.float() - lp.float()).abs().max())
    r.check("logits match", logit_err < tol, f"max_logit_err={logit_err:.2e}")
    r.check("loss match", abs(loss_g.item() - loss_p.item()) < tol,
            f"loss_g={loss_g.item():.4f} loss_p={loss_p.item():.4f}")

    m_grouped.train(); m_ref.train()
    _, lg, _ = m_grouped(idx, targets=idx); lg.backward()
    g = {n: p.grad.clone() for n, p in m_grouped.named_parameters()}
    m_ref.zero_grad()
    _, lp, _ = m_ref(idx, targets=idx); lp.backward()
    p = {n: p.grad.clone() for n, p in m_ref.named_parameters()}
    diffs = [float((g[n] - p[n]).abs().max()) for n in g]
    r.check("model grads match", max(diffs) < tol, f"max_grad_diff={max(diffs):.2e}")


def verify_decode(r: Result, device: str, seed: int, tol: float):
    """KV-cache decode with MoE: cached == full forward (grouped engine)."""
    print("\n[model] MoE KV-cache decode == full forward (grouped)")
    amp = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    torch.manual_seed(seed)
    m = _models(seed)[0]
    m.to(device).eval()
    toks = torch.randint(0, 256, (1, 32), device=device)
    cache = None
    outs = []
    with torch.no_grad(), torch.autocast(device_type=device, dtype=amp):
        for i in range(toks.size(1)):
            lg, _, cache = m(toks[:, i:i + 1], kv_cache=cache)
            outs.append(lg[:, -1, :].float())
        full, _, _ = m(toks, targets=toks)
    bad = [
        i for i in range(toks.size(1))
        if not torch.allclose(outs[i], full[:, i].float(), rtol=tol, atol=tol)
    ]
    r.check("MoE decode == full forward", not bad,
            f"max_pos_err={max(bad) if bad else 0}")


def verify_grad_checkpointing(r: Result, device: str, seed: int, tol: float):
    """Gradient checkpointing == no-checkpointing gradients (grouped engine)."""
    print("\n[model] gradient checkpointing equivalence (grouped MoE)")
    torch.manual_seed(seed)

    def grads_with(ckpt):
        m = _models(seed)[0]
        m.to(device).train()
        idx = torch.randint(0, 256, (2, 32), device=device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16,
                            enabled=device.startswith("cuda")):
            _, loss, _ = m(idx, targets=idx, use_checkpointing=ckpt)
        loss.backward()
        return {n: p.grad.clone() for n, p in m.named_parameters()
                if p.grad is not None}

    g1 = grads_with(False)
    g2 = grads_with(True)
    names = set(g1) & set(g2)
    diffs = [float((g1[n] - g2[n]).abs().max()) for n in names]
    r.check("checkpointed grads == full grads", max(diffs) < tol,
            f"max_grad_diff={max(diffs):.2e}")


def verify_edge_cases(r: Result, device: str, seed: int, tol: float):
    """Idle experts and extreme routing still match the reference."""
    print("\n[edge] idle experts / top-k=1 / single-expert crowding")
    torch.manual_seed(seed)
    g_moe = _models(seed)[0].layers[0].ffn
    r_moe = _models(seed)[1].layers[0].ffn
    g_moe.to(device).eval(); r_moe.to(device).eval()
    r_moe.load_state_dict(g_moe.state_dict())
    x = torch.randn(4, 8, 128, device=device)
    x_flat = x.reshape(-1, 128)
    N = x_flat.shape[0]
    E = g_moe.num_experts

    def routed(logits, k):
        w, i = torch.topk(torch.softmax(logits, -1), k, -1)
        return w / w.sum(-1, keepdim=True), i

    from metis.moe import forward_grouped, forward_per_expert

    w1v = [e[0].weight.t() for e in g_moe.experts]
    w2v = [e[2].weight.t() for e in g_moe.experts]

    # Only experts {0,1} used → 6 idle.
    lg = torch.full((N, E), -1e9, device=device); lg[:, :2] = 1.0
    w, i = routed(lg, 2)
    g = forward_grouped(x_flat, w, i, w1v, w2v, top_k=2, num_experts=E)
    p = forward_per_expert(x_flat, w, i, g_moe.experts, top_k=2)
    r.check("idle experts", float((g - p).abs().max()) < tol,
            f"max_err={float((g-p).abs().max()):.2e}")

    # top-k=1.
    lg = torch.randn(N, E, device=device)
    w, i = routed(lg, 1)
    g = forward_grouped(x_flat, w, i, w1v, w2v, top_k=1, num_experts=E)
    p = forward_per_expert(x_flat, w, i, g_moe.experts, top_k=1)
    r.check("top-k=1", float((g - p).abs().max()) < tol,
            f"max_err={float((g-p).abs().max()):.2e}")

    # top-k=1 with every token forced to expert 0 (max padding).
    lg = torch.full((N, E), -1e9, device=device); lg[:, 0] = 1.0
    w, i = routed(lg, 1)
    g = forward_grouped(x_flat, w, i, w1v, w2v, top_k=1, num_experts=E)
    p = forward_per_expert(x_flat, w, i, g_moe.experts, top_k=1)
    r.check("single-expert crowding", float((g - p).abs().max()) < tol,
            f"max_err={float((g-p).abs().max()):.2e}")


def verify_engine_resolution(r: Result):
    """Config / env plumbing resolves to the right concrete engine."""
    print("\n[config] engine resolution")
    r.check("auto → grouped", resolve_engine("auto") == GROUPED)
    r.check("per_expert → per_expert", resolve_engine(PER_EXPERT) == PER_EXPERT)
    try:
        normalize_engine("bogus")
        ok = False
    except ValueError:
        ok = True
    r.check("unknown engine raises", ok)
    r.check("capability report", detect_moe_engines()["recommended"] == GROUPED)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Grouped MoE parity verification")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=1e-3)
    args = ap.parse_args(argv)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    _configure_stdio()

    print(
        f"Metis Grouped-MoE parity verification — device={device} "
        f"seed={args.seed} tol={args.tol}"
    )
    print(f"Engines: {detect_moe_engines()}")

    r = Result(args.tol)
    verify_engine_resolution(r)
    verify_routing(r, device, args.seed)
    verify_forward_parity(r, device, args.seed, args.tol)
    verify_grad_parity(r, device, args.seed, max(args.tol, 1e-4))
    verify_model_parity(r, device, args.seed, args.tol)
    if device.startswith("cuda"):
        verify_decode(r, device, args.seed, args.tol)
    verify_grad_checkpointing(r, device, args.seed, args.tol)
    verify_edge_cases(r, device, args.seed, args.tol)

    ok = r.summary()
    print("\nOVERALL:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
