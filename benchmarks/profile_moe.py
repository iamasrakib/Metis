#!/usr/bin/env python3
"""
Metis — MoE kernel-profile capture
==================================
Profiles one full MetisLM train step (forward + backward, bf16 AMP) for each
MoE engine and dumps a per-op CUDA-time breakdown to
``benchmarks/results/profile_moe_<timestamp>.json``.

On this torch build the profiler attributes CUDA time to aten ops rather than
individual kernels, so the breakdown is at op level — enough to show exactly
where each engine spends its device time (per_expert: boolean-mask scatters;
grouped: grouped bmms).

Usage:
    python benchmarks/profile_moe.py [--device cuda] [--iters 3]
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from metis.config import ModelConfig  # noqa: E402
from metis.model import MetisLM  # noqa: E402
from metis.moe import GROUPED, PER_EXPERT  # noqa: E402


def _build(engine: str) -> MetisLM:
    cfg = ModelConfig(
        d_model=128, n_heads=4, n_kv_heads=2, n_layers=4, max_seq_len=256,
        vocab_size=256, dropout=0.0, use_moe=True, moe_num_experts=8,
        moe_top_k=2, moe_engine=engine,
    )
    return MetisLM(cfg).cuda().train()


def _profile_step(engine: str, iters: int) -> dict:
    m = _build(engine)
    idx = torch.randint(0, 256, (2, 128), device="cuda")
    amp = torch.bfloat16

    def step():
        m.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=amp):
            _, loss, _ = m(idx, targets=idx)
        loss.backward()

    for _ in range(iters):
        step()
    torch.cuda.synchronize()

    with torch.autograd.profiler.profile(use_device="cuda") as prof:
        step()
    torch.cuda.synchronize()

    table = prof.key_averages()
    ops = []
    for e in table:
        cuda_ms = e.self_device_time_total / 1000.0
        calls = e.count
        ops.append({
            "op": e.key,
            "self_cuda_ms": round(cuda_ms, 3),
            "calls": calls,
            "avg_cuda_ms": round(cuda_ms / max(calls, 1), 4),
        })
    ops.sort(key=lambda o: -o["self_cuda_ms"])
    # Aggregate the scatter family to quantify the per-expert overhead.
    scatter = ["index_put", "index_put_", "index", "nonzero", "IndexBackward"]
    agg = {
        "scatter_family_ms": round(sum(o["self_cuda_ms"] for o in ops
                                       if any(s in o["op"] for s in scatter)), 3),
        "bmm_ms": round(sum(o["self_cuda_ms"] for o in ops if "bmm" in o["op"]), 3),
        "total_self_cuda_ms": round(sum(o["self_cuda_ms"] for o in ops), 3),
    }
    return {"top_ops": ops[:12], "aggregates": agg}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=3)
    args = ap.parse_args()

    results = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "device": "cuda",
        "torch": torch.__version__,
        "engines": {},
    }
    for engine in (PER_EXPERT, GROUPED):
        prof = _profile_step(engine, args.iters)
        results["engines"][engine] = prof
        print(f"\n[{engine}] total self CUDA time = {prof['aggregates']['total_self_cuda_ms']} ms")
        print(f"  scatter-family = {prof['aggregates']['scatter_family_ms']} ms, "
              f"bmm = {prof['aggregates']['bmm_ms']} ms")
        for o in prof["top_ops"]:
            print(f"  {o['self_cuda_ms']:9.2f}ms  {o['calls']:4d} calls  {o['op']}")

    out = Path(__file__).resolve().parent / "results" / (
        f"profile_moe_{datetime.now():%Y%m%d_%H%M%S}.json")
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nProfile written -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
