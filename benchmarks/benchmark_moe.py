#!/usr/bin/env python3
"""
Μῆτις (Metis) — Grouped MoE benchmark
======================================
Compares the LEGACY per-expert MoE loop against the NEW grouped execution
engine (token sorting → expert batching → grouped GEMM → grouped SwiGLU →
grouped output projection) at three levels:

  layer   — one MoE layer forward+backward, swept over token counts and
            expert counts; CUDA kernel-launch counts via torch.profiler
  model   — full MetisLM train step under AMP (bf16/fp16) and eval forward
  memory  — peak activation memory of one train step per engine

Every measurement records median wall time and, on CUDA, peak GPU memory.
Results are written as JSON plus a Markdown report under ``benchmarks/results/``.

Usage:
    python benchmarks/benchmark_moe.py                    # all modes, auto device
    python benchmarks/benchmark_moe.py --mode layer
    python benchmarks/benchmark_moe.py --mode model
    python benchmarks/benchmark_moe.py --mode memory
    python benchmarks/benchmark_moe.py --iters 20 --out results/my_run.json
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
)

# ── Timing / memory helpers ───────────────────────────────────────────────────

class Timer:
    """Median wall time over ``runs``; peak CUDA memory delta if on GPU."""

    def __init__(self, device: str, runs: int = 10, warmup: int = 3):
        self.device = device
        self.runs = runs
        self.warmup = warmup
        self.is_cuda = device.startswith("cuda")

    def time(self, fn, *args, **kwargs) -> dict:
        for _ in range(self.warmup):
            fn(*args, **kwargs)
        torch.cuda.synchronize() if self.is_cuda else None

        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats()
        base_mem = torch.cuda.memory_allocated() if self.is_cuda else 0

        samples = []
        for _ in range(self.runs):
            if self.is_cuda:
                start, end = torch.cuda.Event(True), torch.cuda.Event(True)
                start.record()
                fn(*args, **kwargs)
                end.record()
                torch.cuda.synchronize()
                samples.append(start.elapsed_time(end))  # ms
            else:
                t0 = time.perf_counter()
                fn(*args, **kwargs)
                samples.append((time.perf_counter() - t0) * 1e3)
        peak = (torch.cuda.max_memory_allocated() - base_mem) if self.is_cuda else 0
        samples.sort()
        return {
            "median_ms": samples[len(samples) // 2],
            "mean_ms": sum(samples) / len(samples),
            "peak_mem_MB": peak / 1e6,
        }

    def release(self):
        if self.is_cuda:
            torch.cuda.empty_cache()


def kernel_launch_count(fn, device: str, *args) -> int:
    """Count CUDA-launching ops for one call of ``fn`` (launch-count proxy).

    The torch profilers on this Windows CUDA build surface aten-op-level CUDA
    time but not leaf CUDA kernel events (and no Nsight tooling is present),
    so this counts the aten ops that consumed CUDA device time — a consistent,
    machine-comparable proxy for kernel launches.
    """
    if not device.startswith("cuda"):
        return 0
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CUDA,
            torch.profiler.ProfilerActivity.CPU,
        ]
    ) as prof:
        fn(*args)
    torch.cuda.synchronize()
    return sum(
        1 for e in prof.events()
        if e.device_time is not None and "memcpy" not in e.name.lower()
    )


def _moes(device: str, d_model: int, experts: int, top_k: int, seed: int = 0):
    """Identical-weight MoE twins, one per engine."""
    base = dict(
        d_model=d_model, n_heads=4, n_kv_heads=2, n_layers=2, max_seq_len=1024,
        vocab_size=256, dropout=0.0, use_moe=True,
        moe_num_experts=experts, moe_top_k=top_k,
    )
    torch.manual_seed(seed)
    g = MoE(ModelConfig(**base, moe_engine=GROUPED)).to(device)
    p = MoE(ModelConfig(**base, moe_engine=PER_EXPERT)).to(device)
    p.load_state_dict(g.state_dict())
    return g, p


def bench_layer(device: str, iters: int) -> list:
    """MoE layer forward+backward: per_expert vs grouped across sizes.

    Runs under bf16 AMP (the realistic training dtype). The per-expert path's
    per-expert boolean-mask scatters dominate there; fp32 GEMMs hide that.
    """
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    amp_ctx = (torch.autocast(device_type=device, dtype=amp_dtype)
               if device.startswith("cuda") else torch.no_grad())
    timer = Timer(device, runs=iters, warmup=3)
    rows = []

    def train_step(moe, x):
        with amp_ctx:
            out = moe(x)
        out.sum().backward()

    for d_model in (256, 512):
        for n_tokens in (128, 512, 1024, 2048):
            for experts in (8, 16):
                g, p = _moes(device, d_model, experts, 2)
                x = torch.randn(1, n_tokens, d_model, device=device)
                g.train(); p.train()

                def _g():
                    g.zero_grad(set_to_none=True)
                    train_step(g, x)
                def _p():
                    p.zero_grad(set_to_none=True)
                    train_step(p, x)

                tg = timer.time(_g)
                tp = timer.time(_p)
                launches_g = kernel_launch_count(_g, device)
                launches_p = kernel_launch_count(_p, device)
                speedup = tp["median_ms"] / max(tg["median_ms"], 1e-9)
                rows.append({
                    "d_model": d_model, "n_tokens": n_tokens, "n_experts": experts,
                    "grouped_ms": tg["median_ms"], "per_expert_ms": tp["median_ms"],
                    "speedup": speedup,
                    "grouped_launches": launches_g, "per_expert_launches": launches_p,
                    "launch_reduction_pct": 100.0 * (1 - launches_g / max(launches_p, 1)),
                    "grouped_peak_mem_MB": tg["peak_mem_MB"],
                    "per_expert_peak_mem_MB": tp["peak_mem_MB"],
                })
                print(
                    f"  d={d_model:4d} N={n_tokens:5d} E={experts:2d}: "
                    f"per_expert={tp['median_ms']:7.2f}ms grouped={tg['median_ms']:7.2f}ms "
                    f"({speedup:5.2f}x) launches {launches_p}→{launches_g}"
                )
                timer.release()
    return rows


def _routing(n_tokens, n_experts, top_k, skew, seed=0):
    """Craft a routing matrix; ``skew`` concentrates tokens on expert 0."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.stack([torch.randperm(n_experts, generator=g)[:top_k]
                       for _ in range(n_tokens)])
    if skew > 0:
        n_hot = int(n_tokens * skew)
        idx[:n_hot, 0] = 0
        idx[:n_hot, 1:] = torch.randint(1, n_experts, (n_hot, top_k - 1),
                                        generator=g)
    w = torch.rand(n_tokens, top_k, generator=g)
    w = w / w.sum(-1, keepdim=True)
    return w, idx


def _padding_waste(counts, groups):
    """Total padded-but-empty slots for a group schedule."""
    waste = 0
    for g in groups:
        A_g = len(g)
        ga = torch.as_tensor(sorted(g), dtype=counts.dtype)
        gt = int(counts[ga].sum())
        max_m = max(int(counts[ga].max()), (gt + A_g - 1) // A_g)
        waste += A_g * max_m - gt
    return waste


def bench_schedule(device: str, iters: int) -> list:
    """Before/after: OLD (global-max-padded) vs NEW (grouped + dynamic) schedule.

    Runs the same MoE layer forward+backward under both schedulers with
    identical weights and routing, sweeping routing skew (uniform → expert 0
    swamped). Reports wall time, padded-waste %, expert-group count, and a
    parity check that the NEW scheduler reproduces the OLD output.
    """
    timer = Timer(device, runs=iters, warmup=2)
    rows = []

    for d_model, hidden in ((256, 512), (512, 1024)):
        for n_experts in (8, 16):
            for skew in (0.0, 0.25, 0.5):
                n_tokens = 1024
                tw, idx = _routing(n_tokens, n_experts, 2, skew)
                tw, idx = tw.to(device), idx.to(device)
                x = torch.randn(n_tokens, d_model, device=device)
                w1 = [torch.randn(d_model, hidden, device=device,
                                  requires_grad=True) for _ in range(n_experts)]
                w2 = [torch.randn(hidden, d_model, device=device,
                                  requires_grad=True) for _ in range(n_experts)]

                def step(fn, ratio=None):
                    def _run():
                        for p in w1 + w2:
                            p.grad = None
                        if ratio is None:
                            out = fn(x, tw, idx, w1, w2, top_k=2,
                                     num_experts=n_experts)
                        else:
                            out = fn(x, tw, idx, w1, w2, top_k=2,
                                     num_experts=n_experts, group_max_ratio=ratio)
                        out.sum().backward()
                    return _run

                f_new = step(forward_grouped, ratio=2.0)
                f_old = step(forward_grouped_legacy)
                t_new = timer.time(f_new)
                t_old = timer.time(f_old)
                timer.release()

                # parity: new reproduces old (fp32 reference, no AMP)
                with torch.no_grad():
                    o_old = forward_grouped_legacy(
                        x.float(), tw.float(), idx, [p.float() for p in w1],
                        [p.float() for p in w2], top_k=2, num_experts=n_experts)
                    o_new = forward_grouped(
                        x.float(), tw.float(), idx, [p.float() for p in w1],
                        [p.float() for p in w2], top_k=2, num_experts=n_experts,
                        group_max_ratio=2.0)
                max_parity_diff = float((o_new - o_old).abs().max())

                counts = torch.bincount(idx.reshape(-1), minlength=n_experts)
                active = torch.nonzero(counts, as_tuple=False).flatten()
                groups = _group_active_experts(counts, active, 2.0)
                waste_old = _padding_waste(counts, [active.tolist()])
                waste_new = _padding_waste(counts, groups)
                total = int(counts.sum())
                rows.append({
                    "d_model": d_model, "n_tokens": n_tokens, "n_experts": n_experts,
                    "skew": skew, "n_groups": len(groups),
                    "old_ms": t_old["median_ms"], "new_ms": t_new["median_ms"],
                    "speedup": t_old["median_ms"] / max(t_new["median_ms"], 1e-9),
                    "old_pad_waste_pct": 100.0 * waste_old / total,
                    "new_pad_waste_pct": 100.0 * waste_new / total,
                    "waste_reduction_pct": 100.0 * (1 - waste_new / max(waste_old, 1)),
                    "max_parity_diff": max_parity_diff,
                })
                print(
                    f"  d={d_model:3d} E={n_experts:2d} skew={skew:4.2f}: "
                    f"old={t_old['median_ms']:7.2f}ms new={t_new['median_ms']:7.2f}ms "
                    f"({rows[-1]['speedup']:5.2f}x) waste {rows[-1]['old_pad_waste_pct']:5.1f}%"
                    f"→{rows[-1]['new_pad_waste_pct']:5.1f}% parity={max_parity_diff:.1e}"
                )
    return rows


def bench_model(device: str, iters: int) -> list:
    """Full MetisLM train step (AMP) and eval forward, both engines."""
    timer = Timer(device, runs=iters, warmup=1)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    rows = []

    def build(engine):
        cfg = ModelConfig(
            d_model=128, n_heads=4, n_kv_heads=2, n_layers=4, max_seq_len=256,
            vocab_size=256, dropout=0.0, use_moe=True, moe_num_experts=8,
            moe_top_k=2, moe_engine=engine,
        )
        m = MetisLM(cfg).to(device)
        return m

    for engine, label in ((PER_EXPERT, "per_expert"), (GROUPED, "grouped")):
        m = build(engine)
        m.train()
        idx = torch.randint(0, 256, (2, 128), device=device)

        def step():
            m.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=amp_dtype):
                _, loss, _ = m(idx, targets=idx)
            loss.backward()

        r = timer.time(step)
        used = m.layers[0].ffn.last_engine
        rows.append({
            "engine": label, "backend_used": used, "dtype": str(amp_dtype),
            "train_ms": r["median_ms"], "train_peak_mem_MB": r["peak_mem_MB"],
            "tokens_per_s": 2 * 128 / (r["median_ms"] / 1e3),
            "train_launches": kernel_launch_count(step, device),
        })
        print(f"  train {label:11s} ({used}): {r['median_ms']:8.2f}ms  "
              f"peak={r['peak_mem_MB']:7.1f}MB  {rows[-1]['tokens_per_s']:.0f} tok/s  "
              f"launches={rows[-1]['train_launches']}")
        timer.release()
    return rows


def bench_memory(device: str) -> list:
    """Peak activation memory of one train step vs sequence length."""
    if not device.startswith("cuda"):
        print("  (memory comparison requires CUDA — skipped)")
        return []
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    rows = []
    for T in (128, 256, 512):
        for engine in (PER_EXPERT, GROUPED):
            cfg = ModelConfig(
                d_model=128, n_heads=4, n_kv_heads=2, n_layers=4, max_seq_len=1024,
                vocab_size=512, dropout=0.0, use_moe=True, moe_num_experts=8,
                moe_top_k=2, moe_engine=engine,
            )
            m = MetisLM(cfg).to(device)
            m.train()
            idx = torch.randint(0, 512, (1, T), device=device)

            # Warmup (allocates caches/lazy state), then measured pass.
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                _, loss, _ = m(idx, targets=idx)
            loss.backward()
            m.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

            torch.cuda.reset_peak_memory_stats()
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                _, loss, _ = m(idx, targets=idx)
            loss.backward()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / 1e6
            rows.append({"T": T, "engine": engine, "peak_mem_MB": peak})
            print(f"  T={T:5d} {engine:11s}: peak={peak:8.1f}MB")
            torch.cuda.empty_cache()
            del m
    return rows


# ── Report ────────────────────────────────────────────────────────────────────

def write_report(results: dict, out_path: str) -> str:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    md = out.with_suffix(".md")
    L = []
    L.append("# Metis — Grouped MoE benchmark report")
    L.append("")
    L.append(f"- Date: {results['date']}")
    L.append(f"- Device: `{results['device']}`")
    L.append(f"- PyTorch: `{results['torch']}`")
    L.append(f"- Git SHA: `{results.get('git_sha', 'n/a')}`")
    L.append("")
    L.append("## Engine capabilities")
    for k, v in results["engines"].items():
        L.append(f"- {k}: {v}")
    L.append("")

    if results.get("schedule"):
        L.append("## Scheduling: legacy (max-padded) vs redesigned (grouped+dynamic)")
        L.append("")
        L.append("| d | E | skew | groups | old (ms) | new (ms) | speedup | "
                 "pad waste old→new (%) | parity diff |")
        L.append("|---|--:|-----:|-------:|---------:|---------:|--------:|"
                 "------------------:|:---:|")
        for r in results["schedule"]:
            L.append(
                f"| {r['d_model']} | {r['n_experts']} | {r['skew']:.2f} | "
                f"{r['n_groups']} | {r['old_ms']:.2f} | {r['new_ms']:.2f} | "
                f"{r['speedup']:.2f}x | {r['old_pad_waste_pct']:.1f}→"
                f"{r['new_pad_waste_pct']:.1f} | {r['max_parity_diff']:.1e} |"
            )
        L.append("")
        L.append("* `skew` = fraction of tokens forced onto expert 0; "
                 "`groups` = expert groups formed by the new scheduler; "
                 "`parity diff` = max |new − old| output (fp32).")
        L.append("")

    if results.get("layer"):
        L.append("## Layer-level: MoE forward+backward")
        L.append("")
        L.append("| d | N tokens | E | per_expert (ms) | grouped (ms) | speedup "
                 "| launches p→g | mem p→g (MB) |")
        L.append("|---|---------:|--:|----------------:|-------------:|--------:|:---:|:---:|")
        for r in results["layer"]:
            L.append(
                f"| {r['d_model']} | {r['n_tokens']} | {r['n_experts']} | "
                f"{r['per_expert_ms']:.3f} | {r['grouped_ms']:.3f} | "
                f"{r['speedup']:.2f}x | {r['per_expert_launches']}→{r['grouped_launches']} | "
                f"{r['per_expert_peak_mem_MB']:.1f}→{r['grouped_peak_mem_MB']:.1f} |"
            )
        L.append("")

    if results.get("model"):
        L.append("## Model-level: full MetisLM train step (AMP)")
        L.append("")
        L.append("| engine | dtype | train (ms) | peak mem (MB) | tok/s | launches |")
        L.append("|--------|-------|-----------:|--------------:|------:|---------:|")
        for r in results["model"]:
            L.append(
                f"| {r['engine']} | {r['dtype']} | {r['train_ms']:.2f} | "
                f"{r['train_peak_mem_MB']:.1f} | {r['tokens_per_s']:.0f} | "
                f"{r['train_launches']} |"
            )
        L.append("")

    if results.get("memory"):
        L.append("## Memory: peak activation memory vs sequence length")
        L.append("")
        L.append("| T | per_expert peak (MB) | grouped peak (MB) | saving (MB) | saving % |")
        L.append("|---|---------------------:|------------------:|------------:|---------:|")
        by_t = {}
        for r in results["memory"]:
            by_t.setdefault(r["T"], {})[r["engine"]] = r["peak_mem_MB"]
        for T in sorted(by_t):
            p, g = by_t[T][PER_EXPERT], by_t[T][GROUPED]
            L.append(f"| {T} | {p:.1f} | {g:.1f} | {p-g:.1f} | {100*(p-g)/max(p,1e-9):.1f}% |")
        L.append("")

    md.write_text("\n".join(L), encoding="utf-8")
    return str(md)


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=os.path.dirname(__file__),
        )
        return out.stdout.strip() or "n/a"
    except Exception:
        return "n/a"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark per-expert vs grouped MoE execution."
    )
    parser.add_argument("--mode", choices=["layer", "model", "memory",
                                           "schedule", "both"],
                        default="both")
    parser.add_argument("--device", default=None, help="cpu | cuda (default: auto)")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--out", default=None, help="Output JSON path")
    args = parser.parse_args(argv)

    _configure_stdio()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    results = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "device": device,
        "torch": torch.__version__,
        "git_sha": git_sha(),
        "engines": detect_moe_engines(),
        "layer": [], "model": [], "memory": [], "schedule": [],
    }

    print(f"Metis MoE benchmark — device={device} engine=grouped vs per_expert")
    print(f"Capabilities: {results['engines']}\n")

    if args.mode in ("layer", "both"):
        print("[layer-level] MoE forward+backward:")
        results["layer"] = bench_layer(device, args.iters)
    if args.mode in ("model", "both"):
        print("\n[model-level] full MetisLM train step (AMP):")
        results["model"] = bench_model(device, args.iters)
    if args.mode in ("memory", "both"):
        print("\n[memory] peak activation memory vs seq len:")
        results["memory"] = bench_memory(device)
    if args.mode in ("schedule", "both"):
        print("\n[schedule] before/after: legacy max-padded vs grouped+dynamic:")
        results["schedule"] = bench_schedule(device, args.iters)

    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results", f"benchmark_moe_{datetime.now():%Y%m%d_%H%M%S}.json",
    )
    md = write_report(results, out)
    print("\nReport written:")
    print(f"  JSON: {out}")
    print(f"  Markdown: {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
