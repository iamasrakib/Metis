#!/usr/bin/env python3
"""
Metis — Execution-scheduler benchmark
======================================
Compares the EAGER forward path against the SCHEDULED path, measuring wall
time, allocation count, peak memory, and sync points.

Results are written as JSON plus a Markdown report under ``benchmarks/results/``.

Usage:
    python benchmarks/benchmark_exec_plan.py                     # CPU, auto
    python benchmarks/benchmark_exec_plan.py --device cpu
    python benchmarks/benchmark_exec_plan.py --device cuda
    python benchmarks/benchmark_exec_plan.py --mode decode        # decode-only
    python benchmarks/benchmark_exec_plan.py --iters 20
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from metis.config import ModelConfig  # noqa: E402
from metis.model import MetisLM  # noqa: E402
from metis.scheduler import INFER, build_scheduler  # noqa: E402


class Timer:
    """Median wall time; peak CUDA memory if available."""

    def __init__(self, device, runs=10, warmup=3):
        self.device = device
        self.runs = runs
        self.warmup = warmup
        self.is_cuda = device.startswith("cuda")

    def time(self, fn, *args, **kwargs):
        for _ in range(self.warmup):
            fn(*args, **kwargs)
        if self.is_cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        base_mem = torch.cuda.memory_allocated() if self.is_cuda else 0
        samples = []
        for _ in range(self.runs):
            if self.is_cuda:
                s, e = torch.cuda.Event(True), torch.cuda.Event(True)
                s.record()
                fn(*args, **kwargs)
                e.record()
                torch.cuda.synchronize()
                samples.append(s.elapsed_time(e))
            else:
                t0 = time.perf_counter()
                fn(*args, **kwargs)
                samples.append((time.perf_counter() - t0) * 1e3)
        peak = (torch.cuda.max_memory_allocated() - base_mem) if self.is_cuda else 0
        samples.sort()
        return {
            "median_ms": round(samples[len(samples) // 2], 4),
            "mean_ms": round(sum(samples) / len(samples), 4),
            "peak_mem_MB": round(peak / 1e6, 2),
        }


def _cfg(**kw) -> ModelConfig:
    defaults = dict(vocab_size=256, d_model=64, n_heads=4, n_kv_heads=0,
                    n_layers=2, max_seq_len=32, dropout=0.0, use_rmsnorm=True,
                    use_swiglu=True, use_rope=True, tie_weights=True,
                    use_moe=False, use_qk_norm=False, use_attention_sink=False,
                    moe_num_experts=4, moe_top_k=2)
    defaults.update(kw)
    return ModelConfig(**defaults)


def bench_prefill(model, config, device, iters=10):
    idx = torch.randint(0, config.vocab_size, (1, config.max_seq_len), device=device)
    timer = Timer(device, runs=iters, warmup=3)

    # eager
    model.eval()
    eager = timer.time(lambda: model(idx))

    # scheduled
    sched = build_scheduler(model, mode=INFER, calibrate_run=False,
                            ref_shape=(1, config.max_seq_len), device=device)
    with torch.no_grad():
        scheduled = timer.time(lambda: sched.execute(idx))

    plan = sched.plan
    return {
        "eager": eager,
        "scheduled": scheduled,
        "speedup_ms": round(eager["median_ms"] - scheduled["median_ms"], 4),
        "plan": {
            "arena_bytes": plan.arena_bytes,
            "naive_peak_bytes": plan.naive_peak,
            "folded_allocs": plan.folded_allocs,
            "sync_points": plan.sync_points,
            "critical_path_len": len(plan.critical_path),
            "est_total_ms": round(plan.est_total_ms, 4),
        },
        "counters": sched.counters,
    }


def bench_decode(model, config, device, iters=10):
    timer = Timer(device, runs=iters, warmup=3)
    idx = torch.randint(0, config.vocab_size, (1, 1), device=device)
    model.eval()

    # warm up KV cache
    with torch.no_grad():
        _, _, cache_e = model(idx)
    sched = build_scheduler(model, mode=INFER, calibrate_run=False,
                            ref_shape=(1, 1), device=device)
    with torch.no_grad():
        _, _, cache_s = sched.execute(idx)

    def decode_eager():
        c = cache_e
        for _ in range(5):
            t = torch.randint(0, config.vocab_size, (1, 1), device=device)
            _, _, c = model(t, kv_cache=c)

    def decode_sched():
        c = cache_s
        for _ in range(5):
            t = torch.randint(0, config.vocab_size, (1, 1), device=device)
            _, _, c = sched.execute(t, kv_cache=c)

    eager = timer.time(decode_eager)
    scheduled = timer.time(decode_sched)
    return {"eager": eager, "scheduled": scheduled,
            "speedup_ms": round(eager["median_ms"] - scheduled["median_ms"], 4),
            "counters": sched.counters}


def main():
    parser = argparse.ArgumentParser(description="Benchmark execution scheduler")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--mode", default="both", choices=["prefill", "decode", "both"])
    parser.add_argument("--out", default=None)
    parser.add_argument("--preset", default=None)
    args = parser.parse_args()

    if args.preset:
        config = ModelConfig.from_preset(args.preset)
    else:
        config = _cfg()
    model = MetisLM(config).to(args.device)
    results = {"device": args.device, "config": args.preset or "tiny",
               "timestamp": datetime.now().isoformat()}

    if args.mode in ("prefill", "both"):
        results["prefill"] = bench_prefill(model, config, args.device, iters=args.iters)
        print(f"prefill eager={results['prefill']['eager']['median_ms']:.3f}ms "
              f"scheduled={results['prefill']['scheduled']['median_ms']:.3f}ms "
              f"folded={results['prefill']['plan']['folded_allocs']}")
    if args.mode in ("decode", "both"):
        results["decode"] = bench_decode(model, config, args.device, iters=args.iters)
        print(f"decode  eager={results['decode']['eager']['median_ms']:.3f}ms "
              f"scheduled={results['decode']['scheduled']['median_ms']:.3f}ms")

    # Write results
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = args.out or out_dir / f"benchmark_exec_plan_{ts}.json"
    md_path = Path(str(json_path).replace(".json", ".md"))

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    lines = ["# Execution-scheduler benchmark\n",
             f"Device: `{args.device}`  Iters: {args.iters}  "
             f"Time: {results['timestamp']}\n"]
    if "prefill" in results:
        p = results["prefill"]
        lines.append("## Prefill\n")
        lines.append("| Metric | Eager | Scheduled | Delta |\n")
        lines.append("|--------|-------|-----------|-------|\n")
        lines.append(f"| Median ms | {p['eager']['median_ms']} | "
                     f"{p['scheduled']['median_ms']} | "
                     f"{p['speedup_ms']} |\n")
        lines.append(f"| Peak mem MB | {p['eager']['peak_mem_MB']} | "
                     f"{p['scheduled']['peak_mem_MB']} |\n")
        lines.append(f"\nPlan: arena={p['plan']['arena_bytes']}B, naive_peak="
                     f"{p['plan']['naive_peak_bytes']}B, folded="
                     f"{p['plan']['folded_allocs']}, syncs="
                     f"{p['plan']['sync_points']}\n")
    if "decode" in results:
        d = results["decode"]
        lines.append("\n## Decode (5 steps)\n")
        lines.append("| Metric | Eager | Scheduled | Delta |\n")
        lines.append("|--------|-------|-----------|-------|\n")
        lines.append(f"| Median ms | {d['eager']['median_ms']} | "
                     f"{d['scheduled']['median_ms']} | "
                     f"{d['speedup_ms']} |\n")

    with open(md_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"results → {json_path}")


if __name__ == "__main__":
    main()
