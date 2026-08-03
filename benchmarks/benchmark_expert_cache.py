#!/usr/bin/env python3
"""
Metis — Expert cache benchmark
==============================
Measures the persistent expert weight cache across three dimensions:

  hitrate   — eval loop over N forwards; report hits/misses, hit rate,
              bandwidth reduction, resident memory.
  throughput — eval forward ms + full train-step ms (grad accumulation),
               cache on vs off → speedup and tok/s.
  memory    — peak GPU memory, cache on vs off.

Every measurement records median wall time and, on CUDA, peak GPU memory.
Results are written as JSON plus a Markdown report under ``benchmarks/results/``.

Usage:
    python benchmarks/benchmark_expert_cache.py
    python benchmarks/benchmark_expert_cache.py --mode hitrate
    python benchmarks/benchmark_expert_cache.py --mode throughput
    python benchmarks/benchmark_expert_cache.py --iters 20
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

# ── Timing / memory helpers ───────────────────────────────────────────────

class Timer:
    def __init__(self, device, runs=10, warmup=3):
        self.device = device
        self.runs = runs
        self.warmup = warmup
        self.is_cuda = device.startswith("cuda")

    def time(self, fn, *args, **kwargs):
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
                samples.append(start.elapsed_time(end))
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


def _make_cfg(device, cache_size, **kw):
    base = dict(
        d_model=128, n_heads=4, n_kv_heads=2, n_layers=4,
        max_seq_len=256, vocab_size=256, dropout=0.0,
        use_moe=True, moe_num_experts=8, moe_top_k=2,
        moe_engine="auto", device=device, moe_cache_size=cache_size,
    )
    base.update(kw)
    return ModelConfig(**base)


# ── Benchmark: hit rate ─────────────────────────────────────────────────

def bench_hitrate(device, iters):
    """Eval loop: same input → high hit rate; different inputs → cache misses."""
    torch.manual_seed(0)
    m = MetisLM(_make_cfg(device, cache_size=64)).to(device)
    m.eval()
    idx = torch.randint(0, 256, (2, 128), device=device)

    total_hits = total_misses = 0
    for _ in range(iters):
        with torch.no_grad():
            m(idx)  # same input every time → high hit rate
    for layer in m.layers:
        ffn = getattr(layer, "ffn", None)
        if hasattr(ffn, "cache_stats") and ffn.cache_stats() is not None:
            s = ffn.cache_stats()
            total_hits += s["hits"]
            total_misses += s["misses"]

    total = total_hits + total_misses
    hit_rate = total_hits / total if total else 0
    stats_list = m.get_moe_cache_stats()
    resident = sum(s["resident_bytes"] for s in stats_list if s)
    bw = sum(s["bytes_saved"] for s in stats_list if s) / max(
        sum(s["bytes_saved"] + s["bytes_built"] for s in stats_list if s), 1
    )
    print(f"  same-input forwards: {total_hits}/{total} hits ({hit_rate:.1%})")
    print(f"  bandwidth reduction: {bw*100:.1f}%")
    print(f"  resident memory:    {resident/1e6:.1f} MB")
    return {
        "hits": total_hits, "misses": total_misses,
        "hit_rate": hit_rate,
        "bandwidth_reduction_pct": bw * 100,
        "resident_bytes": resident,
    }


# ── Benchmark: throughput ───────────────────────────────────────────────

def bench_throughput(device, iters):
    """Eval forward + train step, cache on vs off."""
    use_amp = device.startswith("cuda")  # CPU fp16 autocast breaks grouped MoE
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    timer = Timer(device, runs=iters, warmup=2)
    rows = []

    for label, cache_size in [("cache_on", 64), ("cache_off", 0)]:
        torch.manual_seed(42)
        m = MetisLM(_make_cfg(device, cache_size=cache_size)).to(device)
        m.train()
        idx = torch.randint(0, 256, (2, 128), device=device)

        def train_step():
            m.zero_grad(set_to_none=True)
            if use_amp:
                with torch.autocast(device_type=device, dtype=amp_dtype):
                    _, loss, _ = m(idx, targets=idx)
            else:
                _, loss, _ = m(idx, targets=idx)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)

        r_train = timer.time(train_step)
        rows.append({
            "label": label, "eval_ms": r_train["median_ms"],
            "peak_mem_MB": r_train["peak_mem_MB"],
            "tokens_per_s": 2 * 128 / (r_train["median_ms"] / 1e3),
        })
        print(f"  {label:10s} train step: {r_train['median_ms']:7.2f}ms  "
              f"peak={r_train['peak_mem_MB']:7.1f}MB  "
              f"{rows[-1]['tokens_per_s']:.0f} tok/s")
        timer.release()

    speedup = rows[0]["eval_ms"] / max(rows[1]["eval_ms"], 1e-9)
    print(f"  speedup: {speedup:.2f}x")
    rows.append({"label": "speedup", "value": speedup})
    return rows


# ── Benchmark: real train step (with optimizer.step + invalidation) ────

def bench_real_train_step(device, iters):
    """Genuine training step (optimizer.step + invalidate) — cache cold each step."""
    use_amp = device.startswith("cuda")
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else \
                torch.float16 if use_amp else torch.float32
    timer = Timer(device, runs=iters, warmup=2)
    rows = []

    for label, cache_size in [("cache_on", 64), ("cache_off", 0)]:
        torch.manual_seed(42)
        m = MetisLM(_make_cfg(device, cache_size=cache_size)).to(device)
        m.train()
        idx = torch.randint(0, 256, (2, 128), device=device)
        optimizer = torch.optim.AdamW(m.parameters(), lr=3e-4, fused=False)

        def real_train_step():
            m.zero_grad(set_to_none=True)
            if use_amp:
                with torch.autocast(device_type=device, dtype=amp_dtype):
                    _, loss, _ = m(idx, targets=idx)
            else:
                _, loss, _ = m(idx, targets=idx)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            optimizer.step()
            m.invalidate_moe_caches()

        r = timer.time(real_train_step)
        rows.append({
            "label": label, "eval_ms": r["median_ms"],
            "peak_mem_MB": r["peak_mem_MB"],
            "tokens_per_s": 2 * 128 / (r["median_ms"] / 1e3),
        })
        print(f"  {label:10s} real step: {r['median_ms']:7.2f}ms  "
              f"peak={r['peak_mem_MB']:7.1f}MB  "
              f"{rows[-1]['tokens_per_s']:.0f} tok/s")
        timer.release()

    if len(rows) >= 2:
        speedup = rows[0]["eval_ms"] / max(rows[1]["eval_ms"], 1e-9)
        print(f"  speedup: {speedup:.2f}x")
        rows.append({"label": "speedup", "value": speedup})
    return rows


# ── Benchmark: decode throughput (1 token/step, KV cache) ──────────────

@torch.no_grad()
def bench_decode(device, iters):
    """Autoregressive decode: frozen weights, KV cache, 1 token/step."""
    use_amp = device.startswith("cuda")
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else \
                torch.float16 if use_amp else torch.float32
    N_TOKENS = 32
    timer = Timer(device, runs=max(iters, 5), warmup=2)
    rows = []

    for label, cache_size in [("cache_on", 64), ("cache_off", 0)]:
        torch.manual_seed(42)
        m = MetisLM(_make_cfg(device, cache_size=cache_size)).to(device)
        m.eval()
        prompt_len = 16
        prompt = torch.randint(0, 256, (1, prompt_len), device=device)

        def decode_step():
            # Feed prompt with KV cache, then decode N_TOKENS greedily
            kv_cache = None
            idx = prompt
            for _ in range(N_TOKENS):
                with torch.autocast(device_type=device.split(":")[0],
                                    dtype=amp_dtype, enabled=use_amp):
                    logits, _, kv_cache = m(idx, kv_cache=kv_cache)
                idx = logits[:, -1:, :].argmax(dim=-1)
            return idx

        r = timer.time(decode_step)
        ms_per_token = r["median_ms"] / N_TOKENS
        tok_s = 1000.0 / ms_per_token if ms_per_token > 0 else 0
        rows.append({
            "label": label, "median_ms": r["median_ms"],
            "ms_per_token": ms_per_token, "tokens_per_s": tok_s,
            "peak_mem_MB": r["peak_mem_MB"],
        })
        print(f"  {label:10s} decode: {r['median_ms']:7.2f}ms total  "
              f"{ms_per_token:.2f}ms/tok  {tok_s:.0f} tok/s  "
              f"peak={r['peak_mem_MB']:7.1f}MB")
        timer.release()

    if len(rows) >= 2:
        speedup = rows[0]["median_ms"] / max(rows[1]["median_ms"], 1e-9)
        print(f"  speedup: {speedup:.2f}x")
        rows.append({"label": "speedup", "value": speedup})
    return rows


# ── Benchmark: hit rate (mixed input) ──────────────────────────────────

@torch.no_grad()
def bench_hitrate_mixed(device, iters):
    """Mixed-input forwards: fresh random input each iteration."""
    torch.manual_seed(0)
    m = MetisLM(_make_cfg(device, cache_size=64)).to(device)
    m.eval()

    total_hits = total_misses = 0
    for _ in range(iters):
        idx = torch.randint(0, 256, (2, 128), device=device)
        m(idx)
    for layer in m.layers:
        ffn = getattr(layer, "ffn", None)
        if hasattr(ffn, "cache_stats") and ffn.cache_stats() is not None:
            s = ffn.cache_stats()
            total_hits += s["hits"]
            total_misses += s["misses"]
    total = total_hits + total_misses
    hit_rate = total_hits / total if total else 0
    print(f"  mixed-input forwards: {total_hits}/{total} hits ({hit_rate:.1%})")
    return {"hits": total_hits, "misses": total_misses, "hit_rate": hit_rate}


# ── Benchmark: memory ───────────────────────────────────────────────────

def bench_memory(device):
    """Peak GPU memory with and without cache."""
    if not device.startswith("cuda"):
        print("  (memory comparison requires CUDA — skipped)")
        return []
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    rows = []
    for T in (128, 256):
        for cache_size in (0, 64):
            label = "on" if cache_size else "off"
            torch.manual_seed(42)
            m = MetisLM(_make_cfg(device, cache_size=cache_size,
                                  max_seq_len=max(T, 256))).to(device)
            m.train()
            idx = torch.randint(0, 256, (1, T), device=device)

            # warmup
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
            rows.append({"T": T, "cache": label, "peak_MB": peak})
            print(f"  T={T:5d} cache_{label:3s}: peak={peak:8.1f}MB")
            torch.cuda.empty_cache()
            del m
    return rows


# ── Report ───────────────────────────────────────────────────────────────

def write_report(results, out_path):
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    md = out.with_suffix(".md")
    L = ["# Metis — Expert cache benchmark report", "",
         f"- Date: {results['date']}",
         f"- Device: `{results['device']}`",
         f"- PyTorch: `{results['torch']}`", ""]

    if results.get("hitrate"):
        h = results["hitrate"]
        L.append("## Hit rate (same-input eval forwards)")
        L.append("")
        L.append(f"- Hit rate: **{h['hit_rate']:.1%}** ({h['hits']}/{h['hits']+h['misses']})")
        L.append(f"- Stack+cast avoided: **{h['bandwidth_reduction_pct']:.1f}%**")
        L.append(f"- Resident memory: {h['resident_bytes']/1e6:.1f} MB")
        L.append("")

    if results.get("hitrate_mixed"):
        h = results["hitrate_mixed"]
        L.append("## Hit rate (mixed-input eval forwards)")
        L.append("")
        L.append(f"- Hit rate: **{h['hit_rate']:.1%}** ({h['hits']}/{h['hits']+h['misses']})")
        L.append("")

    if results.get("throughput"):
        L.append("## Throughput (steady-state warm cache, bf16 AMP)")
        L.append("")
        rows = [r for r in results["throughput"] if r["label"] != "speedup"]
        spd = [r for r in results["throughput"] if r["label"] == "speedup"]
        L.append("| config | train (ms) | peak mem (MB) | tok/s |")
        L.append("|--------|-----------:|--------------:|------:|")
        for r in rows:
            L.append(f"| {r['label']} | {r['eval_ms']:.2f} | "
                     f"{r['peak_mem_MB']:.1f} | {r['tokens_per_s']:.0f} |")
        if spd:
            L.append(f"\nSpeedup (cache_on / cache_off): **{spd[0]['value']:.2f}x**")
        L.append("")

    if results.get("real_train_step"):
        L.append("## Throughput (real train step — optimizer.step + invalidation)")
        L.append("")
        rows = [r for r in results["real_train_step"] if r["label"] != "speedup"]
        spd = [r for r in results["real_train_step"] if r["label"] == "speedup"]
        L.append("| config | step (ms) | peak mem (MB) | tok/s |")
        L.append("|--------|----------:|--------------:|------:|")
        for r in rows:
            L.append(f"| {r['label']} | {r['eval_ms']:.2f} | "
                     f"{r['peak_mem_MB']:.1f} | {r['tokens_per_s']:.0f} |")
        if spd:
            L.append(f"\nSpeedup (cache_on / cache_off): **{spd[0]['value']:.2f}x**")
        L.append("")

    if results.get("decode"):
        L.append("## Decode throughput (autoregressive, 1 token/step)")
        L.append("")
        rows = [r for r in results["decode"] if r["label"] != "speedup"]
        spd = [r for r in results["decode"] if r["label"] == "speedup"]
        L.append("| config | total (ms) | ms/tok | tok/s | peak mem (MB) |")
        L.append("|--------|----------:|-------:|------:|--------------:|")
        for r in rows:
            L.append(f"| {r['label']} | {r['median_ms']:.2f} | "
                     f"{r['ms_per_token']:.2f} | {r['tokens_per_s']:.0f} | "
                     f"{r['peak_mem_MB']:.1f} |")
        if spd:
            L.append(f"\nDecode speedup (cache_on / cache_off): **{spd[0]['value']:.2f}x**")
        L.append("")

    if results.get("memory"):
        L.append("## Peak memory")
        L.append("")
        by_t = {}
        for r in results["memory"]:
            by_t.setdefault(r["T"], {})[r["cache"]] = r["peak_MB"]
        L.append("| T | cache_off (MB) | cache_on (MB) | delta (MB) |")
        L.append("|---|---------------:|--------------:|-----------:|")
        for T in sorted(by_t):
            off = by_t[T].get("off", 0)
            on = by_t[T].get("on", 0)
            L.append(f"| {T} | {off:.1f} | {on:.1f} | {on-off:+.1f} |")
        L.append("")

    md.write_text("\n".join(L), encoding="utf-8")
    return str(md)


def git_sha():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True,
            cwd=os.path.dirname(__file__),
        )
        return out.stdout.strip() or "n/a"
    except Exception:
        return "n/a"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Benchmark the persistent expert weight cache."
    )
    parser.add_argument("--mode",
                        choices=["hitrate", "throughput", "memory", "decode",
                                 "realtrain", "hitrate_mixed", "both"],
                        default="both")
    parser.add_argument("--device", default=None)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    results = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "device": device,
        "torch": torch.__version__,
        "git_sha": git_sha(),
        "hitrate": {}, "hitrate_mixed": {}, "throughput": [],
        "real_train_step": [], "decode": [], "memory": [],
    }

    print(f"Metis Expert-Cache benchmark — device={device}\n")

    if args.mode in ("hitrate", "both"):
        print("[hitrate] same-input eval forwards:")
        results["hitrate"] = bench_hitrate(device, args.iters)
    if args.mode in ("hitrate_mixed", "both"):
        print("\n[hitrate_mixed] mixed-input eval forwards:")
        results["hitrate_mixed"] = bench_hitrate_mixed(device, args.iters)
    if args.mode in ("throughput", "both"):
        print("\n[throughput] steady-state train step (warm cache) cache-on vs off:")
        results["throughput"] = bench_throughput(device, args.iters)
    if args.mode in ("realtrain", "both"):
        print("\n[real_train_step] real train step (with optimizer.step + invalidate):")
        results["real_train_step"] = bench_real_train_step(device, args.iters)
    if args.mode in ("decode", "both"):
        print("\n[decode] autoregressive decode (1 token/step, KV cache):")
        results["decode"] = bench_decode(device, args.iters)
    if args.mode in ("memory", "both"):
        print("\n[memory] peak GPU memory:")
        results["memory"] = bench_memory(device)

    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results", f"benchmark_expert_cache_{datetime.now():%Y%m%d_%H%M%S}.json",
    )
    md = write_report(results, out)
    print("\nReport written:")
    print(f"  JSON: {out}")
    print(f"  Markdown: {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
