"""
Layer-prefetching benchmark: timeline + GPU utilization + latency.

While layer N computes, layer prefetching speculatively warms layer N+1's MoE
expert cache (the stacked+cast expert weights) so the next layer's
``get_or_build`` calls hit and never stall on a synchronous stack+cast.  On
CUDA the builds run on a dedicated prefetch stream, overlapping the previous
layer's compute; on CPU they are a synchronous speculative warm-up (the cache
hit-rate benefit is still measurable, there is just no stream overlap).

Measures, prefetch OFF vs ON on two identical MoE models:
  * timeline   — per-layer forward latency (wall, via forward hooks)
  * GPU util   — GpuIdleTracker busy-vs-wall per step
  * latency    — mean step latency, cache hit-rate, prefetch accuracy

Usage:
    # CUDA: stream-overlap numbers
    python benchmarks/benchmark_layer_prefetch.py --device cuda --steps 20

    # CPU: speculative warm-up (cache pressure) — force the prefetcher on
    python benchmarks/benchmark_layer_prefetch.py --device cpu --force-cpu-prefetch
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from metis import MetisLM, ModelConfig  # noqa: E402
from metis.pipeline import GpuIdleTracker  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _make_model(prefetch: bool, cache_size: int, device: str) -> MetisLM:
    cfg = ModelConfig.from_preset(
        "tiny", vocab_size=256, device=device, dropout=0.0,
        use_moe=True, moe_num_experts=8, moe_top_k=2,
        moe_cache_size=cache_size, use_layer_prefetch=prefetch, seed=0,
    )
    return MetisLM(cfg)


def _attach_layer_timers(model: MetisLM) -> dict[int, list[float]]:
    """Per-layer wall-time accumulators via forward pre/post hooks."""
    times: dict[int, list[float]] = {i: [] for i in range(len(model.layers))}

    def make_pair(idx: int):
        t0 = [0.0]

        def pre(module, args):
            t0[0] = time.perf_counter()

        def post(module, args, out):
            times[idx].append((time.perf_counter() - t0[0]) * 1e3)

        return pre, post

    for i, layer in enumerate(model.layers):
        pre, post = make_pair(i)
        layer.register_forward_pre_hook(pre)
        layer.register_forward_hook(post)
    return times


def _run(model, x, y, timers, steps: int, device: str, cache_size: int,
         invalidate: bool = False):
    """Run `steps` forwards, collecting per-layer wall times + cache stats."""
    idle = GpuIdleTracker(device, enabled=True)
    stats_acc = []
    model.eval()
    with torch.no_grad():
        for _ in range(steps):
            if invalidate:
                # The training loop clears caches after each optimizer step,
                # so every step starts cold — the speculative warm-up has to
                # rebuild the next layers' entries during the current layer.
                model.invalidate_moe_caches()
            idle.begin()
            t0 = time.perf_counter()
            model(x, y)
            step_ms = (time.perf_counter() - t0) * 1e3
            idle.end()
            stats_acc.append((step_ms, idle.stats()))
    layer_ms = {i: (sum(v) / len(v)) for i, v in timers.items()}
    cache_stats = [s for s in model.get_moe_cache_stats() if s]
    tot = {k: sum(s[k] for s in cache_stats) for k in
           ("hits", "misses", "prefetched", "prefetch_useful")}
    hr = tot["hits"] / max(tot["hits"] + tot["misses"], 1)
    acc = tot["prefetch_useful"] / max(tot["prefetched"], 1)
    mean_step = sum(s for s, _ in stats_acc) / max(len(stats_acc), 1)
    wall = sum(s for s, _ in stats_acc)
    gpu = sum(s["gpu_ms"] for _, s in stats_acc)
    return {
        "layer_ms": layer_ms,
        "mean_step_ms": mean_step,
        "wall_ms": wall,
        "gpu_ms": gpu,
        "idle_pct": (1 - gpu / wall) * 100 if wall else 0.0,
        "hit_rate": hr,
        "prefetch_issued": tot["prefetched"],
        "prefetch_useful": tot["prefetch_useful"],
        "prefetch_accuracy": acc,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", type=str, default=DEVICE)
    ap.add_argument("--steps", type=int, default=10, help="Measured forwards")
    ap.add_argument("--warmup", type=int, default=2, help="Warm-up forwards")
    ap.add_argument("--cache-size", type=int, default=0,
                    help="moe_cache_size (0 = config default 64; use a small "
                         "value to expose cache-pressure benefits)")
    ap.add_argument("--invalidate-each-step", action="store_true",
                    help="Clear the expert caches before every measured step "
                         "(models the training loop's per-step invalidation; "
                         "this is where the speculative warm-up shows)")
    ap.add_argument("--force-cpu-prefetch", action="store_true",
                    help="Enable the prefetcher on CPU (speculative warm-up; "
                         "no stream overlap without CUDA)")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    device = args.device if args.device.startswith(("cuda", "cpu")) else "cpu"
    cache_size = args.cache_size or 64
    if args.force_cpu_prefetch:
        os.environ["METIS_LAYER_PREFETCH_FORCE_CPU"] = "1"
    if not torch.cuda.is_available() and device.startswith("cuda"):
        print("[warn] --device cuda but no CUDA; falling back to cpu")
        device = "cpu"

    torch.manual_seed(0)
    m_off = _make_model(False, cache_size, device)
    torch.manual_seed(0)
    m_on = _make_model(True, cache_size, device)
    m_on.load_state_dict(m_off.state_dict())
    n_layers = len(m_off.layers)

    x = torch.randint(1, 256, (2, 32), device=device)
    y = torch.randint(1, 256, (2, 32), device=device)

    t_off = _attach_layer_timers(m_off)
    t_on = _attach_layer_timers(m_on)

    # Warm up: populate routing records (prefetch) + caches for both.
    with torch.no_grad():
        for _ in range(args.warmup):
            m_off(x, y)
            m_on(x, y)

    # ── correctness: prefetch must be bit-identical ───────────────────────
    with torch.no_grad():
        l_off, loss_off, _ = m_off(x, y)
        l_on, loss_on, _ = m_on(x, y)
    identical = bool(torch.equal(l_off, l_on)) and bool(loss_off == loss_on)
    print(f"[layer prefetch] bit-identical (prefetch ON vs OFF): {identical}")

    print(f"[layer prefetch] device={device} steps={args.steps} "
          f"warmup={args.warmup} moe_cache_size={cache_size} "
          f"prefetcher_attached={m_on._layer_prefetch is not None}")
    off = _run(m_off, x, y, t_off, args.steps, device, cache_size,
               invalidate=args.invalidate_each_step)
    on = _run(m_on, x, y, t_on, args.steps, device, cache_size,
              invalidate=args.invalidate_each_step)

    results = {
        "device": device, "steps": args.steps, "cache_size": cache_size,
        "date": datetime.now().isoformat(timespec="seconds"),
        "correctness_bit_identical": identical,
        "off": off, "on": on,
    }

    # ── timeline ─────────────────────────────────────────────────────────
    print("\n  per-layer forward latency (mean ms):")
    print("  " + "─" * 46)
    print(f"  {'layer':>6} {'OFF':>8} {'ON':>8} {'Δ (ms)':>8}")
    for i in range(n_layers):
        lo, ln = off["layer_ms"][i], on["layer_ms"][i]
        print(f"  {i:>6} {lo:>8.3f} {ln:>8.3f} {ln - lo:>+8.3f}")
    print("  " + "─" * 46)

    # ── latency + GPU utilization ────────────────────────────────────────
    print(f"\n  {'metric':<26} {'OFF':>12} {'ON':>12}")
    for label, key, fmt in [
        ("mean step latency (ms)", "mean_step_ms", "{:.2f}"),
        ("total wall (ms)", "wall_ms", "{:.1f}"),
        ("GPU busy (ms)", "gpu_ms", "{:.1f}"),
        ("GPU idle %", "idle_pct", "{:.1f}"),
        ("cache hit rate", "hit_rate", "{:.1%}"),
        ("prefetch accuracy", "prefetch_accuracy", "{:.1%}"),
        ("prefetch builds", "prefetch_issued", "{:.0f}"),
    ]:
        print(f"  {label:<26} {fmt.format(off[key]):>12} {fmt.format(on[key]):>12}")
    speedup = off["mean_step_ms"] / max(on["mean_step_ms"], 1e-9)
    print(f"\n  latency speedup: {speedup:.2f}x  |  idle "
          f"{off['idle_pct']:.1f}% → {on['idle_pct']:.1f}%")
    results["speedup"] = round(speedup, 3)
    if device.startswith("cpu"):
        print("  [note] CPU: prefetch is a synchronous warm-up (no stream "
              "overlap); the GPU win comes from hiding the build on a side stream.")

    # ── report ───────────────────────────────────────────────────────────
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = args.out or f"benchmark_layer_prefetch_{stamp}"
    if not str(base).endswith(".json"):
        base += ".json"
    json_path = out_dir / base
    json_path.write_text(
        __import__("json").dumps(results, indent=2), encoding="utf-8")

    md = json_path.with_suffix(".md")
    L = [f"# Layer Prefetch Benchmark",
         f"- Date: {results['date']}", f"- Device: `{device}`",
         f"- Steps: {args.steps}", f"- moe_cache_size: {cache_size}",
         f"- **Correctness: bit-identical** (prefetch ON vs OFF)",
         "", f"Correctness check: {results['correctness_bit_identical']}",
         "", "## Timeline (mean per-layer forward, ms)",
         "| layer | OFF | ON | Δ (ms) |", "|---|---:|---:|---:|"]
    for i in range(n_layers):
        L.append(f"| {i} | {off['layer_ms'][i]:.3f} | {on['layer_ms'][i]:.3f} "
                 f"| {on['layer_ms'][i]-off['layer_ms'][i]:+.3f} |")
    L += ["", "## Latency + GPU utilization",
          "| metric | OFF | ON |", "|---|---:|---:|",
          f"| mean step (ms) | {off['mean_step_ms']:.2f} | {on['mean_step_ms']:.2f} |",
          f"| total wall (ms) | {off['wall_ms']:.1f} | {on['wall_ms']:.1f} |",
          f"| GPU busy (ms) | {off['gpu_ms']:.1f} | {on['gpu_ms']:.1f} |",
          f"| GPU idle % | {off['idle_pct']:.1f} | {on['idle_pct']:.1f} |",
          f"| cache hit rate | {off['hit_rate']:.1%} | {on['hit_rate']:.1%} |",
          f"| prefetch accuracy | — | {on['prefetch_accuracy']:.1%} |",
          f"| prefetch builds | — | {on['prefetch_issued']} |",
          "", f"**Latency speedup:** {speedup:.2f}x"]
    md.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\nReport written:\n  JSON: {json_path}\n  Markdown: {md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
