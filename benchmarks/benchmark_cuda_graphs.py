#!/usr/bin/env python3
"""
Μῆτις (Metis) — CUDA Graphs benchmark
=====================================
Compares three ways to run one training iteration (``N`` micro-batch
forward+backward passes + scaler + clipping + optimizer step):

  * ``eager_ckpt``  — the original loop: eager, gradient checkpointing ON.
  * ``eager_nockpt``— eager, checkpointing OFF (the same compute the graph runs).
  * ``graph``       — one captured CUDA graph replayed per iteration.

Reported speedups:

  * ``graph vs eager_ckpt`` — the end-to-end win a user sees by enabling
    CUDA graphs (includes the checkpointing-off effect).
  * ``graph vs eager_nockpt`` — the *pure* CUDA-graph win (same compute, same
    kernels, only launch/sync overhead removed).

Modes:

  --mode step    median ms/step and speedup for one config
  --mode scaling sweep gradient_accumulation_steps {1,2,4,8} at one size
  --mode memory  peak CUDA activation memory per step (graph vs eager)

Results are written as JSON plus a Markdown report under ``benchmarks/results/``.

Usage:
    python benchmarks/benchmark_cuda_graphs.py                  # all modes, auto
    python benchmarks/benchmark_cuda_graphs.py --mode step
    python benchmarks/benchmark_cuda_graphs.py --iters 30 --out results/my_run.json
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
from metis.cuda_graphs import CUDAGraphStep  # noqa: E402
from metis.model import MetisLM  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def _make_cfg(**kw) -> ModelConfig:
    cfg = dict(
        d_model=128,
        n_heads=4,
        n_kv_heads=0,
        n_layers=4,
        max_seq_len=256,
        vocab_size=512,
        dropout=0.0,
        use_flash_attn=True,
        micro_batch_size=8,
        gradient_accumulation_steps=4,
        max_grad_norm=1.0,
        learning_rate=3e-4,
        use_moe=False,
        tie_weights=True,
    )
    cfg.update(kw)
    return ModelConfig(**cfg)


class Timer:
    """Median wall time over ``runs`` (GPU-synced); peak CUDA memory delta."""

    def __init__(self, runs: int = 20, warmup: int = 4):
        self.runs = runs
        self.warmup = warmup

    def timeit(self, fn) -> float:
        torch.cuda.synchronize()
        for _ in range(self.warmup):
            fn()
        torch.cuda.synchronize()
        times = []
        for _ in range(self.runs):
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
        times.sort()
        return times[len(times) // 2]

    def peak_delta(self, fn) -> float:
        """Peak CUDA allocator bytes added by one ``fn()`` call."""
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        fn()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        return max(0, peak - before)


def _config_summary(cfg) -> dict:
    return {
        k: v
        for k, v in cfg.__dict__.items()
        if not k.startswith("_") and isinstance(v, (int, float, str, bool))
    }


def _rand_batches(cfg, seed=0):
    torch.manual_seed(seed)
    N = cfg.gradient_accumulation_steps
    B, T = cfg.micro_batch_size, cfg.max_seq_len
    return [
        (torch.randint(0, cfg.vocab_size, (B, T)), torch.randint(0, cfg.vocab_size, (B, T)))
        for _ in range(N)
    ]


def _build(cfg, use_graphs: bool):
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    model = MetisLM(cfg).to(DEVICE).train()
    optimizer = model.configure_optimizers(0.1, cfg.learning_rate, DEVICE)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    amp = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    step = CUDAGraphStep(
        model, optimizer, scaler, cfg,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        micro_batch_size=cfg.micro_batch_size, max_seq_len=cfg.max_seq_len,
        amp_dtype=amp, device=DEVICE,
    )
    if use_graphs and not step.active:
        raise RuntimeError(f"CUDA graph capture failed: {step.reason}")
    return model, optimizer, scaler, amp, step


def _eager_step_fn(model, optimizer, scaler, cfg, amp, checkpoint: bool):
    N = cfg.gradient_accumulation_steps

    def fn():
        batches = _rand_batches(cfg, seed=int(time.time_ns() % (2**31)))
        optimizer.zero_grad(set_to_none=True)
        for x, y in batches:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            with torch.autocast("cuda", dtype=amp):
                _, loss, _ = model(x, y, use_checkpointing=checkpoint)
                loss = loss / N
            scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

    return fn


def _graph_step_fn(step, cfg):
    def fn():
        step.train_step(_rand_batches(cfg, seed=int(time.time_ns() % (2**31))))

    return fn


def measure_one(cfg, use_graphs, checkpoint, timer):
    model, optimizer, scaler, amp, step = _build(cfg, use_graphs)
    if use_graphs:
        fn = _graph_step_fn(step, cfg)
    else:
        fn = _eager_step_fn(model, optimizer, scaler, cfg, amp, checkpoint)
    ms = timer.timeit(fn)
    peak = timer.peak_delta(fn)
    return {"ms": ms, "peak_mb": peak / 2**20}


def mode_step(args, timer):
    print("\n[mode: step] one config, three execution strategies")
    cfg = _make_cfg()
    cfg.dropout = 0.0
    results = {}
    e_ckpt = measure_one(cfg, use_graphs=False, checkpoint=True, timer=timer)
    e_nock = measure_one(cfg, use_graphs=False, checkpoint=False, timer=timer)
    g = measure_one(cfg, use_graphs=True, checkpoint=False, timer=timer)
    results = {
        "eager_ckpt": e_ckpt,
        "eager_nockpt": e_nock,
        "graph": g,
        "speedup_vs_eager_ckpt": e_ckpt["ms"] / g["ms"],
        "speedup_vs_eager_nockpt": e_nock["ms"] / g["ms"],
        "config": _config_summary(cfg),
    }
    print(f"  eager_ckpt  : {e_ckpt['ms']:8.3f} ms/step ({e_ckpt['peak_mb']:8.1f} MB peak)")
    print(f"  eager_nockpt: {e_nock['ms']:8.3f} ms/step ({e_nock['peak_mb']:8.1f} MB peak)")
    print(f"  graph       : {g['ms']:8.3f} ms/step ({g['peak_mb']:8.1f} MB peak)")
    print(f"  speedup vs eager_ckpt  : {results['speedup_vs_eager_ckpt']:.3f}x")
    print(f"  speedup vs eager_nockpt: {results['speedup_vs_eager_nockpt']:.3f}x")
    return results


def mode_scaling(args, timer):
    print("\n[mode: scaling] gradient_accumulation_steps sweep (1→8)")
    results = {}
    for N in (1, 2, 4, 8):
        cfg = _make_cfg(gradient_accumulation_steps=N)
        cfg.dropout = 0.0
        e = measure_one(cfg, use_graphs=False, checkpoint=True, timer=timer)
        g = measure_one(cfg, use_graphs=True, checkpoint=False, timer=timer)
        results[str(N)] = {
            "eager_ckpt_ms": e["ms"],
            "graph_ms": g["ms"],
            "speedup": e["ms"] / g["ms"],
        }
        print(f"  N={N}: eager {e['ms']:7.3f} ms → graph {g['ms']:7.3f} ms "
              f"({e['ms'] / g['ms']:.3f}x)")
    return results


def mode_memory(args, timer):
    print("\n[mode: memory] peak CUDA activation memory per iteration")
    cfg = _make_cfg()
    cfg.dropout = 0.0

    # Eager baseline: peak regular-allocator memory of one checkpointed step.
    e = measure_one(cfg, use_graphs=False, checkpoint=True, timer=timer)

    # Graph: the graph pool (held at capture) + the regular-allocator replay
    # delta. Pool size = memory_allocated growth across a single capture.
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    model = MetisLM(cfg).to(DEVICE).train()
    optimizer = model.configure_optimizers(0.1, cfg.learning_rate, DEVICE)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    amp = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    before = torch.cuda.memory_allocated()
    step = CUDAGraphStep(
        model, optimizer, scaler, cfg,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        micro_batch_size=cfg.micro_batch_size, max_seq_len=cfg.max_seq_len,
        amp_dtype=amp, device=DEVICE,
    )
    torch.cuda.synchronize()
    pool_mb = (torch.cuda.memory_allocated() - before) / 2**20
    replay_peak = timer.peak_delta(lambda: step.train_step(_rand_batches(cfg))) / 2**20
    graph_total = pool_mb + replay_peak

    results = {
        "eager_ckpt_peak_mb": e["peak_mb"],
        "graph_pool_mb": pool_mb,
        "graph_replay_peak_mb": replay_peak,
        "graph_total_mb": graph_total,
        "delta_mb": graph_total - e["peak_mb"],
        "config": _config_summary(cfg),
    }
    print(f"  eager_ckpt peak      : {e['peak_mb']:7.1f} MB")
    print(f"  graph pool (capture) : {pool_mb:7.1f} MB")
    print(f"  graph replay peak    : {replay_peak:7.1f} MB")
    print(f"  graph total          : {graph_total:7.1f} MB")
    print(f"  delta                : {results['delta_mb']:+7.1f} MB")
    return results


def write_report(args, all_results: dict):
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = args.out or f"benchmark_cuda_graphs_{stamp}"
    json_path = RESULTS_DIR / f"{tag}.json"
    md_path = RESULTS_DIR / f"{tag}.md"
    json_path.write_text(json.dumps(all_results, indent=2, default=str))

    lines = ["# CUDA Graphs benchmark", "",
             f"- torch `{torch.__version__}`, CUDA `{torch.version.cuda}`",
             f"- GPU `{torch.cuda.get_device_name(0)}`", "",
             "## Results", ""]
    if "step" in all_results:
        s = all_results["step"]
        lines += ["### step", "",
                  f"- eager_ckpt: `{s['eager_ckpt']['ms']:.3f}` ms/step, "
                  f"`{s['eager_ckpt']['peak_mb']:.1f}` MB peak",
                  f"- eager_nockpt: `{s['eager_nockpt']['ms']:.3f}` ms/step, "
                  f"`{s['eager_nockpt']['peak_mb']:.1f}` MB peak",
                  f"- graph: `{s['graph']['ms']:.3f}` ms/step, "
                  f"`{s['graph']['peak_mb']:.1f}` MB peak",
                  f"- **speedup vs eager_ckpt**: `{s['speedup_vs_eager_ckpt']:.3f}x`",
                  f"- **speedup vs eager_nockpt**: `{s['speedup_vs_eager_nockpt']:.3f}x`",
                  ""]
    if "scaling" in all_results:
        lines += ["### scaling (gradient_accumulation_steps)", "",
                  "| N | eager (ms) | graph (ms) | speedup |", "|---|---|---|---|"]
        for n, row in all_results["scaling"].items():
            lines.append(f"| {n} | {row['eager_ckpt_ms']:.3f} | {row['graph_ms']:.3f} "
                         f"| {row['speedup']:.3f}x |")
        lines.append("")
    if "memory" in all_results:
        m = all_results["memory"]
        lines += ["### memory", "",
                  f"- eager_ckpt peak: `{m['eager_ckpt_peak_mb']:.1f}` MB",
                  f"- graph pool (capture): `{m['graph_pool_mb']:.1f}` MB",
                  f"- graph replay peak: `{m['graph_replay_peak_mb']:.1f}` MB",
                  f"- graph total: `{m['graph_total_mb']:.1f}` MB",
                  f"- delta: `{m['delta_mb']:+.1f}` MB", ""]
    md_path.write_text("\n".join(lines))
    print(f"\n  results → {json_path}")
    print(f"  report  → {md_path}")
    return str(json_path)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Metis CUDA Graphs benchmark")
    ap.add_argument("--mode", choices=["step", "scaling", "memory", "all"], default="all")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    _configure_stdio()

    if not DEVICE.startswith("cuda"):
        print("CUDA Graphs benchmark requires CUDA (found CPU) — skipping.")
        return 0

    print(f"Metis CUDA Graphs benchmark — GPU {torch.cuda.get_device_name(0)}")
    timer = Timer(runs=args.iters, warmup=args.warmup)
    all_results = {"device": torch.cuda.get_device_name(0),
                   "torch": torch.__version__,
                   "iters": args.iters}
    if args.mode in ("step", "all"):
        all_results["step"] = mode_step(args, timer)
    if args.mode in ("scaling", "all"):
        all_results["scaling"] = mode_scaling(args, timer)
    if args.mode in ("memory", "all"):
        all_results["memory"] = mode_memory(args, timer)
    write_report(args, all_results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
