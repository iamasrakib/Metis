#!/usr/bin/env python3
"""
Metis — Training-pipeline overlap benchmark
============================================
Measures GPU idle time for the serial vs the overlapped (software-pipelined)
training loop: thread prefetching, pinned memory, non-blocking H2D on a copy
stream, and async checkpoints.

For each mode it reports, per step, the **wall time**, the **GPU-busy time**
(CUDA events), and the **GPU idle %** (fraction of wall time the GPU waited on
the CPU). A ``--slow-ms`` option injects per-sample CPU latency to simulate a
slow disk / expensive tokenizer — where the overlap win grows.

Both modes train the *same* tiny MoE model on the *same* batches with the same
seed; losses are bit-identical by construction (see
``verify_pipeline_parity.py``).

Usage:
    python benchmarks/benchmark_pipeline_overlap.py                  # both modes
    python benchmarks/benchmark_pipeline_overlap.py --steps 30
    python benchmarks/benchmark_pipeline_overlap.py --slow-ms 5      # slow disk
    python benchmarks/benchmark_pipeline_overlap.py --device cpu
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from metis.data import CharTokenizer, create_dataloader, train_val_split  # noqa: E402
from metis.pipeline import (  # noqa: E402
    GpuBatchStager,
    GpuIdleTracker,
    ThreadPrefetcher,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_env(slow_ms):
    """A tiny MoE model + loader, same for both pipeline modes.

    When ``slow_ms > 0`` the loader is wrapped so every micro-batch costs an
    extra ``slow_ms`` of CPU time (disk/tokenization stand-in). In serial mode
    that latency lands on the hot path; in overlapped mode it hides inside the
    prefetch thread.
    """
    torch.manual_seed(0)
    text = "The quick brown fox jumps over the lazy dog. " * 500
    tok = CharTokenizer()
    tok.fit(text)
    train_text, _ = train_val_split(text, 0.9)
    base = create_dataloader(
        train_text, tok, 64, 8, shuffle=False, use_mmap=False, num_workers=0
    )

    def slow_loader():
        while True:
            it = iter(base)
            try:
                for batch in it:
                    if slow_ms:
                        time.sleep(slow_ms / 1000)
                    yield batch
            except StopIteration:
                pass

    loader = slow_loader() if slow_ms else base
    from metis import ModelConfig, MetisLM

    cfg = ModelConfig.from_preset(
        "tiny", max_iters=1, vocab_size=tok.vocab_size, device=DEVICE,
        use_moe=True, moe_num_experts=4, moe_top_k=2, gradient_accumulation_steps=4,
    )
    torch.manual_seed(0)
    m = MetisLM(cfg)
    m.to(DEVICE)
    m.train()
    opt = m.configure_optimizers(0.1, 3e-4, DEVICE)
    scaler = torch.amp.GradScaler(DEVICE, enabled=True)
    return loader, m, opt, scaler


def run_steps(mode, steps, slow_ms):
    """Run `steps` training steps; return per-step stats + loss list.

    Only the consumer for the active mode touches the loader (a generator is
    single-iter; the prefetcher must not steal batches from the serial path).
    """
    loader, m, opt, scaler = make_env(slow_ms)
    use_amp = DEVICE.startswith("cuda")  # CPU bf16 autocast is ~20× slower; keep it fp32
    if mode == "overlapped":
        prefetcher = ThreadPrefetcher(loader, micro_batches=4, prefetch_depth=2).start()
    else:
        prefetcher = None
        data_iter = iter(loader)
    stager = GpuBatchStager(DEVICE, depth=3)
    idle = GpuIdleTracker(DEVICE)
    losses = []
    try:
        for _ in range(steps):
            idle.begin()
            opt.zero_grad(set_to_none=True)
            loss_accum = 0.0
            # ── data pull (data_wait) ── pull the whole step's batches.
            if mode == "overlapped":
                batches = prefetcher.next_step()
            else:  # serial
                batches = [next(data_iter) for _ in range(4)]
            idle.tick("data_wait")
            if mode == "overlapped":
                stager.stage(batches[0])
            idle.tick("h2d")
            # ── compute (forward/backward loop) ──
            for i in range(4):
                if mode == "overlapped":
                    if i + 1 < 4:
                        stager.stage(batches[i + 1])
                    x, y, extra = stager.device()
                else:
                    x, y = batches[i]
                    x, y = x.to(DEVICE), y.to(DEVICE)
                if use_amp:
                    with torch.autocast(DEVICE, dtype=torch.bfloat16):
                        _, loss, _ = m(x, y, use_checkpointing=True)
                else:
                    _, loss, _ = m(x, y, use_checkpointing=True)
                loss = loss / 4
                scaler.scale(loss).backward()
                loss_accum += loss.item()
                if mode == "overlapped":
                    stager.mark_done()
            idle.tick("compute")
            # ── optimizer ──
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            m.invalidate_moe_caches()
            idle.tick("optimizer")
            losses.append(loss_accum)
            idle.end()
    finally:
        if prefetcher is not None:
            prefetcher.stop()
    return idle.stats(), losses


def run_checkpoint_stress(steps, slow_ms, tmp_dir):
    """Measure the per-step overhead of checkpointing every step.

    Compares three variants on the same model + batches:
      * none       — no checkpointing
      * sync       — synchronous ``torch.save`` on the hot path
      * async      — ``AsyncCheckpointer`` (async D2H + writer-thread write)

    Returns per-variant wall_ms.
    """
    from metis.pipeline import AsyncCheckpointer
    results = {}

    def variant(mode, steps, slow_ms, tmp_dir):
        loader, m, opt, scaler = make_env(slow_ms)
        prefetcher = ThreadPrefetcher(loader, micro_batches=4,
                                      prefetch_depth=2).start()
        stager = GpuBatchStager(DEVICE, depth=3)
        idle = GpuIdleTracker(DEVICE)
        checkpointer = AsyncCheckpointer(max_pending=1) if mode == "async" else None
        try:
            t0 = time.perf_counter()
            for s in range(steps):
                opt.zero_grad(set_to_none=True)
                batches = prefetcher.next_step()
                stager.stage(batches[0])
                for i in range(4):
                    if i + 1 < 4:
                        stager.stage(batches[i + 1])
                    x, y, extra = stager.device()
                    loss = m(x, y, use_checkpointing=False)[1] / 4
                    scaler.scale(loss).backward()
                    stager.mark_done()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                if checkpointer is not None:
                    checkpointer.wait_pending()
                scaler.step(opt)
                scaler.update()
                m.invalidate_moe_caches()
                path = os.path.join(tmp_dir, f"ck_{s}.pt")
                if mode == "async":
                    ckpt = {"model": m.state_dict(), "step": s}
                    compute_done = torch.cuda.Event() if DEVICE.startswith("cuda") else None
                    if compute_done is not None:
                        compute_done.record()
                    checkpointer.submit_async(path, ckpt, compute_done=compute_done)
                elif mode == "sync":
                    torch.save({"model": m.state_dict(), "step": s}, path)
            if checkpointer is not None:
                checkpointer.flush()
            return (time.perf_counter() - t0) * 1e3
        finally:
            prefetcher.stop()
            if checkpointer is not None:
                checkpointer.close()

    for mode in ("none", "sync", "async"):
        results[mode] = variant(mode, steps, slow_ms, tmp_dir)
        print(f"  checkpoint[{mode:5s}] {steps} steps: {results[mode]:8.1f}ms  "
              f"({results[mode]/steps:6.2f} ms/step)")
    return results


def _stdio():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", type=str, default=DEVICE)
    ap.add_argument("--steps", type=int, default=25,
                    help="Training steps per mode")
    ap.add_argument("--slow-ms", type=float, default=0.0,
                    help="Simulated per-sample CPU latency (ms) — slow disk/Tokenizer")
    ap.add_argument("--stage", action="store_true",
                    help="Report the per-stage wall-time breakdown (data_wait / "
                         "h2d / compute / optimizer)")
    ap.add_argument("--checkpoint-stress", action="store_true",
                    help="Measure per-step checkpoint overhead (none / sync / async)")
    ap.add_argument("--out", type=str, default=None,
                    help="Optional JSON output path (also writes .md next to it)")
    args = ap.parse_args()
    _stdio()

    device = args.device if args.device.startswith(("cuda", "cpu")) else "cpu"
    label = f"{'slow-disk' if args.slow_ms else 'fast'} pipeline comparison"
    print(f"[pipeline overlap] device={device} steps={args.steps} "
          f"slow_ms={args.slow_ms}")

    results = {"device": device, "steps": args.steps, "slow_ms": args.slow_ms,
               "date": datetime.now().isoformat(timespec="seconds")}
    rows = []
    for mode in ("serial", "overlapped"):
        stats, losses = run_steps(mode, args.steps, args.slow_ms)
        tok_s = (4 * 8 * 64) / (stats["wall_ms"] / args.steps / 1e3) if args.steps else 0
        rows.append({
            "mode": mode,
            "wall_ms": round(stats["wall_ms"], 1),
            "gpu_ms": round(stats["gpu_ms"], 1),
            "idle_pct": round(stats["idle_pct"], 2),
            "tok_per_s": round(tok_s),
            "stages": stats["stages"],
        })
        print(f"  {mode:10s} wall={stats['wall_ms']:8.1f}ms  "
              f"gpu={stats['gpu_ms']:8.1f}ms  idle={stats['idle_pct']:5.1f}%  "
              f"{tok_s:6.0f} tok/s  (mean loss {sum(losses)/len(losses):.4f})")
        if args.stage and stats.get("stages"):
            parts = [f"{k}={v:.0f}ms" for k, v in stats["stages"].items()]
            print(f"              stages: {', '.join(parts)}")
    results["rows"] = rows

    speedup = rows[0]["wall_ms"] / max(rows[1]["wall_ms"], 1e-9)
    idle_delta = rows[0]["idle_pct"] - rows[1]["idle_pct"]
    print(f"\n  speedup: {speedup:.2f}x  |  GPU idle: "
          f"{rows[0]['idle_pct']:.1f}% → {rows[1]['idle_pct']:.1f}% "
          f"(removed {idle_delta:.1f}pp)")
    results["speedup"] = round(speedup, 3)
    results["idle_before_pct"] = rows[0]["idle_pct"]
    results["idle_after_pct"] = rows[1]["idle_pct"]

    # ── checkpoint stress (optional) ──────────────────────────────────────
    if args.checkpoint_stress:
        import tempfile as _tf
        with _tf.TemporaryDirectory() as _tmp:
            print("\n[checkpoint] per-step save overhead (none / sync / async):")
            ckpt_res = run_checkpoint_stress(args.steps, args.slow_ms, _tmp)
        results["checkpoint_stress"] = ckpt_res
        if ckpt_res["sync"] and ckpt_res["async"]:
            print(f"  async vs sync: {ckpt_res['sync']/max(ckpt_res['async'],1e-9):.2f}x "
                  f"faster wall ({(ckpt_res['sync']-ckpt_res['async'])/args.steps:.1f} ms/step saved)")

    # ── report ────────────────────────────────────────────────────────────
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = args.out or f"benchmark_pipeline_overlap_{stamp}"
    if not str(base).endswith(".json"):
        base += ".json"
    json_path = out_dir / base
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    md = json_path.with_suffix(".md")
    L = [f"# Pipeline Overlap Benchmark",
         f"- Date: {results['date']}", f"- Device: `{device}`",
         f"- Steps: {args.steps}", f"- Simulated slow I/O: {args.slow_ms} ms/sample",
         "", "| mode | wall (ms) | gpu busy (ms) | idle % | tok/s |",
         "|---|--:|--:|--:|--:|"]
    for row in rows:
        L.append(f"| {row['mode']} | {row['wall_ms']} | {row['gpu_ms']} | "
                 f"{row['idle_pct']} | {row['tok_per_s']} |")
    L.append("")
    L.append(f"**Speedup:** {speedup:.2f}x — GPU idle "
             f"{rows[0]['idle_pct']:.1f}% → {rows[1]['idle_pct']:.1f}% "
             f"(removed {idle_delta:.1f}pp).")
    if any(r.get("stages") for r in rows):
        L.append("")
        L.append("### Per-stage wall breakdown (total ms over run)")
        L.append("| stage | serial | overlapped |")
        L.append("|-------|-------:|-----------:|")
        stages = sorted(set().union(*(r["stages"].keys() for r in rows if r.get("stages"))))
        per = {r["mode"]: r.get("stages", {}) for r in rows}
        for st in stages:
            L.append(f"| {st} | {per.get('serial', {}).get(st, 0):.0f} | "
                     f"{per.get('overlapped', {}).get(st, 0):.0f} |")
        L.append("")
    if results.get("checkpoint_stress"):
        ck = results["checkpoint_stress"]
        L.append("### Checkpoint stress (per-step save overhead)")
        L.append("| variant | total (ms) | ms/step |")
        L.append("|---------|-----------:|--------:|")
        for mode in ("none", "sync", "async"):
            L.append(f"| {mode} | {ck[mode]:.1f} | {ck[mode]/max(args.steps,1):.2f} |")
        L.append("")
    md.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\nReport written:\n  JSON: {json_path}\n  Markdown: {md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
