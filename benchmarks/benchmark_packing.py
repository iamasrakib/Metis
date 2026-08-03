#!/usr/bin/env python3
"""
Μῆτις (Metis) — Dynamic Sequence Packing Benchmark
======================================================
Compares dynamic sequence packing against the unpadded baseline across three
metrics:

  efficiency   — padding-waste (%) for each batching strategy at a given
                 doc-length distribution.

  throughput   — full-model training step (fwd + bwd) wall-clock time,
                 effective real tokens/sec, and loss-equivalence sanity check.

  memory       — peak GPU allocation for packed vs. padded batches processing
                 the same real-token budget.

Results are written as JSON + Markdown report under ``benchmarks/results/``.

Usage:
    python benchmarks/benchmark_packing.py
    python benchmarks/benchmark_packing.py --mode efficiency
    python benchmarks/benchmark_packing.py --mode throughput --iters 20
    python benchmarks/benchmark_packing.py --mode memory
    python benchmarks/benchmark_packing.py --device cuda --out results/my.json
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from metis.config import ModelConfig  # noqa: E402
from metis.model import MetisLM  # noqa: E402
from metis.packing import (  # noqa: E402
    BIN,
    PackedBatch,
    PackedDataset,
    pack_documents,
)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).resolve().parent.parent),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "n/a"


def generate_docs(n, rng):
    """Variable-length documents (Pareto-ish distribution)."""
    lengths = np.maximum(2, (rng.pareto(1.5, size=n) * 5 + 2).astype(int))
    docs = [
        list(rng.randint(4, 100, size=int(L)))
        for L in lengths
    ]
    return docs


class Timer:
    """Median wall time over ``runs``; peak CUDA memory delta if on GPU."""

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
                start_evt, end_evt = torch.cuda.Event(True), torch.cuda.Event(True)
                start_evt.record()
                fn(*args, **kwargs)
                end_evt.record()
                torch.cuda.synchronize()
                samples.append(start_evt.elapsed_time(end_evt))
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


# ──────────────────────────────────────────────────────────────────────────────
# Packing efficiency
# ──────────────────────────────────────────────────────────────────────────────


def bench_efficiency(docs, seq_len, eos_id, pad_id, strategies):
    rows = []
    for strat in strategies:
        _, _, _, n_pad = pack_documents(
            docs, seq_len, eos_id=eos_id, pad_id=pad_id, strategy=strat,
        )
        n_seqs = len(docs)  # approximation
        total = len(docs) * seq_len
        waste = n_pad
        rows.append({
            "strategy": strat,
            "total_padded_tokens": total,
            "waste_tokens": waste,
            "waste_pct": round(100.0 * waste / total, 2) if total else 0,
        })
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# Throughput
# ──────────────────────────────────────────────────────────────────────────────


def bench_throughput(docs, seq_len, batch_size, device, iters, amp_dtype):
    cfg = ModelConfig(
        d_model=128, n_heads=4, n_layers=4, max_seq_len=seq_len,
        dropout=0.0, vocab_size=128, use_rope=True,
    )
    model = MetisLM(cfg)
    model.to(device)
    model.train()
    scaler = torch.amp.GradScaler(device, enabled=device.startswith("cuda"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    timer = Timer(device, runs=iters, warmup=2)

    PAD = 0
    EOS = 3

    # ── packed (stream) ──────────────────────────────────────────
    ds_pack = PackedDataset(
        docs, seq_len, batch_size, eos_id=EOS, pad_id=PAD,
        strategy="stream", shuffle=True, seed=0,
    )
    def step_packed():
        optimizer.zero_grad(set_to_none=True)
        batch = next(iter(ds_pack)).to(device)
        with torch.autocast(device_type=device.split(":")[0], dtype=amp_dtype, enabled=device.startswith("cuda")):
            _, loss, _ = model(
                batch.input_ids, batch.labels,
                attention_mask=batch.attention_mask,
                position_ids=batch.position_ids,
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        return loss.item()
    packed = timer.time(step_packed)
    packed["strategy"] = "packed_stream"
    packed["real_tokens_per_step"] = batch_size * seq_len
    packed["tokens_per_sec"] = round(
        packed["real_tokens_per_step"] / (packed["median_ms"] / 1e3), 0
    )
    packed["loss"] = step_packed()
    timer.release()

    # ── packed (bin) ─────────────────────────────────────────────
    ds_bin = PackedDataset(
        docs, seq_len, batch_size, eos_id=EOS, pad_id=PAD,
        strategy=BIN, shuffle=True, seed=0,
    )
    def step_bin():
        optimizer.zero_grad(set_to_none=True)
        batch = next(iter(ds_bin)).to(device)
        with torch.autocast(device_type=device.split(":")[0], dtype=amp_dtype, enabled=device.startswith("cuda")):
            _, loss, _ = model(
                batch.input_ids, batch.labels,
                attention_mask=batch.attention_mask,
                position_ids=batch.position_ids,
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        return loss.item()
    binned = timer.time(step_bin)
    binned["strategy"] = "packed_bin"
    binned["real_tokens_per_step"] = batch_size * seq_len
    binned["tokens_per_sec"] = round(
        binned["real_tokens_per_step"] / (binned["median_ms"] / 1e3), 0
    )
    binned["loss"] = step_bin()
    timer.release()

    # ── padded baseline ──────────────────────────────────────────
    rng = np.random.RandomState(99)
    padded_input = np.full((batch_size, seq_len), PAD, dtype=np.int64)
    padded_labels = np.full((batch_size, seq_len), PAD, dtype=np.int64)
    for r in range(batch_size):
        doc = list(rng.randint(4, 128, size=min(rng.randint(4, seq_len), seq_len)))
        padded_input[r, :len(doc)] = doc
        padded_labels[r, :len(doc)-1] = doc[1:]
        padded_labels[r, len(doc)-1] = EOS
    p_inputs = torch.from_numpy(padded_input).long().to(device)
    p_labels = torch.from_numpy(padded_labels).long().to(device)
    def step_padded():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.split(":")[0], dtype=amp_dtype, enabled=device.startswith("cuda")):
            _, loss, _ = model(p_inputs, p_labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        return loss.item()
    padded_t = timer.time(step_padded)
    padded_t["strategy"] = "padded_baseline"
    padded_t["real_tokens_per_step"] = int(padded_input[padded_input != PAD].size)
    padded_t["tokens_per_sec"] = round(
        padded_t["real_tokens_per_step"] / (padded_t["median_ms"] / 1e3), 0
    )
    padded_t["loss"] = step_padded()
    timer.release()

    return [packed, binned, padded_t]


# ──────────────────────────────────────────────────────────────────────────────
# Memory (GPU peak allocation for packed vs padded)
# ──────────────────────────────────────────────────────────────────────────────


def bench_memory(docs, seq_len, batch_size, device, iters):
    """Peak GPU memory: packed vs. padded, same real-token budget."""
    if not device.startswith("cuda"):
        return {"note": "GPU memory benchmark requires CUDA device"}

    cfg = ModelConfig(
        d_model=128, n_heads=4, n_layers=4, max_seq_len=seq_len,
        dropout=0.0, vocab_size=128, use_rope=True,
    )
    model = MetisLM(cfg)
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    EOS, PAD = 3, 0

    ds_pack = PackedDataset(
        docs, seq_len, batch_size, eos_id=EOS, pad_id=PAD,
        strategy="stream", shuffle=True, seed=0,
    )
    batch = next(iter(ds_pack)).to(device)

    # warmup
    torch.cuda.reset_peak_memory_stats()
    for _ in range(iters):
        optimizer.zero_grad(set_to_none=True)
        _, loss, _ = model(
            batch.input_ids, batch.labels,
            attention_mask=batch.attention_mask,
            position_ids=batch.position_ids,
        )
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    packed_mem = torch.cuda.max_memory_allocated() / 1e6
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()

    # padded
    rng = np.random.RandomState(99)
    p_input = np.full((batch_size, seq_len), PAD, dtype=np.int64)
    p_labels = np.full((batch_size, seq_len), PAD, dtype=np.int64)
    for r in range(batch_size):
        doc = list(rng.randint(4, 128, size=min(rng.randint(4, seq_len), seq_len)))
        p_input[r, :len(doc)] = doc
        p_labels[r, :len(doc)-1] = doc[1:]
        p_labels[r, len(doc)-1] = EOS
    p_in = torch.from_numpy(p_input).long().to(device)
    p_lbl = torch.from_numpy(p_labels).long().to(device)

    for _ in range(iters):
        optimizer.zero_grad(set_to_none=True)
        _, loss, _ = model(p_in, p_lbl)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    padded_mem = torch.cuda.max_memory_allocated() / 1e6
    optimizer.zero_grad(set_to_none=True)

    real_tok = int((p_input != PAD).sum())
    return {
        "packed_peak_MB": round(packed_mem, 1),
        "padded_peak_MB": round(padded_mem, 1),
        "real_tokens": real_tok,
        "saved_MB": round(padded_mem - packed_mem, 1),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────────────────────────


def write_report(results, out_path):
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    md = out.with_suffix(".md")
    L = [
        "# Μῆτις — Dynamic Sequence Packing Benchmark",
        "",
        f"- Date: {results['date']}",
        f"- Device: `{results['device']}`",
        f"- PyTorch: `{results['torch']}`",
        f"- Git SHA: `{results.get('git_sha', 'n/a')}`",
        "",
    ]
    if "efficiency" in results:
        L.append("## Efficiency (padding waste %)")
        L.append("")
        L.append("| strategy | total tokens | waste tokens | waste % |")
        L.append("|----------|-------------:|-------------:|--------:|")
        for row in results["efficiency"]:
            L.append(
                f"| {row['strategy']} | {row['total_padded_tokens']:,} | "
                f"{row['waste_tokens']:,} | {row['waste_pct']:.1f}% |"
            )
        L.append("")
    if "throughput" in results:
        L.append("## Throughput (train step fwd+bwd)")
        L.append("")
        L.append("| strategy | median (ms) | real tokens/s | loss | peak mem (MB) |")
        L.append("|----------|------------:|--------------:|-----:|--------------:|")
        for row in results["throughput"]:
            L.append(
                f"| {row['strategy']} | {row['median_ms']:.2f} | "
                f"{row.get('tokens_per_sec', 'n/a'):>10} | {row.get('loss', 0):.4f} | "
                f"{row.get('peak_mem_MB', 0):.1f} |"
            )
        L.append("")
    if "memory" in results and isinstance(results["memory"], dict) and "packed_peak_MB" in results["memory"]:
        m = results["memory"]
        L.append("## Memory (GPU peak allocation)")
        L.append("")
        L.append(f"- Packed: **{m['packed_peak_MB']:.1f} MB**")
        L.append(f"- Padded: **{m['padded_peak_MB']:.1f} MB**")
        L.append(f"- Saved:  **{m['saved_MB']:.1f} MB** ({m['real_tokens']} real tokens)")
        L.append("")
    md.write_text("\n".join(L), encoding="utf-8")
    return str(md)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def _configure_stdio():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def main(argv=None):
    ap = argparse.ArgumentParser(description="Dynamic sequence packing benchmark")
    ap.add_argument(
        "--mode",
        choices=["efficiency", "throughput", "memory", "all"],
        default="all",
    )
    ap.add_argument("--device", default=None)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--n-docs", type=int, default=200)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    _configure_stdio()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.RandomState(0)
    docs = generate_docs(args.n_docs, rng)
    EOS, PAD = 3, 0

    use_amp = device.startswith("cuda")
    amp_dtype = (
        torch.bfloat16
        if use_amp and torch.cuda.is_bf16_supported()
        else (torch.float16 if use_amp else torch.float32)
    )

    results = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "device": device,
        "torch": torch.__version__,
        "git_sha": git_sha(),
        "n_docs": args.n_docs,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
    }
    print(
        f"Metis packing benchmark — device={device}, "
        f"docs={args.n_docs}, seq_len={args.seq_len}, "
        f"batch={args.batch_size}"
    )

    if args.mode in ("efficiency", "all"):
        print("\n[efficiency] padding waste across strategies:")
        eff = bench_efficiency(
            docs, args.seq_len, EOS, PAD,
            strategies=["stream", "bin"],
        )
        for row in eff:
            print(f"  {row['strategy']:20s}: waste {row['waste_pct']:.1f}% ({row['waste_tokens']:,} tokens)")
        results["efficiency"] = eff

    if args.mode in ("throughput", "all"):
        print("\n[throughput] model train step (fwd+bwd):")
        tp = bench_throughput(
            docs, args.seq_len, args.batch_size, device, args.iters, amp_dtype,
        )
        for row in tp:
            print(
                f"  {row['strategy']:20s}: {row['median_ms']:8.2f} ms  "
                f"tokens/s={row.get('tokens_per_sec', 'n/a'):>10}  "
                f"loss={row.get('loss', 0):.4f}"
            )
        results["throughput"] = tp

    if args.mode in ("memory", "all"):
        print("\n[memory] GPU peak allocation:")
        mem = bench_memory(
            docs, args.seq_len, args.batch_size, device, args.iters,
        )
        if "packed_peak_MB" in mem:
            print(f"  packed: {mem['packed_peak_MB']:.1f} MB")
            print(f"  padded: {mem['padded_peak_MB']:.1f} MB")
            print(f"  saved:  {mem['saved_MB']:.1f} MB")
        else:
            print(f"  {mem.get('note', 'skipped')}")
        results["memory"] = mem

    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results",
        f"benchmark_packing_{datetime.now():%Y%m%d_%H%M%S}.json",
    )
    md = write_report(results, out)
    print(f"\nResults written to:\n  {out}\n  {md}")


if __name__ == "__main__":
    raise SystemExit(main())
