#!/usr/bin/env python3
"""
Μῆτις (Metis) — FlashAttention benchmark
==========================================
Compares the OLD attention path (exact manual math / dead-cache re-prefix)
against the NEW FlashAttention dispatch path, at two levels:

  kernel — raw attention kernels on synthetic Q/K/V (prefill + decode):
      old = math_attention (manual masked-softmax)
      new = causal_attention (flash-attn → SDPA → math dispatch)

  model — full MetisLM steps:
      train  = forward + backward under AMP (bf16/fp16), with and without
               gradient checkpointing
      decode = one autoregressive token step; OLD simulates the legacy
               dead-KV-cache behavior (re-processes the whole prefix each
               step), NEW uses the now-working KV cache (1 token in, O(1)
               attention per step)

Every measurement records wall time and, on CUDA, peak GPU memory
(``torch.cuda.max_memory_allocated``). Results are written as JSON plus a
Markdown report under ``benchmarks/results/``.

Usage:
    python benchmarks/benchmark_attention.py                     # kernel+model+memory, auto device
    python benchmarks/benchmark_attention.py --mode kernel       # kernels only
    python benchmarks/benchmark_attention.py --mode model        # model steps only
    python benchmarks/benchmark_attention.py --mode memory       # peak activation memory vs seq len
    python benchmarks/benchmark_attention.py --backend mem_efficient --iters 20
    python benchmarks/benchmark_attention.py --device cpu --mode model
    python benchmarks/benchmark_attention.py --out results/my_run.json
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

from metis.attn import (  # noqa: E402
    MATH,
    causal_attention,
    detect_attention_backends,
    math_attention,
    set_backend_flags,
)
from metis.config import ModelConfig  # noqa: E402
from metis.model import MetisLM  # noqa: E402

# ── Timing / memory helpers ───────────────────────────────────────────────────

class Timer:
    """Median wall time over ``runs``; peak CUDA memory delta if on GPU."""

    def __init__(self, device: str, runs: int = 10, warmup: int = 3):
        self.device = device
        self.runs = runs
        self.warmup = warmup
        self.is_cuda = device.startswith("cuda")

    def time(self, fn, *args, **kwargs) -> dict:
        # Warmup
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


# ── Kernel-level ───────────────────────────────────────────────────────────────

def _repeat_kv(x, n_groups):
    if n_groups == 1:
        return x
    B, n_kv, T, D = x.size()
    x = x[:, :, None, :, :].expand(B, n_kv, n_groups, T, D)
    return x.reshape(B, n_kv * n_groups, T, D)


def bench_kernels(device: str, backend: str, iters: int) -> list:
    """Old (math) vs new (dispatch) raw-attention timings + memory."""
    timer = Timer(device, runs=iters, warmup=3)
    rows = []

    def run_pair(label, kind, B, T_q, T_k, H, H_kv, dtype, scale):
        D = 64
        q = torch.randn(B, H, T_q, D, device=device, dtype=dtype)
        k = torch.randn(B, H_kv, T_k, D, device=device, dtype=dtype)
        v = torch.randn(B, H_kv, T_k, D, device=device, dtype=dtype)
        groups = H // H_kv

        # OLD: exact manual math on pre-expanded KV
        ke, ve = _repeat_kv(k, groups), _repeat_kv(v, groups)
        old = timer.time(lambda: math_attention(q, ke, ve, scale=scale))

        # NEW: dispatched attention (auto or forced backend)
        bl = []
        def _new():
            bl.clear()
            return causal_attention(
                q, k, v, n_heads=H, n_kv_heads=H_kv, scale=scale,
                backend=backend, use_flash_attn=True, out_backend=bl,
            )
        new = timer.time(_new)
        used = bl[0] if bl else "?"
        rows.append({
            "kind": kind, "label": label, "B": B, "T_q": T_q, "T_k": T_k,
            "H": H, "H_kv": H_kv, "dtype": str(dtype),
            "old_backend": MATH, "new_backend": used,
            "old_median_ms": old["median_ms"], "old_peak_mem_MB": old["peak_mem_MB"],
            "new_median_ms": new["median_ms"], "new_peak_mem_MB": new["peak_mem_MB"],
            "speedup": old["median_ms"] / max(new["median_ms"], 1e-9),
            "mem_saving_MB": old["peak_mem_MB"] - new["peak_mem_MB"],
        })
        print(f"  {label:46s} old={old['median_ms']:8.2f}ms "
              f"new={new['median_ms']:8.2f}ms ({used:20s}) "
              f"speedup={rows[-1]['speedup']:6.2f}x")
        timer.release()
        return rows[-1]

    if device.startswith("cuda"):
        dtype = torch.float16
        for T in (128, 256, 512):
            for B in (1, 4):
                run_pair(f"prefill MHA  B={B} T={T}", "prefill", B, T, T, 8, 8, dtype, None)
                run_pair(f"prefill GQA  B={B} T={T}", "prefill", B, T, T, 8, 4, dtype, None)
        for T_k in (64, 128, 256, 512):
            run_pair(f"decode  MHA  T_k={T_k}", "decode", 1, 1, T_k, 8, 8, dtype, None)
            run_pair(f"decode  GQA  T_k={T_k}", "decode", 1, 1, T_k, 8, 4, dtype, None)
    else:
        # CPU: fp32 math only — informational
        dtype = torch.float32
        for T in (128, 256):
            run_pair(f"prefill MHA CPU B=1 T={T}", "prefill", 1, T, T, 8, 8, dtype, None)
        for T_k in (64, 128, 256):
            run_pair(f"decode  MHA CPU T_k={T_k}", "decode", 1, 1, T_k, 8, 8, dtype, None)
    return rows


# ── Model-level ───────────────────────────────────────────────────────────────

def _model(device: str, use_flash: bool, backend: str) -> MetisLM:
    cfg = ModelConfig(
        d_model=128, n_heads=4, n_kv_heads=2, n_layers=4, max_seq_len=512,
        vocab_size=512, dropout=0.0, use_flash_attn=use_flash,
        attn_backend=backend if use_flash else "math",
    )
    m = MetisLM(cfg)
    m.to(device)
    return m


def bench_train_steps(device: str, backend: str, iters: int) -> list:
    """Old (use_flash_attn=False → math) vs new (fused) train step."""
    timer = Timer(device, runs=iters, warmup=1)
    amp_dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    rows = []
    for use_flash in (False, True):
        for ckpt in (False, True):
            m = _model(device, use_flash, backend)
            m.train()
            opt = m.configure_optimizers(0.1, 1e-3, device)
            idx = torch.randint(0, 512, (2, 256), device=device)
            scaler = torch.amp.GradScaler(
                "cuda", enabled=device.startswith("cuda")
            )

            def step():
                opt.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.split(":")[0], dtype=amp_dtype,
                    enabled=device.startswith("cuda"),
                ):
                    _, loss, _ = m(idx, targets=idx, use_checkpointing=ckpt)
                if device.startswith("cuda"):
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                    opt.step()
                return loss.item()

            r = timer.time(step)
            used = m.layers[0].attn.last_backend
            rows.append({
                "op": "train", "use_flash_attn": use_flash,
                "grad_checkpointing": ckpt, "backend_used": used,
                "median_ms": r["median_ms"], "mean_ms": r["mean_ms"],
                "peak_mem_MB": r["peak_mem_MB"],
                "tokens_per_s": 2 * 256 / (r["median_ms"] / 1e3),
            })
            print(f"  train use_flash={use_flash} ckpt={ckpt}: "
                  f"{r['median_ms']:8.2f}ms  peak={r['peak_mem_MB']:7.1f}MB  "
                  f"({used})")
            timer.release()
    return rows


def bench_memory(device: str, backend: str) -> list:
    """Peak activation memory of one train step vs sequence length.

    OLD (math) vs NEW (fused dispatch) at a FIXED configuration (no gradient
    checkpointing), isolating the attention kernel's memory contribution:
    the manual path materializes the (B, H, T, T) scores matrix — O(T²)
    activations that must be retained for backward — while the fused kernels
    keep attention activations O(T) (online softmax, recomputed in backward).
    """
    if not device.startswith("cuda"):
        print("  (memory comparison requires CUDA — skipped)")
        return []
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    rows = []
    for T in (128, 256, 512, 1024, 2048):
        for use_flash in (False, True):
            cfg = ModelConfig(
                d_model=128, n_heads=4, n_kv_heads=2, n_layers=4,
                max_seq_len=2048, vocab_size=512, dropout=0.0,
                use_flash_attn=use_flash,
                attn_backend=backend if use_flash else "math",
            )
            m = MetisLM(cfg).to(device)
            m.train()
            opt = m.configure_optimizers(0.1, 1e-3, device)
            scaler = torch.amp.GradScaler("cuda", enabled=True)
            idx = torch.randint(0, 512, (1, T), device=device)

            # Warmup pass (allocates cuDNN/allocator caches, lazily-created state)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                _, loss, _ = m(idx, targets=idx, use_checkpointing=False)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

            # Measured pass: forward + backward, no optimizer step
            torch.cuda.reset_peak_memory_stats()
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                _, loss, _ = m(idx, targets=idx, use_checkpointing=False)
            scaler.scale(loss).backward()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / 1e6
            used = m.layers[0].attn.last_backend
            rows.append({
                "T": T, "use_flash_attn": use_flash, "backend_used": used,
                "peak_mem_MB": peak,
            })
            print(f"  T={T:5d} use_flash={use_flash}: peak={peak:8.1f}MB  ({used})")
            torch.cuda.empty_cache()
            del m, opt, scaler

    # Pair old/new per T and compute savings
    by_t: dict[int, dict] = {}
    for r in rows:
        by_t.setdefault(r["T"], {})[r["use_flash_attn"]] = r
    out = []
    for T in sorted(by_t):
        old, new = by_t[T][False], by_t[T][True]
        saving = old["peak_mem_MB"] - new["peak_mem_MB"]
        out.append({
            "T": T, "backend_used": new["backend_used"],
            "old_peak_mem_MB": old["peak_mem_MB"],
            "new_peak_mem_MB": new["peak_mem_MB"],
            "mem_saving_MB": saving,
            "mem_saving_pct": 100.0 * saving / max(old["peak_mem_MB"], 1e-9),
        })
    return out


def bench_decode(device: str, backend: str, iters: int, n_tokens: int = 200) -> list:
    """OLD: legacy re-prefix decode (dead cache) vs NEW: cached decode.

    Both use the SAME model weights; only the decode strategy differs.
    """
    amp_dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    rows = []

    def _time_strategy(m, strategy: str, prefix: int) -> dict:
        # Prefill: warm the cache / establish the prefix
        idx = torch.randint(0, 512, (1, prefix + 1), device=device)
        cache = None
        with torch.no_grad(), torch.autocast(
            device_type=device.split(":")[0], dtype=amp_dtype,
            enabled=device.startswith("cuda"),
        ):
            if strategy == "cached":
                _, _, cache = m(idx[:, :prefix], kv_cache=None)
                def dec():
                    with torch.no_grad(), torch.autocast(
                        device_type=device.split(":")[0], dtype=amp_dtype,
                        enabled=device.startswith("cuda"),
                    ):
                        logits, _, c2 = m(idx[:, -1:], kv_cache=cache)
                    return logits
            else:  # "reprefix"
                def dec():
                    with torch.no_grad(), torch.autocast(
                        device_type=device.split(":")[0], dtype=amp_dtype,
                        enabled=device.startswith("cuda"),
                    ):
                        logits, _, _ = m(idx, targets=None)
                    return logits

        # Measure one decode step
        timer = Timer(device, runs=iters, warmup=2)
        r = timer.time(dec)
        return r

    for use_flash in (False, True):
        m = _model(device, use_flash, backend)
        m.eval()
        for prefix in (64, 128, 256):
            old = _time_strategy(m, "reprefix", prefix)
            new = _time_strategy(m, "cached", prefix)
            used = m.layers[0].attn.last_backend
            rows.append({
                "op": "decode", "use_flash_attn": use_flash,
                "prefix_len": prefix, "backend_used": used,
                "old_reprefix_ms": old["median_ms"],
                "new_cached_ms": new["median_ms"],
                "old_peak_mem_MB": old["peak_mem_MB"],
                "new_peak_mem_MB": new["peak_mem_MB"],
                "speedup": old["median_ms"] / max(new["median_ms"], 1e-9),
            })
            print(f"  decode use_flash={use_flash} prefix={prefix}: "
                  f"reprefix={old['median_ms']:8.2f}ms "
                  f"cached={new['median_ms']:8.2f}ms "
                  f"speedup={rows[-1]['speedup']:6.2f}x  ({used})")
    return rows


# ── Report ────────────────────────────────────────────────────────────────────

def _fmt(ms: float) -> str:
    return f"{ms:.2f} ms" if ms >= 0.01 else f"{ms * 1e3:.2f} us"


def write_report(results: dict, out_path: str) -> str:
    """Write JSON + a Markdown report; return the Markdown path."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    md = out.with_suffix(".md")
    L = []
    L.append("# Metis — FlashAttention benchmark report")
    L.append("")
    L.append(f"- Date: {results['date']}")
    L.append(f"- Device: `{results['device']}`")
    L.append(f"- PyTorch: `{results['torch']}`")
    L.append(f"- Git SHA: `{results.get('git_sha', 'n/a')}`")
    L.append("")
    L.append("## Backend capabilities")
    b = results["backends"]
    for k, v in b.items():
        L.append(f"- {k}: {v}")
    L.append("")

    if results.get("kernel"):
        L.append("## Kernel-level: old manual math vs new dispatch")
        L.append("")
        L.append("| kind | label | old (ms) | new (ms) | backend | speedup "
                 "| old mem (MB) | new mem (MB) |")
        L.append("|------|-------|---------:|---------:|---------|--------:|-------------:|-------------:|")
        for r in results["kernel"]:
            L.append(
                f"| {r['kind']} | {r['label']} | {r['old_median_ms']:.3f} | "
                f"{r['new_median_ms']:.3f} | {r['new_backend']} | "
                f"{r['speedup']:.2f}x | {r['old_peak_mem_MB']:.1f} | "
                f"{r['new_peak_mem_MB']:.1f} |"
            )
        L.append("")

    if results.get("train"):
        L.append("## Model-level: training step (forward + backward, AMP)")
        L.append("")
        L.append("| use_flash_attn | grad ckpt | backend | time (ms) | peak mem (MB) | tok/s |")
        L.append("|----------------|-----------|---------|----------:|--------------:|------:|")
        for r in results["train"]:
            L.append(
                f"| {r['use_flash_attn']} | {r['grad_checkpointing']} | "
                f"{r['backend_used']} | {r['median_ms']:.2f} | "
                f"{r['peak_mem_MB']:.1f} | {r['tokens_per_s']:.0f} |"
            )
        L.append("")

    if results.get("decode"):
        L.append("## Model-level: one decode step (T_q=1)")
        L.append("")
        L.append("OLD = legacy dead-KV-cache behavior (re-prefix whole context per step); "
                 "NEW = working KV cache (1 token in).")
        L.append("")
        L.append("| use_flash_attn | prefix len | backend | OLD reprefix (ms) "
                 "| NEW cached (ms) | speedup |")
        L.append("|----------------|-----------:|---------|------------------:|----------------:|--------:|")
        for r in results["decode"]:
            L.append(
                f"| {r['use_flash_attn']} | {r['prefix_len']} | "
                f"{r['backend_used']} | {r['old_reprefix_ms']:.3f} | "
                f"{r['new_cached_ms']:.3f} | {r['speedup']:.2f}x |"
            )
        L.append("")

    if results.get("memory"):
        L.append("## Memory comparison: peak activation memory "
                 "(train step, no gradient checkpointing)")
        L.append("")
        L.append("| T | OLD math peak (MB) | NEW fused peak (MB) "
                 "| saving (MB) | saving % | backend |")
        L.append("|---|-------------------:|--------------------:|------------:|---------:|---------|")
        for r in results["memory"]:
            L.append(
                f"| {r['T']} | {r['old_peak_mem_MB']:.1f} | {r['new_peak_mem_MB']:.1f} | "
                f"{r['mem_saving_MB']:.1f} | {r['mem_saving_pct']:.1f}% | "
                f"{r['backend_used']} |"
            )
        L.append("")

    md.write_text("\n".join(L), encoding="utf-8")
    return str(md)


def _configure_stdio() -> None:
    """Make console output encoding-safe on Windows (cp1252 vs UTF-8).

    The repository path can contain non-ASCII characters (e.g. the Greek name
    Μῆτις); printing an absolute path under cp1252 then raises
    ``UnicodeEncodeError``. Reconfigure stdout/stderr to UTF-8 so reporting is
    never console-codepage-dependent.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass  # stream has no reconfigure (e.g. embedded); leave as-is


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=os.path.dirname(__file__),
        )
        return out.stdout.strip() or "n/a"
    except Exception:
        return "n/a"


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark old vs new Metis attention (FlashAttention dispatch)."
    )
    parser.add_argument("--mode", choices=["kernel", "model", "memory", "both"], default="both")
    parser.add_argument("--device", default=None, help="cpu | cuda (default: auto)")
    parser.add_argument("--backend", default="auto",
                        choices=["auto", "math", "mem_efficient", "flash", "flash_attn"],
                        help="Backend for the NEW path (default: auto)")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--out", default=None, help="Output JSON path")
    args = parser.parse_args(argv)

    _configure_stdio()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.backend != "auto":
        set_backend_flags(args.backend)

    results = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "device": device,
        "torch": torch.__version__,
        "git_sha": git_sha(),
        "backend_requested": args.backend,
        "backends": detect_attention_backends(),
        "kernel": [], "train": [], "decode": [], "memory": [],
    }

    print(f"Metis attention benchmark — device={device} backend={args.backend}")
    print(f"Capabilities: {results['backends']}\n")

    if args.mode in ("kernel", "both"):
        print("[kernel-level] old manual math vs new dispatch (fp16 on CUDA):")
        results["kernel"] = bench_kernels(device, args.backend, args.iters)
    if args.mode in ("model", "both"):
        print("\n[model-level] training step (AMP, with/without gradient checkpointing):")
        results["train"] = bench_train_steps(device, args.backend, args.iters)
        print("\n[model-level] decode step: legacy re-prefix vs working KV cache:")
        results["decode"] = bench_decode(device, args.backend, args.iters)
    if args.mode in ("memory", "both"):
        print("\n[memory] peak activation memory vs sequence length (no checkpointing):")
        results["memory"] = bench_memory(device, args.backend)

    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results", f"benchmark_{datetime.now():%Y%m%d_%H%M%S}.json",
    )
    md = write_report(results, out)
    print("\nReport written:")
    print(f"  JSON: {out}")
    print(f"  Markdown: {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
