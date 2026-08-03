#!/usr/bin/env python3
"""
Metis — KV-cache subsystem benchmark
=====================================
Compares the legacy growable cache (``"default"``) against the new optional
backends (``"static"``, ``"quantized"``, ``"mla"``) across three dimensions:

  memory    — per-layer KV-cache bytes at each context length T
              (analytic formula + actual tensor storage)

  throughput — decode speed (median ms per step, tokens/second) and prefill
               time for each backend at various cache lengths

  parity    — logit error of quantized and MLA paths vs the default baseline

All benchmarks run on CPU (auto-detected), are device-agnostic, and write a
JSON + Markdown report to ``benchmarks/results/``.

Usage:
    python benchmarks/benchmark_kv.py
    python benchmarks/benchmark_kv.py --mode memory
    python benchmarks/benchmark_kv.py --mode throughput
    python benchmarks/benchmark_kv.py --preset small
    python benchmarks/benchmark_kv.py --device cpu
"""

from __future__ import annotations

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
from metis.kv import (  # noqa: E402
    cache_memory_bytes,
)
from metis.model import MetisLM  # noqa: E402

# ── Helpers ──────────────────────────────────────────────────────────────────

def _configure_stdio():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def _cfg(**ov) -> ModelConfig:
    defaults = dict(
        d_model=128, n_heads=4, n_kv_heads=2, n_layers=4,
        max_seq_len=512, vocab_size=512, dropout=0.0,
        use_rmsnorm=True, use_swiglu=True, use_rope=True,
        tie_weights=True, use_moe=False, use_qk_norm=False,
        use_attention_sink=False, use_flash_attn=False,
    )
    defaults.update(ov)
    return ModelConfig(**defaults)


def _seeded_model(cfg, seed=42):
    torch.manual_seed(seed)
    m = MetisLM(cfg)
    m.eval()
    return m


def _share_weights(target, source):
    target.load_state_dict({k: v.clone() for k, v in source.state_dict().items()})


def _generate_n(model, idx, n_steps):
    """Run n_steps autoregressive decode steps, return last logit per step."""
    logits, _, cache = model(idx)
    logits_list = [logits[:, -1:, :]]
    for _ in range(n_steps):
        inp = idx[:, -1:]  # only the last token
        logits, _, cache = model(inp, kv_cache=cache)
        logits_list.append(logits[:, -1:, :])
    return torch.cat(logits_list, dim=1), cache


def _time_fn(fn, runs=20, warmup=3):
    """Median wall time over runs (CPU)."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e3)
    samples.sort()
    return samples[len(samples) // 2]


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


# ── Memory benchmark ─────────────────────────────────────────────────────────

def bench_memory(preset: str) -> list:
    """Analytic per-layer cache bytes at each T for every backend."""
    cfg = _cfg(**({"tiny": dict(d_model=128, n_heads=4, n_kv_heads=2, n_layers=4),
                   "small": dict(d_model=256, n_heads=4, n_kv_heads=2, n_layers=4),
                   "medium": dict(d_model=384, n_heads=6, n_kv_heads=2, n_layers=6),
                   "large": dict(d_model=512, n_heads=8, n_kv_heads=2, n_layers=8),
                   }.get(preset, {})))
    dtype = torch.float32
    B = 1
    n_kv = cfg.n_kv_heads
    head_dim = cfg.head_dim
    c_d = cfg.mla_kv_latent_dim or (cfg.d_model // cfg.n_heads)
    rope_d = cfg.mla_rope_head_dim or (head_dim // 2)
    nH = cfg.n_heads

    rows = []
    for T in (32, 64, 128, 256, 512):
        kw = dict(B=B, n_kv_heads=n_kv, head_dim=head_dim, T=T,
                  max_seq_len=cfg.max_seq_len, dtype=dtype)
        d_bytes = cache_memory_bytes("default", **kw)
        s_bytes = cache_memory_bytes("static", **kw)
        q_bytes = cache_memory_bytes("quantized", **kw)
        m_bytes = cache_memory_bytes("mla", T=T, max_seq_len=cfg.max_seq_len,
                                     mla_kv_latent_dim=c_d, mla_rope_head_dim=rope_d,
                                     n_heads=nH, dtype=dtype,
                                     B=B, n_kv_heads=n_kv, head_dim=head_dim)
        rows.append({
            "T": T, "d_model": cfg.d_model, "n_heads": nH, "n_kv_heads": n_kv,
            "head_dim": head_dim, "max_seq_len": cfg.max_seq_len,
            "mla_c_d": c_d, "mla_rope_d": rope_d,
            "default_bytes": d_bytes, "static_bytes": s_bytes,
            "quantized_bytes": q_bytes, "mla_bytes": m_bytes,
            "default_KB": d_bytes / 1024, "static_KB": s_bytes / 1024,
            "quantized_KB": q_bytes / 1024, "mla_KB": m_bytes / 1024,
            "static_vs_default": d_bytes / max(s_bytes, 1),
            "quantized_vs_default": d_bytes / max(q_bytes, 1),
            "mla_vs_default": d_bytes / max(m_bytes, 1),
        })
        print(f"  T={T:4d}  default={d_bytes/1024:7.1f}KB  "
              f"static={s_bytes/1024:7.1f}KB  "
              f"quantized={q_bytes/1024:7.1f}KB  "
              f"mla={m_bytes/1024:7.1f}KB  "
              f"(q/d={d_bytes/max(q_bytes,1):.1f}x  mla/d={d_bytes/max(m_bytes,1):.1f}x)")
    return rows


# ── Throughput benchmark ─────────────────────────────────────────────────────

def bench_throughput(preset: str, iters: int = 15) -> list:
    """Decode ms/step and prefill ms for each backend at various prefix lengths."""
    cfg_base = _cfg(**({
        "tiny": dict(d_model=128, n_heads=4, n_kv_heads=2, n_layers=4, max_seq_len=256),
        "small": dict(d_model=256, n_heads=4, n_kv_heads=2, n_layers=4, max_seq_len=512),
    }.get(preset, {})))

    # Create a reference model for weight sharing (default/static/quantized only)
    model_ref = _seeded_model(cfg_base)

    rows = []
    for T_prefix in (32, 64, 128, 256):
        for backend in ("default", "static", "quantized", "mla"):
            try:
                overrides = {k: v for k, v in vars(cfg_base).items()
                             if k in ModelConfig.__dataclass_fields__
                             and ModelConfig.__dataclass_fields__[k].init
                             and k != "kv_backend" and not k.startswith("_")}
                overrides["kv_backend"] = backend
                # MLA requires n_kv_heads == n_heads (shared latent, incompatible
                # with GQA weights) and is an architecture change with its own
                # random init — no weight sharing with the reference model.
                if backend == "mla":
                    overrides["n_kv_heads"] = overrides["n_heads"]
                cfg = _cfg(**overrides)
                model = _seeded_model(cfg, seed=123)
                if backend != "mla":
                    _share_weights(model, model_ref)
            except Exception as e:
                print(f"  skip {backend} T={T_prefix}: {e}")
                continue

            idx = torch.randint(0, cfg.vocab_size, (1, T_prefix))
            n_decode = 20

            # Prefill time
            def prefill():
                with torch.no_grad():
                    model(idx)

            prefill_ms = _time_fn(prefill, runs=iters, warmup=3)

            # Decode time
            with torch.no_grad():
                _, cache = _generate_n(model, idx, n_decode)

            def decode_step():
                with torch.no_grad():
                    model(idx[:, -1:], kv_cache=cache)

            decode_ms = _time_fn(decode_step, runs=iters, warmup=2)
            tok_s = 1000.0 / decode_ms

            rows.append({
                "backend": backend, "T_prefix": T_prefix, "n_decode": n_decode,
                "prefill_ms": round(prefill_ms, 3),
                "decode_ms": round(decode_ms, 3),
                "tokens_per_sec": round(tok_s, 1),
                "d_model": cfg.d_model, "n_heads": cfg.n_heads,
                "n_kv_heads": cfg.n_kv_heads,
            })
            print(f"  {backend:10s} T={T_prefix:4d}  prefill={prefill_ms:8.2f}ms  "
                  f"decode={decode_ms:8.3f}ms  tok/s={tok_s:8.1f}")

        print()
    return rows


# ── Parity summary ──────────────────────────────────────────────────────────

def bench_parity(preset: str) -> list:
    """Quantized vs default (same weights) and MLA absorbed vs explicit."""
    cfg_base = _cfg(**({
        "tiny": dict(d_model=128, n_heads=4, n_kv_heads=2, n_layers=4, max_seq_len=256),
        "small": dict(d_model=256, n_heads=4, n_kv_heads=2, n_layers=4, max_seq_len=512),
    }.get(preset, {})))
    model_ref = _seeded_model(cfg_base)

    rows = []
    # Quantized: same weights as default
    for backend in ("quantized",):
        overrides = {k: v for k, v in vars(cfg_base).items()
                     if k in ModelConfig.__dataclass_fields__
                     and ModelConfig.__dataclass_fields__[k].init
                     and k != "kv_backend" and not k.startswith("_")}
        overrides["kv_backend"] = backend
        cfg = _cfg(**overrides)
        model = _seeded_model(cfg, seed=123)
        _share_weights(model, model_ref)

        idx = torch.randint(0, cfg.vocab_size, (1, 64))
        with torch.no_grad():
            ref_logits, _, ref_cache = model_ref(idx)
            ref_logits2, _, _ = model_ref(idx[:, -1:], kv_cache=ref_cache)

            new_logits, _, new_cache = model(idx)
            new_logits2, _, _ = model(idx[:, -1:], kv_cache=new_cache)

        max_diff = (ref_logits - new_logits).abs().max().item()
        max_diff2 = (ref_logits2 - new_logits2).abs().max().item()
        rows.append({
            "backend": backend, "max_diff_prefill": round(max_diff, 6),
            "max_diff_decode": round(max_diff2, 6),
        })
        print(f"  {backend:10s} prefill_max_diff={max_diff:.6f}  decode_max_diff={max_diff2:.6f}")

    # MLA: absorbed vs explicit (same model, both paths)
    mla_cfg = _cfg(kv_backend="mla", n_kv_heads=4)
    model_mla = _seeded_model(mla_cfg, seed=42)
    idx = torch.randint(0, mla_cfg.vocab_size, (1, 32))
    with torch.no_grad():
        # Explicit path: full prefill with 33 tokens
        idx33 = torch.cat([idx, idx[:, -1:]], dim=1)
        l_full, _, _ = model_mla(idx33)
        l_ref = l_full[:, -1, :]  # last token's logits

        # Absorbed path: prefill 32, decode 1 more
        l_pf, _, cache_pf = model_mla(idx)
        l_dec, _, _ = model_mla(idx[:, -1:], kv_cache=cache_pf)
        l_cached = l_dec[:, -1, :]

    max_diff = (l_ref - l_cached).abs().max().item()
    rows.append({
        "backend": "mla", "max_diff_prefill": "N/A (different arch)",
        "max_diff_decode": round(max_diff, 6),
    })
    print(f"  {'mla':10s} absorbed vs explicit: max_diff={max_diff:.6f}")
    return rows


# ── Report ───────────────────────────────────────────────────────────────────

def write_report(results: dict, out_path: str) -> str:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    md = out.with_suffix(".md")
    L = [
        "# Metis — KV-cache subsystem benchmark report",
        "",
        f"- Date: {results['date']}",
        f"- Device: `{results['device']}`",
        f"- PyTorch: `{results['torch']}`",
        f"- Preset: `{results['preset']}`",
        f"- Git SHA: `{results.get('git_sha', 'n/a')}`",
        "",
    ]

    if results.get("memory"):
        L.append("## Memory comparison (per-layer KV-cache bytes)")
        L.append("")
        L.append("| T | default (B) | static (B) | quantized (B) | mla (B) "
                 "| quant/default | mla/default |")
        L.append("|---|------------:|-----------:|--------------:|--------:|"
                 "---------------:|------------:|")
        for r in results["memory"]:
            L.append(
                f"| {r['T']} | {r['default_KB']:.1f}K | {r['static_KB']:.1f}K | "
                f"{r['quantized_KB']:.1f}K | {r['mla_KB']:.1f}K | "
                f"{r['quantized_vs_default']:.2f}x | {r['mla_vs_default']:.2f}x |"
            )
        L.append("")

    if results.get("throughput"):
        L.append("## Throughput comparison (CPU decode ms/step)")
        L.append("")
        L.append("| backend | T prefix | prefill (ms) | decode (ms) "
                 "| tokens/s |")
        L.append("|---------|--------:|--------------:|-----------:|"
                 "----------:|")
        for r in results["throughput"]:
            L.append(
                f"| {r['backend']} | {r['T_prefix']} | {r['prefill_ms']:.2f} | "
                f"{r['decode_ms']:.3f} | {r['tokens_per_sec']:.1f} |"
            )
        L.append("")

    if results.get("parity"):
        L.append("## Parity: logit error vs default baseline")
        L.append("")
        L.append("| backend | prefill max diff | decode max diff |")
        L.append("|---------|----------------:|----------------:|")
        for r in results["parity"]:
            pd = (f"{r['max_diff_prefill']:.6f}" if isinstance(
                r['max_diff_prefill'], (int, float))
                else str(r['max_diff_prefill']))
            L.append(
                f"| {r['backend']} | {pd} | "
                f"{r['max_diff_decode']:.6f} |"
            )
        L.append("")

    md.write_text("\n".join(L), encoding="utf-8")
    return str(md)


# ── Main ─────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark Metis KV-cache subsystem: memory, throughput, parity."
    )
    parser.add_argument("--mode", choices=["memory", "throughput", "parity", "all"],
                        default="all")
    parser.add_argument("--preset", choices=["tiny", "small", "medium", "large"],
                        default="small")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--iters", type=int, default=15,
                        help="Throughput: timing iterations")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    _configure_stdio()

    results = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "device": args.device,
        "torch": torch.__version__,
        "preset": args.preset,
        "git_sha": git_sha(),
        "memory": [], "throughput": [], "parity": [],
    }

    print(f"Metis KV-cache benchmark — device={args.device} preset={args.preset}\n")

    if args.mode in ("memory", "all"):
        print("[memory] per-layer cache bytes vs context length:")
        results["memory"] = bench_memory(args.preset)
    if args.mode in ("throughput", "all"):
        print("\n[throughput] decode ms/step and prefill time:")
        results["throughput"] = bench_throughput(args.preset, args.iters)
    if args.mode in ("parity", "all"):
        print("\n[parity] quantized/MLA logit error vs default:")
        results["parity"] = bench_parity(args.preset)

    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results", f"benchmark_kv_{datetime.now():%Y%m%d_%H%M%S}.json",
    )
    md = write_report(results, out)
    print(f"\nReport written:\n  JSON: {out}\n  Markdown: {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
