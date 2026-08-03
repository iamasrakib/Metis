#!/usr/bin/env python3
"""
Metis — fused-block benchmark (kernel count / bandwidth / throughput)
=====================================================================
A/B compares the FUSED transformer block against a faithful pre-fusion
reference (separate q/k/v and w1/w3 projections, manual RMSNorm) across
three metrics:

  kernel_count   — number of CUDA kernel-launching ops per block step
                   (profiler op-count excluding metadata-only ops on builds
                   where raw kernel names are unavailable, e.g. Windows
                   torch 2.6).
  throughput     — wall-clock median ms + tokens/s for block forward+backward
                   (CUDA events), and for a full MetisLM train step.
  bandwidth      — effective memory-bandwidth estimate for the decode path
                   (T_q=1, memory-bound):  bytes read by one block step
                   / median decode time.

Results are written as JSON + a Markdown report under ``benchmarks/results/``.

Usage:
    python benchmarks/benchmark_block.py                # auto device, 20 iters
    python benchmarks/benchmark_block.py --device cuda  # force CUDA
    python benchmarks/benchmark_block.py --mode kernel  # kernel count only
    python benchmarks/benchmark_block.py --mode through # throughput only
    python benchmarks/benchmark_block.py --out results/custom.json
"""

import argparse
import collections
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from metis.config import ModelConfig  # noqa: E402
from metis.model import (  # noqa: E402
    CausalSelfAttention,
    MetisLM,
    RMSNorm,
    SwiGLU,
    apply_rope,
)

# ──────────────────────────────────────────────────────────────────────────────
# Reference (pre-fusion) modules — kept self-contained so the A/B is honest
# ──────────────────────────────────────────────────────────────────────────────

class RefRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        n = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * n).type_as(x) * self.weight


class RefQKNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.norm = RefRMSNorm(dim, eps)

    def forward(self, x):
        return self.norm(x).type_as(x)


class RefAttn(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.n_groups = cfg.n_heads // cfg.n_kv_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.d_model = cfg.d_model
        self.kv_dim = cfg.d_model * cfg.n_kv_heads // cfg.n_heads
        self.use_rope = cfg.use_rope
        self.use_qk_norm = cfg.use_qk_norm
        self.backend_request = getattr(cfg, "attn_backend", "auto")
        self.use_flash_attn = cfg.use_flash_attn
        self.last_backend = None

        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, self.kv_dim, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        if self.use_qk_norm:
            self.q_norm = RefQKNorm(self.head_dim)
            self.k_norm = RefQKNorm(self.head_dim)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len)).view(
                1, 1, cfg.max_seq_len, cfg.max_seq_len
            ),
        )
        if self.use_rope:
            from metis.model import precompute_rope_frequencies
            self.register_buffer(
                "rope_freqs",
                precompute_rope_frequencies(self.head_dim, cfg.max_seq_len),
                persistent=False,
            )

    def forward(self, x, kv_cache=None):
        from metis.attn import _repeat_kv as repeat_kv
        B, T, C = x.size()
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        if self.use_qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        if self.use_rope:
            if kv_cache is not None:
                offset = kv_cache[0].size(2)
                rs = self.rope_freqs[offset:offset + T, :].unsqueeze(0).unsqueeze(0)
                qr = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
                kr = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))
                q = torch.view_as_real(qr * rs).flatten(-2).type_as(q)
                k = torch.view_as_real(kr * rs).flatten(-2).type_as(k)
            else:
                q, k = apply_rope(q, self.rope_freqs), apply_rope(k, self.rope_freqs)
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        new_cache = (k, v)
        from metis.attn import causal_attention
        dp = self.attn_dropout.p if self.training else 0.0
        y = causal_attention(
            q, k, v, dropout_p=dp, is_causal=True,
            n_heads=self.n_heads, n_kv_heads=self.n_kv_heads,
            backend=self.backend_request, use_flash_attn=self.use_flash_attn,
            training=self.training,
        )
        y = y.transpose(1, 2).contiguous().view(B, -1, C)
        return self.resid_dropout(self.o_proj(y)), new_cache


class RefSwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden = int(4 * cfg.d_model * 2 / 3)
        hidden = ((hidden + 7) // 8) * 8
        self.hidden = hidden
        self.w1 = nn.Linear(cfg.d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, cfg.d_model, bias=False)
        self.w3 = nn.Linear(cfg.d_model, hidden, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class RefBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        ncls = RefRMSNorm if cfg.use_rmsnorm else nn.LayerNorm
        self.ln_1 = ncls(cfg.d_model)
        self.attn = RefAttn(cfg)
        self.ln_2 = ncls(cfg.d_model)
        self.ffn = RefSwiGLU(cfg)

    def forward(self, x, kv_cache=None):
        ao, nc = self.attn(self.ln_1(x), kv_cache=kv_cache)
        x = x + ao
        x = x + self.ffn(self.ln_2(x))
        return x, nc


def _load_ref_from_fused(ref, fused_block):
    """Load a RefBlock from a fused TransformerBlock's legacy state_dict."""
    sd = fused_block.state_dict()
    missing, unexpected = ref.load_state_dict(sd, strict=False)
    return ref


# ──────────────────────────────────────────────────────────────────────────────
# Kernel-count metric (Windows torch has no CUPTI kernel names)
# ──────────────────────────────────────────────────────────────────────────────

_META_OPS = frozenset({
    "aten::view", "aten::as_strided", "aten::_unsafe_view", "aten::slice",
    "aten::expand", "aten::t", "aten::transpose", "aten::reshape",
    "aten::_reshape_alias", "aten::unsqueeze", "aten::squeeze", "aten::flatten",
    "aten::detach", "aten::alias", "aten::_conj", "aten::resolve_conj",
    "aten::select", "aten::result_type", "aten::split", "aten::chunk",
    "aten::unbind", "aten::narrow", "aten::permute", "aten::size", "aten::numel",
    "aten::dim", "aten::SymInt", "aten::resolve_neg", "aten::_dim_arange",
    "aten::arange", "aten::_local_scalar_dense", "aten::item",
})


def count_kernel_ops(fn):
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    kc = collections.Counter()
    for e in prof.events():
        if e.cuda_time_total > 0 and e.name not in _META_OPS:
            kc[e.name] += 1
    return kc


# ──────────────────────────────────────────────────────────────────────────────
# Timer (CUDA-event median, peak memory)
# ──────────────────────────────────────────────────────────────────────────────

class Timer:
    def __init__(self, device, runs=20, warmup=3):
        self.device = device
        self.runs = runs
        self.warmup = warmup
        self.is_cuda = device.startswith("cuda")

    def time(self, fn, *a, **kw):
        for _ in range(self.warmup):
            fn(*a, **kw)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base_mem = torch.cuda.memory_allocated()
        samples = []
        for _ in range(self.runs):
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            fn(*a, **kw)
            e.record()
            torch.cuda.synchronize()
            samples.append(s.elapsed_time(e))
        peak = torch.cuda.max_memory_allocated() - base_mem
        samples.sort()
        return {
            "median_ms": samples[len(samples) // 2],
            "mean_ms": sum(samples) / len(samples),
            "peak_mem_MB": peak / 1e6,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Benchmarks
# ──────────────────────────────────────────────────────────────────────────────

def _cfg(**kw):
    d = dict(
        d_model=256, n_heads=4, n_kv_heads=2, n_layers=4, max_seq_len=256,
        vocab_size=512, dropout=0.1, use_rmsnorm=True, use_swiglu=True,
        use_rope=True, use_flash_attn=True, attn_backend="auto",
        tie_weights=False,
    )
    d.update(kw)
    return ModelConfig(**d)


def bench_kernel_count(device, iters):
    """Kernel op-count: fused block vs reference block (eval, bf16)."""
    cfg = _cfg()
    from metis.model import TransformerBlock
    fused = TransformerBlock(cfg).cuda().eval()
    ref = RefBlock(cfg)
    # Load fused → ref via legacy state_dict.
    sd = fused.state_dict()
    missing, unexpected = ref.load_state_dict(sd, strict=False)
    ref = ref.cuda().eval()

    x = torch.randn(2, 128, cfg.d_model, device=device, dtype=torch.float16)

    def run_fused():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            fused(x)

    def run_ref():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ref(x)

    kc_fused = count_kernel_ops(run_fused)
    kc_ref = count_kernel_ops(run_ref)
    return sum(kc_ref.values()), kc_ref, sum(kc_fused.values()), kc_fused


def bench_throughput(device, iters):
    """Wall-clock: block fwd+bwd and full MetisLM train step — fused vs ref."""
    cfg = _cfg()
    from metis.model import TransformerBlock
    timer = Timer(device, runs=iters, warmup=3)
    fused = TransformerBlock(cfg).cuda().train()
    ref = RefBlock(cfg)
    ref.load_state_dict(fused.state_dict(), strict=False)
    ref = ref.cuda().train()

    B, T, D = 2, 128, cfg.d_model
    x = torch.randn(B, T, D, device=device, dtype=torch.float16)
    results = {}

    for label, block in [("fused", fused), ("reference", ref)]:
        def step():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y, _ = block(x)
                y.sum().backward()
            block.zero_grad(set_to_none=True)
        results[label] = timer.time(step)
        torch.cuda.empty_cache()

    # Full MetisLM train step — fused
    mm = MetisLM(cfg).cuda().train()
    idx = torch.randint(0, cfg.vocab_size, (B, T), device=device)
    opt = mm.configure_optimizers(0.1, 1e-3, device)
    scaler = torch.amp.GradScaler("cuda")

    def full_step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss, _ = mm(idx, targets=idx)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(mm.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

    results["fused_full_step"] = timer.time(full_step)
    return results


def bench_decode_bandwidth(device, iters):
    """Decode-path memory bandwidth: bytes read / decode time (GB/s)."""
    cfg = _cfg()
    from metis.model import TransformerBlock
    block = TransformerBlock(cfg).cuda().eval()
    ref = RefBlock(cfg)
    ref.load_state_dict(block.state_dict(), strict=False)
    ref = ref.cuda().eval()

    B, T_new, D = 1, 1, cfg.d_model
    x = torch.randn(B, T_new, D, device=device, dtype=torch.float16)
    results = {}

    for label, blk in [("fused", block), ("reference", ref)]:
        # Prefill: fill KV cache with 128 tokens.
        x_pre = torch.randn(1, 128, D, device=device, dtype=torch.float16)
        with torch.no_grad():
            _, cache = blk(x_pre)
        timer = Timer(device, runs=iters, warmup=5)
        with torch.no_grad():
            timer.time(blk, x, cache)
            peak = torch.cuda.max_memory_allocated()
        r = timer.time(lambda: blk(x, cache))
        # Bytes: all block params at bf16 + activations for 1 token.
        n_params = sum(p.numel() for p in blk.parameters())
        bytes_per_step = n_params * 2 + B * T_new * D * 2  # read params + write new tok
        r["bytes_per_step"] = bytes_per_step
        r["bandwidth_GB_s"] = bytes_per_step / (r["median_ms"] / 1e3) / 1e9
        results[label] = r
        torch.cuda.empty_cache()

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────────────────────────

def git_sha():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=os.path.dirname(__file__),
        )
        return out.stdout.strip() or "n/a"
    except Exception:
        return "n/a"


def write_report(results, out_path):
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    md = out.with_suffix(".md")
    L = []
    L.append("# Metis — fused-block benchmark report")
    L.append("")
    L.append(f"- Date: {results['date']}")
    L.append(f"- Device: `{results['device']}`")
    L.append(f"- PyTorch: `{results['torch']}`")
    L.append(f"- Git SHA: `{results.get('git_sha', 'n/a')}`")
    L.append("")

    if "kernel_count" in results:
        kc = results["kernel_count"]
        L.append("## Kernel count (bf16 autocast block forward)")
        L.append("")
        L.append("| variant | total kernel ops | Δ |")
        L.append("|---------|----------------:|--:|")
        L.append(f"| reference (pre-fusion) | {kc['ref_total']} | — |")
        L.append(f"| fused | {kc['fused_total']} | {kc['fused_total'] - kc['ref_total']} |")
        L.append("")

    if "throughput" in results:
        tp = results["throughput"]
        L.append("## Throughput (block fwd+bwd + full train step, bf16)")
        L.append("")
        L.append("| path | variant | median (ms) | tokens/s | peak mem (MB) |")
        L.append("|------|---------|------------:|---------:|--------------:|")
        for variant in ("fused", "reference"):
            if variant in tp:
                t = tp[variant]
                tps = 2 * 128 / (t["median_ms"] / 1e3)
                L.append(f"| block fwd+bwd | {variant} | {t['median_ms']:.2f} | {tps:.0f} | {t['peak_mem_MB']:.1f} |")
        if "fused_full_step" in tp:
            t = tp["fused_full_step"]
            tps = 2 * 128 / (t["median_ms"] / 1e3)
            L.append(f"| full train step | fused | {t['median_ms']:.2f} | {tps:.0f} | {t['peak_mem_MB']:.1f} |")
        L.append("")

    if "bandwidth" in results:
        bw = results["bandwidth"]
        L.append("## Decode bandwidth (T_q=1, bf16, 128-token cache)")
        L.append("")
        L.append("| variant | decode (ms) | params (M) | BW (GB/s) |")
        L.append("|---------|------------:|-----------:|----------:|")
        for variant in ("fused", "reference"):
            if variant in bw:
                b = bw[variant]
                L.append(f"| {variant} | {b['median_ms']:.2f} | {b['bytes_per_step']/2/1e6:.2f} | {b['bandwidth_GB_s']:.1f} |")
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
    ap = argparse.ArgumentParser(description="Fused vs reference block benchmark")
    ap.add_argument("--mode", choices=["kernel", "through", "bandwidth", "both", "all"], default="all")
    ap.add_argument("--device", default=None)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    _configure_stdio()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    results = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "device": device,
        "torch": torch.__version__,
        "git_sha": git_sha(),
    }

    print(f"Metis fused-block benchmark — device={device}")

    if args.mode in ("kernel", "all"):
        print("\n[kernel] kernel op-count (bf16 autocast block forward):")
        ref_t, ref_kc, fus_t, fus_kc = bench_kernel_count(device, args.iters)
        print(f"  reference: {ref_t} ops")
        print(f"  fused:     {fus_t} ops  (Δ {fus_t - ref_t})")
        results["kernel_count"] = {
            "ref_total": ref_t, "fused_total": fus_t,
            "ref_top10": ref_kc.most_common(10),
            "fused_top10": fus_kc.most_common(10),
        }

    if args.mode in ("through", "all"):
        print("\n[throughput] block fwd+bwd + full train step (bf16):")
        tp = bench_throughput(device, args.iters)
        for k, v in tp.items():
            print(f"  {k:24s}: {v['median_ms']:8.2f}ms  peak={v['peak_mem_MB']:7.1f}MB")
        results["throughput"] = tp

    if args.mode in ("bandwidth", "all"):
        print("\n[bandwidth] decode path (T_q=1, bf16, 128-token cache):")
        bw = bench_decode_bandwidth(device, args.iters)
        for k, v in bw.items():
            print(f"  {k:12s}: {v['median_ms']:6.2f}ms  BW={v['bandwidth_GB_s']:.1f}GB/s")
        results["bandwidth"] = bw

    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results", f"benchmark_block_{datetime.now():%Y%m%d_%H%M%S}.json",
    )
    md = write_report(results, out)
    print(f"\nReport written:\n  JSON: {out}\n  Markdown: {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
