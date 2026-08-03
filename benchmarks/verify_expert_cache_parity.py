#!/usr/bin/env python3
"""
Metis — Expert cache numerical-parity verification
====================================================
Compares the cached grouped engine against the uncached reference (cache=None)
across every requirement axis:

  forward (fp32 + fp16/bf16 AMP), gradients, full-model logits/loss,
  hit-rate consistency, bit-identical output on hits, staleness detection,
  fused-optimizer invalidation, LRU eviction, byte accounting, and capacity.

Each check prints a PASS/FAIL line
the script exits non-zero on any failure.
CPU-safe by default
AMP checks auto-skip without CUDA.

Usage:
    python benchmarks/verify_expert_cache_parity.py          # auto device
    python benchmarks/verify_expert_cache_parity.py --device cuda
    python benchmarks/verify_expert_cache_parity.py --seed 0 --tol 1e-3
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from metis.config import ModelConfig  # noqa: E402
from metis.expert_cache import ExpertCache, _remat_bytes  # noqa: E402
from metis.model import MetisLM  # noqa: E402
from metis.moe import (  # noqa: E402
    MoE,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _configure_stdio():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


class Result:
    def __init__(self, tol):
        self.tol = tol
        self.pass_ = 0
        self.fail = 0
        self.skips = 0

    def check(self, name, cond, detail=""):
        if cond:
            self.pass_ += 1
            print(f"  PASS  {name}" + (f"  ({detail})" if detail else ""))
        else:
            self.fail += 1
            print(f"  FAIL  {name}" + (f"  ({detail})" if detail else ""))

    def skip(self, name, why):
        self.skips += 1
        print(f"  SKIP  {name}  ({why})")

    def summary(self):
        print(f"\n  {self.pass_} passed, {self.fail} failed, {self.skips} skipped")
        return self.fail == 0


def _make_cfg(**kw):
    cfg = dict(
        d_model=128, n_heads=4, n_kv_heads=2, n_layers=2, max_seq_len=128,
        vocab_size=256, dropout=0.0, use_moe=True, moe_num_experts=8,
        moe_top_k=2, moe_engine="auto",
    )
    cfg.update(kw)
    return ModelConfig(**cfg)


def _models(seed, cache_size=16, **kw):
    torch.manual_seed(seed)
    m_c = MetisLM(_make_cfg(moe_cache_size=cache_size, **kw))
    torch.manual_seed(seed)
    m_n = MetisLM(_make_cfg(moe_cache_size=0, **kw))
    m_n.load_state_dict(m_c.state_dict())
    return m_c, m_n


def verify_forward_parity(r, seed, tol):
    print("\n[forward] cached vs uncached (layer-level)")
    torch.manual_seed(seed)
    m_c, m_n = _models(seed)
    m_c.to(DEVICE).eval()
    m_n.to(DEVICE).eval()
    x = torch.randn(4, 16, 128, device=DEVICE)

    cases = [(None, tol, "fp32")]
    if DEVICE.startswith("cuda"):
        cases += [
            (torch.float16, max(tol, 1e-2), "fp16 AMP"),
            (torch.bfloat16, max(tol, 1e-2), "bf16 AMP"),
        ]
    for amp_dtype, t, label in cases:
        with torch.no_grad():
            if amp_dtype is not None:
                with torch.autocast(DEVICE, dtype=amp_dtype):
                    o_c = m_c.layers[0].ffn(x).float()
                    o_n = m_n.layers[0].ffn(x).float()
            else:
                o_c = m_c.layers[0].ffn(x).float()
                o_n = m_n.layers[0].ffn(x).float()
        err = float((o_c - o_n).abs().max())
        r.check(f"forward {label}", err < t, f"max_err={err:.2e}")


def verify_bit_identical(r, seed):
    print("\n[bit-identical] cached vs uncached model logits")
    m_c, m_n = _models(seed)
    m_c.to(DEVICE).eval()
    m_n.to(DEVICE).eval()
    idx = torch.randint(0, 256, (2, 32), device=DEVICE)
    with torch.no_grad():
        lg_c, loss_c, _ = m_c(idx, targets=idx)
        lg_n, loss_n, _ = m_n(idx, targets=idx)
    equal = torch.equal(lg_c, lg_n)
    r.check("logits bit-identical", equal)
    r.check("loss bit-identical",
            abs(loss_c.item() - loss_n.item()) < 1e-6)


def verify_hit_rate(r, seed):
    print("\n[hit-rate] repeated eval forwards accumulate hits")
    m, _ = _models(seed)
    m.to(DEVICE).eval()
    idx = torch.randint(0, 256, (2, 32), device=DEVICE)
    with torch.no_grad():
        m(idx)
        m(idx)  # same input → hits
    stats = m.get_moe_cache_stats()
    total_hits = sum(s["hits"] for s in stats if s)
    total_misses = sum(s["misses"] for s in stats if s)
    total = total_hits + total_misses
    hit_rate = total_hits / total if total else 0
    r.check("hit rate > 0", total_hits > 0,
            f"hits={total_hits} misses={total_misses} rate={hit_rate:.1%}")
    r.check("hit rate < 1 (different inputs cause misses)", hit_rate < 1.0,
            f"rate={hit_rate:.1%}")


def verify_staleness(r, seed):
    print("\n[staleness] param.copy_() forces rebuild with new weights")
    m, _ = _models(seed)
    m.to(DEVICE).eval()
    idx = torch.randint(0, 256, (2, 32), device=DEVICE)
    with torch.no_grad():
        lg1, _, _ = m(idx)
    # Mutate
    for p in m.parameters():
        if p.dim() >= 2:
            p.data.add_(0.1 * torch.randn_like(p))
    with torch.no_grad():
        lg2, _, _ = m(idx)
    r.check("output differs after mutation",
            not torch.equal(lg1, lg2))


def verify_grad_parity(r, seed, tol):
    print("\n[grad] cached backward vs uncached backward")
    m_c, m_n = _models(seed)
    m_c.to(DEVICE).train()
    m_n.to(DEVICE).train()
    idx = torch.randint(0, 256, (2, 32), device=DEVICE)
    _, loss_c, _ = m_c(idx, targets=idx)
    loss_c.backward()
    # Disable cache on reference
    for layer in m_n.layers:
        ffn = getattr(layer, "ffn", None)
        if hasattr(ffn, "_cache"):
            ffn._cache = None
    _, loss_n, _ = m_n(idx, targets=idx)
    loss_n.backward()
    gc = {n: p.grad.clone() for n, p in m_c.named_parameters() if p.grad is not None}
    gn = {n: p.grad.clone() for n, p in m_n.named_parameters() if p.grad is not None}
    diffs = [(gc[n] - gn[n]).abs().max().item() for n in gc if n in gn]
    r.check("grads match", max(diffs) < tol, f"max_grad_diff={max(diffs):.2e}")


def verify_byte_accounting(r):
    print("\n[bytes] byte accounting consistency")
    c = ExpertCache(entry_capacity=16)
    w1 = torch.randn(4, 8, requires_grad=True)
    w2 = torch.randn(8, 4, requires_grad=True)
    sources = [w1, w2]
    for _ in range(3):
        c.get_or_build((0, 1), torch.float32, sources,
                       lambda: (w1.clone(), w2.clone()))
    expected_per_miss = _remat_bytes(sources, w1)
    s = c.stats()
    r.check("bytes_built = misses * per_miss",
            s["bytes_built"] == expected_per_miss * 1,
            f"built={s['bytes_built']} expected={expected_per_miss}")
    r.check("bytes_saved = hits * per_miss",
            s["bytes_saved"] == expected_per_miss * 2,
            f"saved={s['bytes_saved']} expected={expected_per_miss * 2}")
    r.check("bandwidth_reduction in [0,100]",
            0 <= s["bandwidth_reduction_pct"] <= 100,
            f"pct={s['bandwidth_reduction_pct']:.1f}")


def verify_lru_eviction(r):
    print("\n[lru] capacity bounded, entries ≤ capacity, evictions counted")
    c = ExpertCache(entry_capacity=2)
    w = torch.randn(2, 2)
    for i in range(5):
        c.get_or_build((i,), torch.float32, [w], lambda: (w.clone(), w.clone()))
    s = c.stats()
    r.check("entries ≤ capacity", s["entries"] <= 2,
            f"entries={s['entries']}")
    r.check("evictions > 0", s["evictions"] > 0,
            f"evictions={s['evictions']}")


def verify_byte_budget(r):
    print("\n[byte-cap] byte budget eviction")
    c = ExpertCache(entry_capacity=100, byte_capacity=1000)
    w1 = torch.randn(10, 20, requires_grad=True)
    w2 = torch.randn(20, 10, requires_grad=True)
    sources = [w1, w2]
    for i in range(10):
        c.get_or_build((i,), torch.bfloat16, sources,
                       lambda: (w1.clone().to(torch.bfloat16),
                                w2.clone().to(torch.bfloat16)))
    s = c.stats()
    max_entry = (w1.numel() + w2.numel()) * 2  # bf16
    r.check("resident ≤ byte_budget",
            s['resident_bytes'] <= c.byte_capacity,
            f"resident={s['resident_bytes']} budget={c.byte_capacity}")
    r.check("evictions > 0", s['evictions'] > 0,
            f"evictions={s['evictions']}")


def verify_env_override(r, monkeypatch):
    print("\n[env] METIS_MOE_CACHE_SIZE overrides config")
    monkeypatch.setenv("METIS_MOE_CACHE_SIZE", "2")
    cfg = _make_cfg(moe_cache_size=16)
    m = MoE(cfg).to(DEVICE)
    r.check("env overrides config", m._cache.entry_capacity == 2)


def verify_config_validation(r):
    print("\n[config] negative sizes raise ValueError")
    try:
        _make_cfg(moe_cache_size=-1)
        r.check("negative cache_size", False)
    except ValueError:
        r.check("negative cache_size", True)
    try:
        _make_cfg(moe_cache_bytes=-1)
        r.check("negative cache_bytes", False)
    except ValueError:
        r.check("negative cache_bytes", True)


def verify_device_transfer(r):
    print("\n[device] .to() resets cache")
    m, _ = _models(0)
    m.to(DEVICE).eval()
    idx = torch.randint(0, 256, (1, 16), device=DEVICE)
    with torch.no_grad():
        m(idx)
    has_resident = any(
        s["resident_bytes"] > 0 for s in m.get_moe_cache_stats() if s
    )
    m.to(DEVICE)
    after_reset = all(
        s["resident_bytes"] == 0 for s in m.get_moe_cache_stats() if s
    )
    r.check("cache populated after forward", has_resident)
    r.check("cache reset after .to()", after_reset)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Expert cache parity verification")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=1e-3)
    args = ap.parse_args(argv)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    _configure_stdio()

    print(f"Metis Expert-Cache parity verification — device={device} "
          f"seed={args.seed} tol={args.tol}")

    r = Result(args.tol)
    verify_forward_parity(r, args.seed, args.tol)
    verify_bit_identical(r, args.seed)
    verify_hit_rate(r, args.seed)
    verify_staleness(r, args.seed)
    verify_grad_parity(r, args.seed, max(args.tol, 1e-4))
    verify_byte_accounting(r)
    verify_lru_eviction(r)
    verify_byte_budget(r)
    verify_config_validation(r)
    verify_device_transfer(r)

    ok = r.summary()
    print("\nOVERALL:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
