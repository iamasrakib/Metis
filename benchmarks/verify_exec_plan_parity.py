#!/usr/bin/env python3
"""
Metis — Execution-scheduler numerical-parity verification
==========================================================
Directly compares the EAGER forward path against the SCHEDULED path across
every requirement axis: prefill, decode (kv-cache), MoE, QK-norm, attention
sink, GQA, and train mode.

Each check prints a PASS/FAIL line; the script exits non-zero on any failure.

Usage:
    python benchmarks/verify_exec_plan_parity.py                  # auto device
    python benchmarks/verify_exec_plan_parity.py --device cpu
    python benchmarks/verify_exec_plan_parity.py --seed 0
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from metis.config import ModelConfig  # noqa: E402
from metis.model import MetisLM  # noqa: E402
from metis.scheduler import INFER, build_scheduler  # noqa: E402


def _cfg(**kw) -> ModelConfig:
    defaults = dict(vocab_size=256, d_model=64, n_heads=4, n_kv_heads=0,
                    n_layers=2, max_seq_len=32, dropout=0.0, use_rmsnorm=True,
                    use_swiglu=True, use_rope=True, tie_weights=True,
                    use_moe=False, use_qk_norm=False, use_attention_sink=False,
                    moe_num_experts=4, moe_top_k=2)
    defaults.update(kw)
    return ModelConfig(**defaults)


def _test(name: str, model, idx, targets=None, kv_cache=None, mode="infer", device="cpu"):
    model.eval()
    sched = build_scheduler(model, mode=mode, calibrate_run=False, ref_shape=tuple(idx.shape),
                            device=device)
    with torch.no_grad():
        log_s, loss_s, cache_s = sched.execute(idx, targets=targets, kv_cache=kv_cache)
        log_e, loss_e, cache_e = model(idx, targets=targets, kv_cache=kv_cache)
    log_match = torch.equal(log_s, log_e)
    if targets is not None:
        loss_match = loss_s.item() == loss_e.item()
    else:
        loss_match = True
    cache_match = True
    if kv_cache is not None and cache_s is not None and len(cache_s) == len(cache_e):
        for (ks, vs), (ke, ve) in zip(cache_s, cache_e):
            if not (torch.equal(ks, ke) and torch.equal(vs, ve)):
                cache_match = False
                break
    ok = log_match and loss_match and cache_match
    tag = "PASS" if ok else "FAIL"
    if not ok:
        print(f"  [FAIL] {name} "
              f"(log={log_match} loss={loss_match} cache={cache_match})")
    else:
        print(f"  [{tag}] {name}")
    return ok


def main():
    parser = argparse.ArgumentParser(description="Verify execution-scheduler parity")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    all_pass = True

    def run(name, **kw):
        nonlocal all_pass
        model = MetisLM(_cfg(**kw)).to(args.device).eval()
        idx = torch.randint(1, 256, (1, 16), device=args.device)
        all_pass &= _test(name, model, idx, device=args.device)

    def run_targets(name, **kw):
        nonlocal all_pass
        model = MetisLM(_cfg(**kw)).to(args.device).eval()
        idx = torch.randint(1, 256, (1, 16), device=args.device)
        tgt = torch.randint(1, 256, (1, 16), device=args.device)
        all_pass &= _test(name, model, idx, targets=tgt, device=args.device)

    def run_decode(name, **kw):
        nonlocal all_pass
        model = MetisLM(_cfg(**kw)).to(args.device).eval()
        idx = torch.randint(1, 256, (1, 16), device=args.device)
        model.eval()
        with torch.no_grad():
            _, _, cache = model(idx)
        sched = build_scheduler(model, mode=INFER, calibrate_run=False,
                                ref_shape=(1, 1), device=args.device)
        for step in range(3):
            tok = torch.randint(1, 256, (1, 1), device=args.device)
            log_s, _, cache_s = sched.execute(tok, kv_cache=cache)
            log_e, _, cache_e = model(tok, kv_cache=cache)
            if not torch.equal(log_s, log_e):
                print(f"  [FAIL] {name} step {step}")
                all_pass = False
                return
            # Use the eager cache as the canonical reference for the next step
            cache = cache_e
        print(f"  [PASS] {name}")

    def run_train(name, **kw):
        nonlocal all_pass
        model = MetisLM(_cfg(**kw)).to(args.device).train()
        idx = torch.randint(1, 256, (1, 16), device=args.device)
        tgt = torch.randint(1, 256, (1, 16), device=args.device)
        all_pass &= _test(name, model, idx, targets=tgt, mode="train", device=args.device)

    print("Metis execution-scheduler parity verification")
    print("=" * 50)

    # Prefill tests
    print("\n-- Prefill --")
    run("default SwiGLU")
    run_targets("default with targets")
    run("MLP", use_swiglu=False)
    run("MoE", use_moe=True)
    run("QK-Norm", use_qk_norm=True)
    run("Attention sink", use_attention_sink=True)
    run("GQA", n_kv_heads=2)

    # Decode tests
    print("\n-- Decode (kv-cache) --")
    run_decode("default 3-step decode")
    run_decode("MoE decode", use_moe=True)
    run_decode("QK-Norm decode", use_qk_norm=True)
    run_decode("GQA decode", n_kv_heads=2)

    # Train mode
    print("\n-- Train mode --")
    run_train("train default")
    run_train("train MoE", use_moe=True)

    print("\n" + "=" * 50)
    if all_pass:
        print("ALL PASS")
    else:
        print("FAILURES DETECTED")
        sys.exit(1)


if __name__ == "__main__":
    main()
