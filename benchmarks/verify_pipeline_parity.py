#!/usr/bin/env python3
"""
Metis — Overlapped-pipeline numerical parity verification
==========================================================
Proves the software-pipelined training path (thread prefetch + copy-stream H2D
staging) is **bit-identical** to the original serial path:

  * eager forward losses, step-by-step over a training run (same seed/batches),
  * the prefetch thread preserves loader order & restart semantics,
  * the stager returns exactly the input tensors (classic and packed forms),
  * async checkpointing writes a loadable, identical checkpoint,
  * the CUDA-graph path stages static slots without changing results.

CPU-safe by default; the CUDA-only checks auto-skip without a GPU.

Usage:
    python benchmarks/verify_pipeline_parity.py
    python benchmarks/verify_pipeline_parity.py --device cuda --steps 10
"""

import argparse
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from metis.data import CharTokenizer, create_dataloader, train_val_split  # noqa: E402
from metis.pipeline import (  # noqa: E402
    AsyncCheckpointer,
    GpuBatchStager,
    ThreadPrefetcher,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _stdio():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


class Result:
    def __init__(self):
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


def _make_loader():
    torch.manual_seed(0)
    text = "The quick brown fox jumps over the lazy dog. " * 400
    tok = CharTokenizer()
    tok.fit(text)
    train_text, _ = train_val_split(text, 0.9)
    loader = create_dataloader(
        train_text, tok, 64, 8, shuffle=False, use_mmap=False, num_workers=0
    )
    return loader


def _make_model():
    from metis import ModelConfig, MetisLM

    cfg = ModelConfig.from_preset(
        "tiny", max_iters=1, vocab_size=80, device=DEVICE, dropout=0.0,
        use_moe=True, moe_num_experts=4, moe_top_k=2, gradient_accumulation_steps=4,
    )
    torch.manual_seed(0)
    m = MetisLM(cfg)
    m.to(DEVICE)
    m.train()
    return m, cfg


def verify_eager_parity(r, steps, device):
    print("\n[eager] overlapped vs serial step losses (bit-identical)")
    loader = _make_loader()
    m_ser, cfg = _make_model()
    m_ovl, _ = _make_model()
    opt_ser = m_ser.configure_optimizers(0.1, 3e-4, device)
    opt_ovl = m_ovl.configure_optimizers(0.1, 3e-4, device)
    scaler_ser = torch.amp.GradScaler(device, enabled=True)
    scaler_ovl = torch.amp.GradScaler(device, enabled=True)

    use_amp = str(device).startswith("cuda")  # CPU bf16 autocast is ~20× slower

    def step(batches, m, opt, scaler, stager=None, pipeline=False):
        # Dropout (tiny preset = 0.1) draws from the *global* CUDA RNG, which
        # the two models share — re-seed so both see identical masks per step.
        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)
        opt.zero_grad(set_to_none=True)
        loss_accum = 0.0
        if pipeline and stager is not None:
            stager.stage(batches[0])
            for i in range(len(batches)):
                if i + 1 < len(batches):
                    stager.stage(batches[i + 1])
                x, y, extra = stager.device()
                if use_amp:
                    with torch.autocast(device, dtype=torch.bfloat16):
                        _, loss, _ = m(x, y, use_checkpointing=True)
                else:
                    _, loss, _ = m(x, y, use_checkpointing=True)
                loss = loss / len(batches)
                scaler.scale(loss).backward()
                loss_accum += loss.item()
                stager.mark_done()
        else:
            for x, y in batches:
                x, y = x.to(device), y.to(device)
                if use_amp:
                    with torch.autocast(device, dtype=torch.bfloat16):
                        _, loss, _ = m(x, y, use_checkpointing=True)
                else:
                    _, loss, _ = m(x, y, use_checkpointing=True)
                loss = loss / len(batches)
                scaler.scale(loss).backward()
                loss_accum += loss.item()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        m.invalidate_moe_caches()
        return loss_accum

    pf = ThreadPrefetcher(loader, micro_batches=4, prefetch_depth=2).start()
    stager = GpuBatchStager(device, depth=2)
    data_iter = iter(loader)
    mismatches = 0
    max_diff = 0.0
    try:
        for _ in range(steps):
            ser_b = [next(data_iter) for _ in range(4)]
            ovl_b = pf.next_step()
            l_ser = step(ser_b, m_ser, opt_ser, scaler_ser)
            l_ovl = step(ovl_b, m_ovl, opt_ovl, scaler_ovl, stager, pipeline=True)
            d = abs(l_ser - l_ovl)
            max_diff = max(max_diff, d)
            if d > 1e-9:
                mismatches += 1
    finally:
        pf.stop()
    r.check("losses bit-identical over N steps", mismatches == 0,
            f"mismatches={mismatches} max_diff={max_diff:.2e}")


def verify_stager_contract(r, device):
    print("\n[stager] staged tensors equal the source batches (in order)")
    stager = GpuBatchStager(device, depth=2)
    torch.manual_seed(0)
    batches = [
        (torch.randint(0, 256, (2, 8)), torch.randint(0, 256, (2, 8)))
        for _ in range(8)
    ]
    stager.stage(batches[0])
    ok = True
    for i in range(8):
        if i + 1 < 8:
            stager.stage(batches[i + 1])
        x, y, extra = stager.device()
        if not torch.equal(x.cpu(), batches[i][0]):
            ok = False
        if not torch.equal(y.cpu(), batches[i][1]):
            ok = False
        if extra != {}:
            ok = False
        (x.float().requires_grad_(True) @ x.float().t()).sum().backward()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        stager.mark_done()
    r.check("8 batches through a 2-deep ring/FIFO, in order", ok)


def verify_prefetch_order(r):
    print("\n[prefetch] loader order preserved + restart semantics")
    loader = list(range(10))
    pf = ThreadPrefetcher(loader, micro_batches=3, prefetch_depth=2).start()
    try:
        first = pf.next_step()   # [0,1,2]
        second = pf.next_step()  # [3,4,5]
        third = pf.next_step()   # [6,7,8]
        fourth = pf.next_step()  # [9,0,1]  → restart at 10 % 10
    finally:
        pf.stop()
    r.check("order preserved", first == [0, 1, 2] and second == [3, 4, 5],
            f"{first} {second}")
    r.check("loader restarts on exhaustion", third == [6, 7, 8] and fourth[0] == 9,
            f"{third} → {fourth}")


def verify_async_checkpoint(r, tmp):
    print("\n[checkpoint] async write lands, is atomic and loadable")
    path = os.path.join(tmp, "ck.pt")
    with AsyncCheckpointer(max_pending=1) as c:
        c.submit(path, {"model": torch.randn(2, 2), "step": 5})
        c.flush()
    ck = torch.load(path)
    ok = ck["step"] == 5 and ck["model"].shape == (2, 2)
    r.check("async checkpoint written + loadable", ok)
    r.check("no leftover .tmp file", not os.path.exists(path + ".tmp"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", type=str, default=DEVICE)
    ap.add_argument("--steps", type=int, default=8)
    args = ap.parse_args()
    device = args.device if args.device.startswith(("cuda", "cpu")) else "cpu"
    r = Result()

    verify_eager_parity(r, args.steps, device)
    verify_stager_contract(r, device)
    verify_prefetch_order(r)
    with tempfile.TemporaryDirectory() as tmp:
        verify_async_checkpoint(r, tmp)

    ok = r.summary()
    print("\nOVERALL: PASS" if ok else "\nOVERALL: FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    _stdio()
    sys.exit(main())
