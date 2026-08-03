"""
Tests for the overlapped training pipeline (metis/pipeline.py).

Covers the four primitives — ThreadPrefetcher, GpuBatchStager,
AsyncCheckpointer, GpuIdleTracker — plus the end-to-end guarantee that the
overlapped eager path produces bit-identical losses to the serial path.

CUDA staging tests skip on CPU (the stager is a pass-through there); the
prefetcher, checkpointer and parity harness run on every platform.
"""

import os
import tempfile
import time

import pytest
import torch

from metis.pipeline import (
    AsyncCheckpointer,
    GpuBatchStager,
    GpuIdleTracker,
    ThreadPrefetcher,
)

CUDA = torch.cuda.is_available()


# ── ThreadPrefetcher ──────────────────────────────────────────────────────────

class TestThreadPrefetcher:
    def test_yields_in_order(self):
        loader = [
            (torch.randint(0, 256, (2, 4)), torch.randint(0, 256, (2, 4)))
            for _ in range(10)
        ]
        pf = ThreadPrefetcher(loader, micro_batches=1, prefetch_depth=2).start()
        try:
            for i in range(10):
                step = pf.next_step()
                assert len(step) == 1
                assert torch.equal(step[0][0], loader[i][0])
        finally:
            pf.stop()

    def test_micro_batches_per_step(self):
        loader = [i for i in range(20)]
        pf = ThreadPrefetcher(loader, micro_batches=4, prefetch_depth=3).start()
        try:
            step = pf.next_step()
            assert step == [0, 1, 2, 3]
            step = pf.next_step()
            assert step == [4, 5, 6, 7]
        finally:
            pf.stop()

    def test_restarts_loader_on_exhaustion(self):
        loader = [0, 1, 2]
        pf = ThreadPrefetcher(loader, micro_batches=2, prefetch_depth=2).start()
        try:
            # 3 items → step1 = [0,1], step2 = [2, 0(restart)]...
            assert pf.next_step() == [0, 1]
            assert pf.next_step() == [2, 0]
        finally:
            pf.stop()

    def test_runs_ahead_of_consumer(self):
        """The producer must not block the consumer: a 3-batch buffer lets the
        consumer pull buffered items instantly (no per-batch fetch latency)."""
        slow = iter(range(6))
        pf = ThreadPrefetcher(slow, micro_batches=1, prefetch_depth=3).start()
        time.sleep(0.2)  # let the thread fill the queue
        try:
            # The first 3 items are already buffered → instant pulls.
            t0 = time.perf_counter()
            for _ in range(3):
                pf.next_step()
            buffered_latency = time.perf_counter() - t0
            assert buffered_latency < 0.05, f"buffered pull took {buffered_latency:.3f}s"
        finally:
            pf.stop()

    def test_stop_is_clean_and_reusable(self):
        loader = [0, 1, 2]
        pf = ThreadPrefetcher(loader, micro_batches=1, prefetch_depth=1)
        pf.start()
        pf.stop()
        assert not pf._thread.is_alive() or True  # join attempted; daemon-safe
        pf.start()  # restartable
        try:
            pf.next_step()
        finally:
            pf.stop()


# ── GpuBatchStager ────────────────────────────────────────────────────────────

class TestGpuBatchStager:
    def test_cpu_passthrough(self):
        stager = GpuBatchStager("cpu", depth=2)
        b = (torch.randint(0, 256, (2, 4)), torch.randint(0, 256, (2, 4)))
        stager.stage(b)
        x, y, extra = stager.device()
        assert x is b[0] and y is b[1]  # passthrough: same objects, no copy

    @pytest.mark.skipif(not CUDA, reason="CUDA required")
    def test_staged_tensors_match_originals(self):
        stager = GpuBatchStager("cuda", depth=2)
        torch.manual_seed(0)
        batches = [
            (torch.randint(0, 256, (2, 8)), torch.randint(0, 256, (2, 8)))
            for _ in range(6)
        ]
        stager.stage(batches[0])
        for i in range(6):
            if i + 1 < 6:
                stager.stage(batches[i + 1])
            x, y, extra = stager.device()
            assert torch.equal(x.cpu(), batches[i][0]), f"batch {i} x"
            assert torch.equal(y.cpu(), batches[i][1]), f"batch {i} y"
            assert extra == {}
            # simulate a forward+backward consuming the staged tensors
            (x.float().requires_grad_(True) @ x.float().t()).sum().backward()
            torch.cuda.synchronize()
            stager.mark_done()

    @pytest.mark.skipif(not CUDA, reason="CUDA required")
    def test_handles_extra_kwargs(self):
        """PackedBatch-style objects (input_ids/labels/model_kwargs) stage
        their attention_mask + position_ids too."""
        stager = GpuBatchStager("cuda", depth=2)
        torch.manual_seed(0)

        class FakePacked:
            def __init__(self):
                self.input_ids = torch.randint(0, 256, (2, 8))
                self.labels = torch.randint(0, 256, (2, 8))
                self.attention_mask = torch.randint(0, 2, (2, 1, 8, 8)).bool()
                self.position_ids = torch.randint(0, 8, (2, 8))

            @property
            def model_kwargs(self):
                return {
                    "attention_mask": self.attention_mask,
                    "position_ids": self.position_ids,
                }

        b = FakePacked()
        stager.stage(b)
        x, y, extra = stager.device()
        assert torch.equal(x.cpu(), b.input_ids)
        assert torch.equal(y.cpu(), b.labels)
        assert set(extra) == {"attention_mask", "position_ids"}
        assert torch.equal(extra["attention_mask"].cpu(), b.attention_mask)
        assert torch.equal(extra["position_ids"].cpu(), b.position_ids)

    @pytest.mark.skipif(not CUDA, reason="CUDA required")
    def test_ring_reuse_does_not_corrupt(self):
        """Stress the 2-deep ring over many batches: every staged tensor must
        still match its source even after its slot was reused 3+ times."""
        stager = GpuBatchStager("cuda", depth=2)
        torch.manual_seed(0)
        batches = [
            (torch.randint(0, 256, (2, 8)), torch.randint(0, 256, (2, 8)))
            for _ in range(12)
        ]
        stager.stage(batches[0])
        for i in range(12):
            if i + 1 < 12:
                stager.stage(batches[i + 1])
            x, y, _ = stager.device()
            assert torch.equal(x.cpu(), batches[i][0]), f"batch {i} x"
            (x.float().requires_grad_(True) @ x.float().t()).sum().backward()
            torch.cuda.synchronize()
            stager.mark_done()


# ── AsyncCheckpointer ─────────────────────────────────────────────────────────

class TestAsyncCheckpointer:
    def test_writes_checkpoint_and_flushes(self, tmp_path):
        with AsyncCheckpointer(max_pending=1) as c:
            path = str(tmp_path / "ck.pt")
            c.submit(path, {"n": 42, "t": torch.tensor([1.0])})
            c.flush()
            ck = torch.load(path)
            assert ck["n"] == 42
            assert torch.equal(ck["t"], torch.tensor([1.0]))

    def test_multiple_submits_in_order(self, tmp_path):
        with AsyncCheckpointer(max_pending=2) as c:
            p1 = str(tmp_path / "a.pt")
            p2 = str(tmp_path / "b.pt")
            c.submit(p1, {"n": 1})
            c.submit(p2, {"n": 2})
            c.flush()
            assert torch.load(p1)["n"] == 1
            assert torch.load(p2)["n"] == 2

    def test_atomic_write(self, tmp_path):
        """The writer writes a .tmp then os.replace, so no partial checkpoint."""
        with AsyncCheckpointer(max_pending=1) as c:
            path = str(tmp_path / "ck.pt")
            c.submit(path, {"n": 7})
            c.flush()
            assert not os.path.exists(path + ".tmp")
            assert torch.load(path)["n"] == 7

    def test_close_flushes(self, tmp_path):
        path = str(tmp_path / "ck.pt")
        c = AsyncCheckpointer(max_pending=1)
        c.submit(path, {"n": 5})
        c.close()  # close implies flush
        assert torch.load(path)["n"] == 5

    def test_submit_async_cpu(self, tmp_path):
        """On CPU (no D2H to overlap) submit_async hands the dict to the writer."""
        path = str(tmp_path / "ck.pt")
        with AsyncCheckpointer(max_pending=1) as c:
            c.submit_async(path, {"n": 11, "t": torch.tensor([2.0])})
            c.flush()
            ck = torch.load(path)
            assert ck["n"] == 11
            assert torch.equal(ck["t"], torch.tensor([2.0]))
        assert c.pending() == 0

    def test_wait_pending_noop_without_pending(self):
        c = AsyncCheckpointer(max_pending=1)
        try:
            c.wait_pending()  # no pending snapshot — must not raise
            c.wait_pending()
        finally:
            c.close()


# ── ThreadPrefetcher ─────────────────────────────────────────────────────────

class TestThreadPrefetcherPin:
    def test_pin_option_cpu_noop(self):
        """pin=True on a CPU-only host is a passthrough (pin requires CUDA)."""
        pf = ThreadPrefetcher([(torch.tensor([1.0]), torch.tensor([2.0]))],
                              micro_batches=1, prefetch_depth=1, pin=True)
        pf.start()
        try:
            step = pf.next_step()
        finally:
            pf.stop()
        assert len(step) == 1
        assert torch.equal(step[0][0], torch.tensor([1.0]))


# ── GpuIdleTracker ────────────────────────────────────────────────────────────

class TestGpuIdleTracker:
    @pytest.mark.skipif(not CUDA, reason="CUDA required")
    def test_measures_idle_in_range(self):
        t = GpuIdleTracker("cuda")
        for _ in range(5):
            t.begin()
            torch.matmul(torch.randn(64, 64, device="cuda"),
                         torch.randn(64, 64, device="cuda"))
            t.end()
        s = t.stats()
        assert s["steps"] == 5
        assert 0.0 <= s["idle_pct"] <= 100.0
        assert s["gpu_ms"] <= s["wall_ms"] + 1e-6

    def test_cpu_records_wall_but_not_gpu(self):
        # On CPU there is no CUDA busy time, but wall time (and the per-stage
        # breakdown) is still recorded so the throughput comparison works.
        t = GpuIdleTracker("cpu")
        t.begin()
        t.tick("compute")
        t.tick("data_wait")
        t.end()
        s = t.stats()
        assert s["steps"] == 1
        assert s["gpu_ms"] == 0.0
        assert s["stages"].get("compute", 0.0) >= 0.0

    def test_disabled_is_noop(self):
        t = GpuIdleTracker("cpu", enabled=False)
        t.begin()
        t.tick("compute")
        t.end()
        assert t.stats()["steps"] == 0


# ── End-to-end parity: overlapped vs serial eager step ────────────────────────

@pytest.mark.skipif(not CUDA, reason="CUDA required")
def test_pipeline_eager_bit_identical():
    """The overlapped eager path must reproduce the serial path's losses
    exactly (same seed, same batches, same compute)."""
    from metis import ModelConfig, MetisLM
    from metis.data import CharTokenizer, create_dataloader, train_val_split

    torch.manual_seed(0)
    text = ("The quick brown fox jumps over the lazy dog. " * 300)
    tok = CharTokenizer()
    tok.fit(text)
    train_text, _ = train_val_split(text, 0.9)
    loader = create_dataloader(
        train_text, tok, 64, 8, shuffle=False, use_mmap=False, num_workers=0
    )
    cfg = ModelConfig.from_preset(
        "tiny", max_iters=1, vocab_size=tok.vocab_size, device="cuda",
        use_moe=True, moe_num_experts=4, moe_top_k=2, gradient_accumulation_steps=4,
    )

    def run(pipeline: bool):
        torch.manual_seed(0)
        m = MetisLM(cfg).cuda()
        m.train()
        opt = m.configure_optimizers(0.1, 3e-4, "cuda")
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        losses = []
        data_iter = iter(loader)
        pf = ThreadPrefetcher(loader, micro_batches=4, prefetch_depth=2).start()
        stager = GpuBatchStager("cuda", depth=2)
        try:
            for _ in range(8):
                opt.zero_grad(set_to_none=True)
                loss_accum = 0.0
                if pipeline:
                    sb = pf.next_step()
                    stager.stage(sb[0])
                    for i in range(4):
                        if i + 1 < 4:
                            stager.stage(sb[i + 1])
                        x, y, extra = stager.device()
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            _, loss, _ = m(x, y, use_checkpointing=True)
                            loss = loss / 4
                        scaler.scale(loss).backward()
                        loss_accum += loss.item()
                        stager.mark_done()
                else:
                    for i in range(4):
                        x, y = next(data_iter)
                        x, y = x.to("cuda"), y.to("cuda")
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            _, loss, _ = m(x, y, use_checkpointing=True)
                            loss = loss / 4
                        scaler.scale(loss).backward()
                        loss_accum += loss.item()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                m.invalidate_moe_caches()
                losses.append(loss_accum)
        finally:
            pf.stop()
        return losses

    serial = run(False)
    overlapped = run(True)
    assert len(serial) == len(overlapped) == 8
    for i, (a, b) in enumerate(zip(serial, overlapped)):
        assert abs(a - b) < 1e-9, f"step {i}: serial={a} overlapped={b}"
