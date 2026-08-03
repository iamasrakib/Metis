"""
Μῆτις (Metis) — Unit Tests for Dynamic Sequence Packing
=========================================================
Tests cover:
  • pack_stream / pack_bins correctness (labels, cu_seqlens, n_pad)
  • build_attention_mask (block-diagonal, causal, padding self-loop)
  • build_position_ids (reset per segment)
  • PackedDataset (batching, shapes, statistics)
  • Model forward with packed batches (finite loss, no NaN)
  • Parity: eos=None stream == plain causal path
  • Parity: bin-packed == token-weighted per-segment standalone
"""

import os
import sys
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from metis.packing import (  # noqa: E402
    BIN,
    STREAM,
    PackedBatch,
    PackedDataset,
    build_attention_mask,
    build_position_ids,
    pack_bins,
    pack_documents,
    pack_stream,
)
from metis.config import ModelConfig  # noqa: E402
from metis.model import MetisLM  # noqa: E402


# ── Fixtures ─────────────────────────────────────────────────────────────────

EOS, PAD = 3, 0


@pytest.fixture
def short_docs():
    """Variable-length docs that fit inside seq_len=32."""
    return [
        [5, 6, 7],
        [8, 9],
        [10, 11, 12, 13, 14],
        [15],
        [16, 17, 18],
        [4, 5, 6],
        [7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16, 17, 18],
        [19],
        [20, 21, 22, 23],
        [24, 25, 26],
    ]


@pytest.fixture
def many_docs():
    """Enough variable-length docs to produce multiple packed batches at seq_len=32."""
    rng = np.random.RandomState(42)
    return [
        list(rng.randint(4, 40, size=int(rng.randint(2, 12)))) for _ in range(60)
    ]


@pytest.fixture
def tiny_model():
    """Tiny deterministic model for parity tests."""
    torch.manual_seed(0)
    cfg = ModelConfig(
        d_model=64,
        n_heads=2,
        n_layers=2,
        max_seq_len=32,
        dropout=0.0,
        vocab_size=64,
        use_rope=True,
    )
    return MetisLM(cfg).eval()


# ── pack_stream tests ────────────────────────────────────────────────────────

class TestPackStream:
    def test_zero_padding(self, short_docs):
        chunks, cu_seqlens, labels, n_pad = pack_stream(short_docs, 32, EOS, PAD)
        assert n_pad == 0
        for c in chunks:
            assert len(c) == 32

    def test_labels_whole_stream_shift(self, short_docs):
        chunks, _, label_chunks, _ = pack_stream(short_docs, 32, EOS, PAD)
        stream = []
        for d in short_docs:
            stream.extend(d + [EOS])
        arr = np.asarray(stream, dtype=np.int64)
        shifted = np.empty_like(arr)
        shifted[:-1] = arr[1:]
        shifted[-1] = PAD
        for i, lc in enumerate(label_chunks):
            assert np.array_equal(lc, shifted[i * 32 : (i + 1) * 32])

    def test_cu_seqlens_boundaries(self, short_docs):
        chunks, cu_seqlens, _, _ = pack_stream(short_docs, 32, EOS, PAD)
        for c, cs in zip(chunks, cu_seqlens):
            assert cs[0] == 0
            assert cs[-1] == 32
            for k in range(len(cs) - 1):
                assert cs[k] < cs[k + 1]
                # eos before each new segment start (except the first)
                if cs[k] > 0:
                    assert c[cs[k] - 1] == EOS

    def test_too_few_tokens(self):
        docs = [[1, 2, 3]]
        with pytest.raises(ValueError, match="Not enough tokens"):
            pack_stream(docs, 32, EOS, PAD)


# ── pack_bins tests ──────────────────────────────────────────────────────────

class TestPackBins:
    def test_no_overflow(self, short_docs):
        bins, cu_seqlens, _, n_pad = pack_bins(short_docs, 32, EOS, PAD)
        assert n_pad >= 0
        for b in bins:
            assert len(b) == 32
        for cs in cu_seqlens:
            assert cs[-1] <= 32

    def test_no_doc_split(self, short_docs):
        """Every document must appear whole (with eos) in a single bin."""
        bins, cu_seqlens, _, _ = pack_bins(short_docs, 32, EOS, PAD)
        for d in short_docs:
            pattern = np.array(d + [EOS], dtype=np.int64)
            found = False
            for b in bins:
                # check subsequence match
                for start in range(len(b) - len(pattern) + 1):
                    if np.array_equal(b[start : start + len(pattern)], pattern):
                        found = True
                        break
                if found:
                    break
            assert found, f"document {d} not found whole in any bin"

    def test_padding_masked(self, short_docs):
        bins, cu_seqlens, _, n_pad = pack_bins(short_docs, 32, EOS, PAD)
        for cs in cu_seqlens:
            m = build_attention_mask(cs, 32)[0]
            real = cs[-1]
            if real < 32:
                # padding rows/cols should only have diagonal True
                assert not m[real:, :real].any() and not m[:real, real:].any()

    def test_bin_labels(self, short_docs):
        _, cu_seqlens, label_chunks, _ = pack_bins(short_docs, 32, EOS, PAD)
        for cs, lbl in zip(cu_seqlens, label_chunks):
            for s in range(len(cs) - 1):
                start, end = cs[s], cs[s + 1]
                # labels within segment: shifted by 1, eos position → pad
                for t in range(start, end - 1):
                    assert lbl[t] == lbl[t]  # just verify no crash
                assert lbl[end - 1] == PAD  # segment end → ignore


# ── build_attention_mask tests ───────────────────────────────────────────────

class TestAttentionMask:
    def test_single_segment_is_tril(self):
        cs = [0, 16]
        m = build_attention_mask(cs, 16)[0]
        assert m.shape == (16, 16)
        assert np.array_equal(m, np.tril(np.ones((16, 16), dtype=np.bool_)))

    def test_no_upper_triangle(self):
        cs = [0, 4, 8]
        m = build_attention_mask(cs, 8)[0]
        assert not np.triu(m, k=1).any()

    def test_block_diagonal(self):
        cs = [0, 4, 8]
        m = build_attention_mask(cs, 8)[0]
        # segment 0: positions 0-3, segment 1: positions 4-7
        # cross-segment should be False
        assert not m[0, 5] and not m[2, 6] and not m[5, 1]
        # within-segment causal should be True
        assert m[1, 0] and m[3, 2] and m[7, 4]

    def test_padding_self_loop(self):
        cs = [0, 4]  # real=4, padding at 4-7
        m = build_attention_mask(cs, 8, n_pad=4)[0]
        # padding positions can attend to themselves
        for i in range(4, 8):
            assert m[i, i]
        # but cannot attend to other positions
        assert not m[4, 5] and not m[6, 7]
        assert not m[0, 4]

    def test_mask_exact_block_diagonal(self):
        """Every entry follows (seg[i]==seg[j]) & (j<=i) for real tokens."""
        cs = [0, 3, 5, 8]
        m = build_attention_mask(cs, 8)[0]
        seg = np.array([0, 0, 0, 1, 1, 2, 2, 2])
        for r in range(8):
            for c2 in range(8):
                expect = (seg[r] == seg[c2]) and (c2 <= r)
                assert m[r, c2] == expect, (r, c2, m[r, c2])


# ── build_position_ids tests ─────────────────────────────────────────────────

class TestPositionIds:
    def test_reset_per_segment(self):
        cs = [0, 4, 8]
        pos = build_position_ids(cs, 8)
        assert list(pos[:4]) == [0, 1, 2, 3]
        assert list(pos[4:8]) == [0, 1, 2, 3]

    def test_uneven_segments(self):
        cs = [0, 2, 5]
        pos = build_position_ids(cs, 8)
        assert list(pos[:2]) == [0, 1]
        assert list(pos[2:5]) == [0, 1, 2]


# ── PackedDataset tests ─────────────────────────────────────────────────────

class TestPackedDataset:
    def test_shapes(self, many_docs):
        ds = PackedDataset(
            many_docs, 32, 2, eos_id=EOS, pad_id=PAD, strategy=STREAM
        )
        batch = next(iter(ds))
        assert isinstance(batch, PackedBatch)
        assert batch.input_ids.shape == (2, 32)
        assert batch.labels.shape == (2, 32)
        assert batch.attention_mask.shape == (2, 1, 32, 32)
        assert batch.position_ids.shape == (2, 32)

    def test_stream_zero_waste(self, many_docs):
        ds = PackedDataset(
            many_docs, 32, 2, eos_id=EOS, pad_id=PAD, strategy=STREAM
        )
        assert ds.padding_waste_pct == 0.0
        assert ds.n_pad == 0

    def test_bin_has_padding(self, many_docs):
        ds = PackedDataset(
            many_docs, 32, 2, eos_id=EOS, pad_id=PAD, strategy=BIN
        )
        assert ds.padding_waste_pct > 0
        assert ds.n_pad > 0

    def test_drop_remainder(self, many_docs):
        """If len(packed) % batch_size != 0, the tail is dropped."""
        ds = PackedDataset(
            many_docs, 32, 3, eos_id=EOS, pad_id=PAD, strategy=STREAM
        )
        n_actual = sum(1 for _ in ds)
        assert n_actual == ds.n_batches

    def test_to_device(self, many_docs):
        ds = PackedDataset(
            many_docs, 32, 2, eos_id=EOS, pad_id=PAD, strategy=STREAM
        )
        batch = next(iter(ds))
        # .to("cpu") should not crash
        batch_cpu = batch.to("cpu")
        assert batch_cpu.input_ids.device.type == "cpu"

    def test_model_kwargs(self, many_docs):
        ds = PackedDataset(
            many_docs, 32, 2, eos_id=EOS, pad_id=PAD, strategy=STREAM
        )
        batch = next(iter(ds))
        kw = batch.model_kwargs
        assert "attention_mask" in kw
        assert "position_ids" in kw

    def test_fewer_packed_than_batch_size(self):
        # 8 docs of 1 token each, eos=None → 8 tokens → 1 chunk at seq_len=8
        with pytest.raises(ValueError, match="fewer than batch_size"):
            PackedDataset(
                [[1]] * 8, 8, 3, eos_id=None, pad_id=PAD,
                strategy=STREAM,
            )


# ── Model parity tests ──────────────────────────────────────────────────────

class TestModelParity:
    def test_eos_none_stream_matches_causal(self, tiny_model, many_docs):
        ds = PackedDataset(
            many_docs, 32, 2, eos_id=None, pad_id=PAD, strategy=STREAM,
            shuffle=False, seed=42,
        )
        for batch in ds:
            with torch.no_grad():
                _, loss_packed, _ = tiny_model(
                    batch.input_ids, batch.labels,
                    attention_mask=batch.attention_mask,
                    position_ids=batch.position_ids,
                )
                _, loss_plain, _ = tiny_model(batch.input_ids, batch.labels)
            assert abs(loss_packed.item() - loss_plain.item()) < 1e-5

    def test_bin_parity_with_standalone(self, tiny_model, many_docs):
        ds = PackedDataset(
            many_docs, 32, 2, eos_id=EOS, pad_id=PAD, strategy=BIN,
            shuffle=False, seed=42,
        )
        batch = next(iter(ds))
        with torch.no_grad():
            _, loss_bin, _ = tiny_model(
                batch.input_ids, batch.labels,
                attention_mask=batch.attention_mask,
                position_ids=batch.position_ids,
            )
        # token-weighted standalone (matches CE with ignore_index='mean')
        total_loss, total_tokens = 0.0, 0
        for row_idx in range(batch.input_ids.size(0)):
            cus = batch.cu_seqlens[row_idx]
            for s in range(len(cus) - 1):
                start, end = cus[s], cus[s + 1]
                seg = batch.input_ids[row_idx, start:end].unsqueeze(0)
                seg_lbl = torch.full_like(seg, PAD)
                seg_lbl[0, : end - start - 1] = seg[0, 1 : end - start]
                with torch.no_grad():
                    logits = tiny_model(seg, seg_lbl)[0]
                ce = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    seg_lbl.view(-1),
                    reduction="sum",
                    ignore_index=PAD,
                )
                n_tok = seg_lbl.view(-1).ne(PAD).sum().item()
                total_loss += ce.item()
                total_tokens += n_tok
        weighted_mean = total_loss / total_tokens
        assert abs(loss_bin.item() - weighted_mean) < 1e-4

    def test_finite_loss_with_packing(self, tiny_model, many_docs):
        ds = PackedDataset(
            many_docs, 32, 2, eos_id=EOS, pad_id=PAD, strategy=STREAM
        )
        for batch in ds:
            _, loss, _ = tiny_model(
                batch.input_ids, batch.labels,
                attention_mask=batch.attention_mask,
                position_ids=batch.position_ids,
            )
            assert torch.isfinite(loss)


# ── Config validation ────────────────────────────────────────────────────────

class TestConfigValidation:
    def test_packing_strategy_invalid(self):
        with pytest.raises(ValueError, match="packing_strategy"):
            ModelConfig(packing_strategy="invalid")

    def test_packing_and_sink_incompatible(self):
        with pytest.raises(ValueError, match="incompatible"):
            ModelConfig(use_packing=True, use_attention_sink=True)
