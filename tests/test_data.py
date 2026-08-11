"""
Μῆτις (Metis) — Unit Tests for Data Pipeline (BPETokenizer, Datasets)
"""

import os
import sys
import tempfile

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from metis.data import (
    BPETokenizer,
    CharTokenizer,
    MMapDataset,
    TextDataset,
    create_dataloader,
    load_text,
    tokenize_and_cache,
    train_val_split,
)

# ── BPETokenizer Tests ───────────────────────────────────────────────────────

class TestBPETokenizer:
    def test_init_char_fallback(self):
        """Char fallback should work without tiktoken."""
        tok = BPETokenizer(encoding_name="char")
        assert tok._is_char_mode is True
        assert tok.is_bpe is False

    def test_fit_and_encode_char(self):
        tok = BPETokenizer(encoding_name="char")
        tok.fit("hello world")
        ids = tok.encode("hello")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)
        assert len(ids) > 0

    def test_encode_decode_roundtrip_char(self):
        tok = BPETokenizer(encoding_name="char")
        tok.fit("hello world")
        text = "hello"
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        assert decoded == text

    def test_special_tokens_encode_decode(self):
        tok = BPETokenizer(encoding_name="char")
        tok.fit("abc")
        ids = tok.encode("abc", add_bos=True, add_eos=True)
        assert ids[0] == tok.bos_id
        assert ids[-1] == tok.eos_id
        decoded = tok.decode(ids, skip_special=True)
        assert decoded == "abc"
        # skip_special=False keeps the special tokens in the decoded string.
        decoded_with_special = tok.decode(ids, skip_special=False)
        assert "<bos>" in decoded_with_special
        assert "<eos>" in decoded_with_special

    def test_bpe_special_ids_do_not_collide_with_native_vocab(self):
        """Specials are registered ABOVE the native tiktoken vocab (regression).

        In cl100k_base, native id 0 is ``'!'`` — if ``<pad>`` & co. reused ids
        0-3 (the original bug), padding and decode would silently alias real
        tokens. Verify every special id is disjoint from the native ranks and
        that a real BPE round-trip still works.
        """
        import tiktoken

        tok = BPETokenizer(encoding_name="gpt2")
        assert tok.is_bpe
        base = tiktoken.get_encoding("gpt2")
        native = set(base._mergeable_ranks.values()) | set(base._special_tokens.values())
        special_ids = {tok.pad_id, tok.unk_id, tok.bos_id, tok.eos_id}
        assert special_ids.isdisjoint(native)
        assert len(special_ids) == 4  # and distinct from each other
        # A native token whose id would have fallen in the old 0-3 slot is not
        # shadowed by any special token.
        assert tok.encode("!")[0] not in special_ids
        assert tok.decode(tok.encode("hello world")) == "hello world"
        # add_bos/add_eos still bracket with the remapped (high) ids.
        ids = tok.encode("hi", add_bos=True, add_eos=True)
        assert ids[0] == tok.bos_id and ids[-1] == tok.eos_id

    def test_vocab_properties(self):
        tok = BPETokenizer(encoding_name="char")
        tok.fit("xyz")
        assert tok.pad_id == 0
        assert tok.unk_id == 1
        assert tok.bos_id == 2
        assert tok.eos_id == 3
        assert len(tok) == tok.vocab_size

    def test_save_load_char_json(self):
        tok = BPETokenizer(encoding_name="char")
        tok.fit("hello world")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
            tok.save(path)
        loaded = BPETokenizer(encoding_name="char")
        loaded.load(path)
        assert loaded.vocab_size == tok.vocab_size
        ids = loaded.encode("hello")
        assert loaded.decode(ids) == "hello"
        os.unlink(path)

    def test_legacy_char_tokenizer_compat(self):
        """New BPETokenizer should load legacy CharTokenizer JSON."""
        legacy = CharTokenizer()
        legacy.fit("test data")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
            legacy.save(path)
        bpe = BPETokenizer(encoding_name="char")
        bpe.load(path)
        assert bpe.vocab_size == legacy.vocab_size
        assert bpe.encode("test") == legacy.encode("test")
        os.unlink(path)


# ── CharTokenizer Tests ──────────────────────────────────────────────────────

class TestCharTokenizer:
    def test_fit_and_encode(self):
        tok = CharTokenizer()
        tok.fit("abc")
        assert tok.vocab_size >= 6  # 3 chars + 4 special = 7
        ids = tok.encode("abc")
        assert len(ids) == 3

    def test_encode_with_bos_eos(self):
        tok = CharTokenizer()
        tok.fit("hello")
        ids = tok.encode("hello", add_bos=True, add_eos=True)
        assert ids[0] == tok.bos_id
        assert ids[-1] == tok.eos_id
        assert len(ids) == 7  # 5 + 2

    def test_decode_skip_special(self):
        tok = CharTokenizer()
        tok.fit("ab")
        ids = [tok.bos_id, *tok.encode("ab"), tok.eos_id]
        assert tok.decode(ids, skip_special=True) == "ab"
        assert tok.decode(ids, skip_special=False) != "ab"  # special tokens present

    def test_unknown_char(self):
        tok = CharTokenizer()
        tok.fit("abc")
        ids = tok.encode("xyz")
        assert tok.unk_id in ids

    def test_save_load_json(self):
        tok = CharTokenizer()
        tok.fit("hello world")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
            tok.save(path)
        loaded = CharTokenizer()
        loaded.load(path)
        assert loaded.vocab_size == tok.vocab_size
        assert loaded.encode("hello") == tok.encode("hello")
        os.unlink(path)

    def test_save_load_pickle(self):
        tok = CharTokenizer()
        tok.fit("test")
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False, mode="wb") as f:
            path = f.name
            import pickle
            pickle.dump({"stoi": tok.stoi, "itos": tok.itos, "vocab_size": tok.vocab_size}, f)
        loaded = CharTokenizer()
        loaded.load(path)
        assert loaded.vocab_size == tok.vocab_size
        os.unlink(path)


# ── Dataset Tests ─────────────────────────────────────────────────────────────

class TestTextDataset:
    def test_basic(self):
        data = torch.arange(100, dtype=torch.long)
        ds = TextDataset(data, seq_len=10)
        assert len(ds) == 90
        x, y = ds[0]
        assert x.shape == (10,)
        assert y.shape == (10,)
        # y[i] = x[i+1], so y[:-1] should equal x[1:]
        assert torch.equal(y[:-1], x[1:])
        assert y[-1] == data[10]  # y's last = data[0 + 10]

    def test_too_short(self):
        with pytest.raises(ValueError):
            TextDataset(torch.arange(5), seq_len=10)


class TestMMapDataset:
    def test_from_array(self):
        data = np.arange(1000, dtype=np.uint16)
        ds = MMapDataset(data, seq_len=50)
        assert len(ds) == 950
        x, y = ds[0]
        assert x.shape == (50,)
        assert y.shape == (50,)


# ── Data Loading Tests ───────────────────────────────────────────────────────

class TestLoadText:
    def test_single_file(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
            f.write("hello world")
            path = f.name
        text = load_text(path)
        assert text == "hello world"
        os.unlink(path)

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_text("/nonexistent/file.txt")


class TestTrainValSplit:
    def test_basic_split(self):
        train, val = train_val_split("abcdefghij", train_ratio=0.8)
        assert len(train) == 8
        assert len(val) == 2

    def test_default_ratio(self):
        train, val = train_val_split("x" * 100)
        assert len(train) == 90
        assert len(val) == 10


class TestCreateDataloader:
    def test_basic(self):
        tok = CharTokenizer()
        tok.fit("hello world this is a test " * 100)
        loader = create_dataloader(
            "hello world this is a test " * 100,
            tok, seq_len=16, batch_size=4, use_mmap=False,
        )
        batch = next(iter(loader))
        x, y = batch
        assert x.shape == (4, 16)
        assert y.shape == (4, 16)

    def test_ddp_ranks_shard_disjoint(self):
        """DDP ranks must iterate disjoint slices (no duplicated training data)."""
        tok = CharTokenizer()
        tok.fit("hello world this is a test " * 100)
        text = "hello world this is a test " * 100
        r0 = create_dataloader(text, tok, seq_len=16, batch_size=4,
                               use_mmap=False, rank=0, world_size=2)
        r1 = create_dataloader(text, tok, seq_len=16, batch_size=4,
                               use_mmap=False, rank=1, world_size=2)
        r0_ids = {tuple(b[0].flatten().tolist()) for b in r0}
        r1_ids = {tuple(b[0].flatten().tolist()) for b in r1}
        assert r0_ids and r1_ids
        assert r0_ids.isdisjoint(r1_ids)


# ── Tokenization Cache ───────────────────────────────────────────────────────

class TestTokenizeAndCache:
    def test_cache_roundtrip(self):
        tok = BPETokenizer(encoding_name="char")
        tok.fit("hello world " * 100)
        text = "hello world " * 100
        unique_path = os.path.join(tempfile.mkdtemp(), f"_test_cache_{id(text)}.txt")
        data, cache_path = tokenize_and_cache(text, tok, seq_len=16,
                                              dataset_path=unique_path)
        assert len(data) > 0
        assert os.path.exists(cache_path)
        # A second call must load from the on-disk cache, not re-tokenize.
        data2, _ = tokenize_and_cache(text, tok, seq_len=16,
                                      dataset_path=unique_path)
        assert np.array_equal(data, data2)
        del data2  # release the memmap file handle (Windows locks the file)
        os.unlink(cache_path)

    def test_create_dataloader_drops_corrupt_cache(self):
        """A corrupt .npy cache is deleted and rebuilt, not loaded silently.

        Regression test for the fallback that catches (OSError, ValueError,
        EOFError) from the mmap cache path and removes the unusable file.
        """
        tok = BPETokenizer(encoding_name="char")
        tok.fit("hello world " * 100)
        text = "hello world " * 100
        unique_path = os.path.join(tempfile.mkdtemp(), "_test_corrupt.txt")

        # Prime a valid cache, then corrupt the file on disk.
        _, cache_path = tokenize_and_cache(text, tok, seq_len=16,
                                           dataset_path=unique_path)
        assert os.path.exists(cache_path)
        with open(cache_path, "wb") as f:
            f.write(b"\x00corrupt-npy-header" * 8)
        assert os.path.exists(cache_path)

        # Loading must fall back to in-memory tokenization AND drop the file.
        loader = create_dataloader(
            text, tok, seq_len=16, batch_size=2,
            dataset_path=unique_path, use_mmap=True,
        )
        assert not os.path.exists(cache_path)
        batches = list(loader)
        assert len(batches) > 0
        x, y = batches[0]
        assert x.shape[0] == 2 and x.shape[1] == 16


# ── Streaming tokenization (large-corpus RAM safety) ─────────────────────────

class TestStreamingTokenizer:
    def test_streamed_matches_in_memory(self, tmp_path):
        """Streaming from disk must equal in-memory chunked tokenization.

        Regression guard for the Colab RAM OOM: a large corpus is never loaded
        as one str, but the tokens produced must be identical to the in-memory
        path so training sees the same data.
        """
        from metis import data as d

        corpus = "\n".join(
            f"doc {i}: quick brown fox — café 日本語 😀 €500" for i in range(300)
        )
        p = tmp_path / "corpus.txt"
        p.write_text(corpus, encoding="utf-8")
        tok = BPETokenizer(encoding_name="cl100k_base")
        tok.fit(corpus)
        d._CHUNK_CHARS = 64  # force many small windows in both paths
        try:
            ref = d._tokenize_large(tok, corpus)
            got = d._tokenize_file_streaming(str(p), tok)
        finally:
            d._CHUNK_CHARS = 64_000_000
        assert got.tolist() == ref.tolist()

    def test_streamed_no_newlines_is_bounded(self, tmp_path):
        """A newline-free file must still tokenize without an unbounded carry."""
        from metis import data as d

        p = tmp_path / "nodata.txt"
        p.write_text("hello world 🚀 " * 50000, encoding="utf-8")
        tok = BPETokenizer(encoding_name="cl100k_base")
        tok.fit("hello world 🚀")
        d._CHUNK_CHARS = 64
        try:
            got = d._tokenize_file_streaming(str(p), tok)
        finally:
            d._CHUNK_CHARS = 64_000_000
        assert len(got) > 0

    def test_create_dataloaders_from_file(self, tmp_path):
        """Builds train/val loaders over the streamed token array."""
        from metis.data import MMapDataset, create_dataloaders_from_file

        corpus = "\n".join(
            f"sentence {i}: the quick brown fox jumps over the lazy dog"
            for i in range(500)
        )
        p = tmp_path / "corpus.txt"
        p.write_text(corpus, encoding="utf-8")
        tok = BPETokenizer(encoding_name="cl100k_base")
        tok.fit(corpus)
        tr, va = create_dataloaders_from_file(
            str(p), tok, seq_len=16, batch_size=4, train_ratio=0.8,
        )
        assert isinstance(tr.dataset, MMapDataset)
        assert isinstance(va.dataset, MMapDataset)
        assert len(tr.dataset) > 0 and len(va.dataset) > 0

    def test_streaming_cache_recovers_from_corruption(self, tmp_path):
        """A cache truncated by a killed run is deleted and rebuilt, not fatal.

        Regression guard for Colab SIGKILL landing mid-save: the partial .npy
        must not crash the next run — it is dropped and re-tokenized.
        """
        from metis.data import create_dataloaders_from_file

        corpus = "\n".join(
            f"line {i}: the quick brown fox jumps over the lazy dog"
            for i in range(200)
        )
        p = tmp_path / "corpus.txt"
        p.write_text(corpus, encoding="utf-8")
        tok = BPETokenizer(encoding_name="cl100k_base")
        tok.fit(corpus)

        tr, va = create_dataloaders_from_file(
            str(p), tok, seq_len=8, batch_size=4, train_ratio=0.8,
        )
        assert len(tr.dataset) > 0

        # Corrupt every cache file this dataset produced, then reload.
        from metis.data import _cache_path, _file_fingerprint, _tokenizer_cache_name
        cache_path = _cache_path(
            str(p), _tokenizer_cache_name(tok), 8, _file_fingerprint(str(p)),
        )
        with open(cache_path, "wb") as f:
            f.write(b"\x00partial" * 4)
        tr2, va2 = create_dataloaders_from_file(
            str(p), tok, seq_len=8, batch_size=4, train_ratio=0.8,
        )
        assert len(tr2.dataset) > 0  # rebuilt, not crashed
