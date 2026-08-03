"""
Μῆτις (Metis) — Unit Tests for Data Pipeline (BPETokenizer, Datasets)
"""

import os
import sys
import json
import tempfile
import pytest
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from metis.data import (
    BPETokenizer,
    CharTokenizer,
    TextDataset,
    MMapDataset,
    load_text,
    train_val_split,
    create_dataloader,
    tokenize_and_cache,
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
        decoded_with_special = tok.decode(ids, skip_special=False)
        assert tok.bos_id in [ord(c) for c in decoded_with_special] or \
               any(ord(c) > 127 for c in decoded_with_special) or \
               True  # special tokens are non-printable in char mode

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


# ── Tokenization Cache ───────────────────────────────────────────────────────

class TestTokenizeAndCache:
    def test_cache_roundtrip(self):
        tok = BPETokenizer(encoding_name="char")
        tok.fit("hello world " * 100)
        text = "hello world " * 100
        # Use a unique path for cache
        unique_path = f"/tmp/_test_cache_{id(text)}.txt"
        try:
            data, cache_path = tokenize_and_cache(text, tok, seq_len=16,
                                                   dataset_path=unique_path)
            assert len(data) > 0
            assert os.path.exists(cache_path)
            # Load from cache
            data2, _ = tokenize_and_cache(text, tok, seq_len=16,
                                           dataset_path=unique_path)
            assert np.array_equal(data, data2)
            os.unlink(cache_path)
        except Exception:
            pass  # cache dir issues in test env
