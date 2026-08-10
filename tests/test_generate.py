"""
Μῆτις (Metis) — Unit Tests for Generation & Chat
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from metis.data import BPETokenizer, CharTokenizer
from metis.generate import generate_text
from metis.model import MetisLM
from tests.test_model import make_config

# ── Skip tests if not enough VRAM
_CI = os.environ.get("CI", "") == "true"


class TestGenerateText:
    @pytest.fixture
    def model_and_tokenizer(self):
        config = make_config(d_model=64, n_layers=2, max_seq_len=64, vocab_size=50)
        model = MetisLM(config)
        model.eval()
        tok = CharTokenizer()
        tok.fit("hello world test prompt generation abcdefghij " * 10)
        config.vocab_size = tok.vocab_size
        # Re-create model with correct vocab size
        config2 = make_config(d_model=64, n_layers=2, max_seq_len=64,
                               vocab_size=tok.vocab_size)
        model2 = MetisLM(config2)
        model2.eval()
        return model2, tok

    def test_generate_basic(self, model_and_tokenizer):
        model, tok = model_and_tokenizer
        output = generate_text(model, tok, "hello", max_new_tokens=10,
                               temperature=0.8, device="cpu")
        assert isinstance(output, str)
        assert len(output) > 0

    def test_generate_greedy(self, model_and_tokenizer):
        model, tok = model_and_tokenizer
        output = generate_text(model, tok, "hello", max_new_tokens=5,
                               temperature=0.0, device="cpu")
        assert isinstance(output, str)
        assert len(output) >= 5

    def test_generate_with_top_k_top_p(self, model_and_tokenizer):
        model, tok = model_and_tokenizer
        output = generate_text(model, tok, "test", max_new_tokens=10,
                               temperature=0.8, top_k=10, top_p=0.9,
                               device="cpu")
        assert isinstance(output, str)

    def test_generate_with_repetition_penalty(self, model_and_tokenizer):
        model, tok = model_and_tokenizer
        output = generate_text(model, tok, "hello", max_new_tokens=10,
                               temperature=0.8, repetition_penalty=1.2,
                               device="cpu")
        assert isinstance(output, str)

    def test_generate_with_stop_token(self, model_and_tokenizer):
        model, tok = model_and_tokenizer
        output = generate_text(model, tok, "hello", max_new_tokens=50,
                               temperature=0.8, stop_token_id=tok.eos_id,
                               device="cpu")
        assert isinstance(output, str)

    def test_generate_no_kv_cache(self, model_and_tokenizer):
        model, tok = model_and_tokenizer
        output = generate_text(model, tok, "hello", max_new_tokens=10,
                               temperature=0.8, use_kv_cache=False,
                               device="cpu")
        assert isinstance(output, str)

    def test_generate_stream_callback(self, model_and_tokenizer):
        model, tok = model_and_tokenizer
        tokens = []
        output = generate_text(model, tok, "hello", max_new_tokens=10,
                               temperature=0.8, device="cpu",
                               stream_callback=lambda t: tokens.append(t))
        assert isinstance(output, str)
        assert len(tokens) > 0

    def test_generate_with_bpe_tokenizer(self):
        """Test generation with BPE tokenizer."""
        tok = BPETokenizer(encoding_name="char")
        tok.fit("hello world test generation " * 20)
        config = make_config(d_model=64, n_layers=2, max_seq_len=64,
                              vocab_size=tok.vocab_size)
        model = MetisLM(config)
        model.eval()
        output = generate_text(model, tok, "hello", max_new_tokens=10,
                               temperature=0.8, device="cpu")
        assert isinstance(output, str)


class TestLoadModelAndTokenizer:
    def test_load_nonexistent_dir(self):
        from metis.generate import load_model_and_tokenizer
        with pytest.raises(SystemExit):
            load_model_and_tokenizer("/nonexistent/path")


class TestChatFunction:
    def test_chat_imports(self):
        from metis.generate import chat
        assert callable(chat)

    def test_newline_token_id_char(self):
        """CharTokenizer exposes stoi directly."""
        from metis.generate import _newline_token_id
        tok = CharTokenizer()
        tok.fit("hello\nworld")
        assert _newline_token_id(tok) == tok.stoi["\n"]

    def test_newline_token_id_bpe_no_stoi(self):
        """Regression: chat turn-stop must not crash on a BPE tokenizer.

        A ``BPETokenizer`` has no ``stoi`` (its vocab is tiktoken's) — the old
        code did ``tokenizer.stoi["\\n"]`` and crashed. ``_newline_token_id``
        must fall back to encoding ``"\\n"``.
        """
        from metis.generate import _newline_token_id
        tok = BPETokenizer(encoding_name="gpt2")
        assert not hasattr(tok, "stoi")
        nid = _newline_token_id(tok)
        assert nid is not None
        assert tok.decode([nid]) == "\n"
