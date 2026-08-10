"""
Μῆτις (Metis) — Unit Tests for the CLI
"""

import io
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from metis.cli import BANNER, _build_train_config, _configure_stdio, build_parser


def _cp1252_stream() -> io.TextIOWrapper:
    """A text stream that uses the Windows cp1252 codepage (like a legacy console)."""
    return io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict")


class TestConfigureStdio:
    def test_reconfigures_legacy_console_encoding(self):
        """cp1252 streams can't encode the banner (Greek + box-drawing) — after
        _configure_stdio the banner must print without UnicodeEncodeError."""
        stream = _cp1252_stream()
        _configure_stdio([stream])

        assert stream.encoding == "utf-8"
        # Printing the banner must not raise (regression: UnicodeEncodeError
        # on Windows cp1252 consoles crashed `metis info` / `train` / `find-lr`).
        stream.write(BANNER)
        stream.flush()

    def test_handles_streams_without_reconfigure(self):
        """Streams lacking .reconfigure (embedded contexts) are left untouched."""
        class DumbStream:
            pass
        dumb = DumbStream()  # no .reconfigure attribute
        _configure_stdio([dumb])  # must not raise


class TestTrainConfig:
    def test_train_optimizer_default(self):
        parser = build_parser()
        args = parser.parse_args(["train"])
        config = _build_train_config(args)
        assert config.optimizer == "adamw"

    def test_train_optimizer_bnb8bit(self):
        parser = build_parser()
        args = parser.parse_args(["train", "--preset", "1b",
                                  "--optimizer", "bnb8bit"])
        config = _build_train_config(args)
        assert config.optimizer == "bnb8bit"

    def test_train_grad_accum_wiring(self):
        parser = build_parser()
        args = parser.parse_args(["train", "--grad-accum", "16"])
        config = _build_train_config(args)
        assert config.gradient_accumulation_steps == 16

    def test_train_no_cuda_graphs_wiring(self):
        parser = build_parser()
        args = parser.parse_args(["train", "--no-cuda-graphs"])
        config = _build_train_config(args)
        assert config.use_cuda_graphs is False

    def test_train_optimizer_invalid_choice(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["train", "--optimizer", "bogus"])
