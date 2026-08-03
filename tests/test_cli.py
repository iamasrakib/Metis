"""
Μῆτις (Metis) — Unit Tests for the CLI
"""

import io
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from metis.cli import BANNER, _configure_stdio


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
