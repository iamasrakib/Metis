#!/usr/bin/env python
"""
Thin shim — preserves the classic ``python generate.py ...`` entry point.

All real logic lives in the ``metis`` package. Without a ``--prompt`` flag this
dispatches to ``metis chat`` (matching the old default); with ``--prompt`` it
dispatches to ``metis generate``.
"""
import sys

from metis.cli import main

if "--prompt" in sys.argv or "-h" in sys.argv or "--help" in sys.argv:
    command = "generate"
else:
    # Old default behaviour: chat when no prompt supplied. Pass through all
    # flags (chat accepts the common --checkpoint-dir/--device/etc. options).
    command = "chat"

if __name__ == "__main__":
    raise SystemExit(main([command, *sys.argv[1:]]))
