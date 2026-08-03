#!/usr/bin/env python
"""
Thin shim — preserves the classic ``python train.py ...`` entry point.

All real logic lives in the ``metis`` package. This simply dispatches to
``metis train`` for backward compatibility.
"""
import sys

from metis.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["train", *sys.argv[1:]]))
