#!/usr/bin/env python
"""
Thin shim → ``metis chat``. Interactive streaming chat with a trained model.
"""
import sys

from metis.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["chat", *sys.argv[1:]]))
