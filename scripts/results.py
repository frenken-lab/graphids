#!/usr/bin/env python3
"""Compatibility wrapper for the canonical result query CLI."""

from __future__ import annotations

import sys

from graphids.__main__ import main

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "exp", "results", *sys.argv[1:]]
    main()
