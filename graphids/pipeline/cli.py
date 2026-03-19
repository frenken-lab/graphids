"""Backward-compat shim — CLI moved to graphids.cli."""

from graphids.cli import main

__all__ = ["main"]

if __name__ == "__main__":
    main()
