"""Monarch-based single-allocation pipeline orchestration.

Runs all 3 GraphIDS stages (autoencoder, supervised, fusion) inside one
SLURM allocation via PyTorch Monarch actors. Requires ``monarch >= 0.4.0``.

No torch imports at module level -- safe on login nodes.
"""

from __future__ import annotations


def available() -> bool:
    """Return True if the monarch package is importable."""
    try:
        import monarch  # noqa: F401

        return True
    except ImportError:
        return False
