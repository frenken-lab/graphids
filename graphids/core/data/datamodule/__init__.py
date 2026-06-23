"""DataModule primitives for graph and fusion datasets."""

from __future__ import annotations

from .fusion import FusionDataModule
from .graph import GraphDataModule

__all__ = [
    "FusionDataModule",
    "GraphDataModule",
]
