"""LightningDataModules for graph datasets.

``GraphDataModule`` accepts a ``dataset_cls`` class-path string so adding
a new graph domain is config-only — no subclass needed. Curriculum learning
is a ``sampler="curriculum"`` toggle, not a separate DataModule class.

Public API re-exports are kept flat so jsonnet ``class_path`` strings
resolve via ``graphids.core.data.datamodule.X``.
"""

from __future__ import annotations

from graphids.core.data.sampler import make_graph_loader

from .fusion import FusionDataModule
from .graph import GraphDataModule, load_datasets

__all__ = [
    "FusionDataModule",
    "GraphDataModule",
    "load_datasets",
    "make_graph_loader",
]
