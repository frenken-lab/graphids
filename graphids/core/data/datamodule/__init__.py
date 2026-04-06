"""LightningDataModules for graph datasets.

Each submodule is a context (graph base, CAN-bus binding, fusion state
caching). Adding a new graph dataset is a single new sibling file next to
``can_bus.py``. Curriculum learning is a ``sampler="curriculum"`` toggle on
``GraphDataModule``, not a separate DataModule class.

Public API re-exports are kept flat so external imports and jsonnet
``class_path`` strings resolve via ``graphids.core.data.datamodule.X``
without knowing which submodule hosts ``X``.
"""

from __future__ import annotations

from graphids.core.data.sampler import make_graph_loader

from .can_bus import CANBusDataModule
from .fusion import FusionDataModule
from .graph import GraphDataModule, load_datasets

__all__ = [
    "CANBusDataModule",
    "FusionDataModule",
    "GraphDataModule",
    "load_datasets",
    "make_graph_loader",
]
