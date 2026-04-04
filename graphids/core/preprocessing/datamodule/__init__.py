"""LightningDataModules for graph datasets.

Function + context layout: this package is the Function (DataModule role);
each submodule is a context (graph base, CAN-bus binding, curriculum
scheduling, fusion state caching). Adding a new graph dataset is a single
new sibling file next to ``can_bus.py``.

Public API re-exports are kept flat so external imports and YAML
``class_path`` strings resolve via ``graphids.core.preprocessing.datamodule.X``
without knowing which submodule hosts ``X``.
"""

from __future__ import annotations

# make_graph_loader is re-exported for backward compatibility — several
# call sites still import it from this package path instead of from
# ``graphids.core.preprocessing.sampler`` where it actually lives.
from graphids.core.preprocessing.sampler import make_graph_loader

from .can_bus import CANBusDataModule
from .curriculum import CurriculumDataModule
from .fusion import FusionDataModule
from .graph import GraphDataModule, load_datasets

__all__ = [
    "GraphDataModule",
    "CANBusDataModule",
    "CurriculumDataModule",
    "FusionDataModule",
    "load_datasets",
    "make_graph_loader",
]
