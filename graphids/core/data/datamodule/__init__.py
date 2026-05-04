"""DataModules for graph and fusion datasets.

``GraphDataModule`` accepts a ``dataset_cls`` class-path string so adding
a new graph domain is config-only — no subclass needed.
``CurriculumDataModule`` subclasses it for tier-bucketed training.

Public API re-exports are kept flat so jsonnet ``class_path`` strings
resolve via ``graphids.core.data.datamodule.X``.
"""

from __future__ import annotations

from .curriculum import CurriculumDataModule
from .fusion import FusionDataModule
from .graph import GraphDataModule

__all__ = [
    "CurriculumDataModule",
    "FusionDataModule",
    "GraphDataModule",
]
