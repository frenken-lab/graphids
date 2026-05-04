"""DataModules for graph and fusion datasets.

``GraphDataModule`` accepts a ``dataset_cls`` class-path string so adding
a new graph domain is config-only — no subclass needed. Curriculum
learning is implemented at the loss end (``CurriculumWeightedLoss``)
reading ``batch.difficulty`` / ``batch.in_scope`` attached by
``GraphDataModule.setup`` when a ``difficulty`` config is set; no
curriculum-specific datamodule subclass exists.

Public API re-exports are kept flat so jsonnet ``class_path`` strings
resolve via ``graphids.core.data.datamodule.X``.
"""

from __future__ import annotations

from .fusion import FusionDataModule
from .graph import GraphDataModule

__all__ = [
    "FusionDataModule",
    "GraphDataModule",
]
