"""Data module: datasets, datamodules, sampling, and preprocessing.

Public API:
    from graphids.core.data import CANBusSource
    source = CANBusSource(name="hcrl_sa")  # preferred: source→build→DatasetState
    state = source.build()                 # train/val/test_* split scoping done here
"""

from __future__ import annotations

from .datamodule import (
    FusionDataModule,
    GraphDataModule,
)
from .datasets import ATTACK_TYPE_CODES, ATTACK_TYPE_NAMES, CANBusDataset, CANBusSource
from .state import DatasetState, clear_cache, get_or_build

__all__ = [
    "GraphDataModule",
    "CANBusDataset",
    "CANBusSource",
    "FusionDataModule",
    "DatasetState",
    "get_or_build",
    "clear_cache",
    "ATTACK_TYPE_CODES",
    "ATTACK_TYPE_NAMES",
]
