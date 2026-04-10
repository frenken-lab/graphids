"""Data module: datasets, datamodules, sampling, and preprocessing.

Public API:
    from graphids.core.data import CANBusDataset
    ds = CANBusDataset(root="cache/set_01", raw_dir="data/train", split="train")
"""

from __future__ import annotations

from .cache import DatasetState, clear_cache, get_or_build
from .datamodule import (
    FusionDataModule,
    GraphDataModule,
)
from .datasets import ATTACK_TYPE_CODES, ATTACK_TYPE_NAMES, CANBusDataset, CANBusSource

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
