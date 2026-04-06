"""Data module: datasets, datamodules, sampling, and preprocessing.

Public API:
    from graphids.core.data import CANBusDataset
    ds = CANBusDataset(root="cache/set_01", raw_dir="data/train", split="train")
"""

from __future__ import annotations

from .datamodule import (
    FusionDataModule,
    GraphDataModule,
)
from .datasets import ATTACK_TYPE_CODES, ATTACK_TYPE_NAMES, CANBusDataset

__all__ = [
    "GraphDataModule",
    "CANBusDataset",
    "FusionDataModule",
    "ATTACK_TYPE_CODES",
    "ATTACK_TYPE_NAMES",
]
