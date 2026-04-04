"""Preprocessing module.

Public API:
    from graphids.core.preprocessing import CANBusDataset
    ds = CANBusDataset(root="cache/set_01", raw_dir="data/train", split="train")
"""

from __future__ import annotations

from .datamodule import (
    CANBusDataModule,
    CurriculumDataModule,
    FusionDataModule,
    GraphDataModule,
)
from .datasets import ATTACK_TYPE_CODES, ATTACK_TYPE_NAMES, CANBusDataset

__all__ = [
    "GraphDataModule",
    "CANBusDataModule",
    "CANBusDataset",
    "CurriculumDataModule",
    "FusionDataModule",
    "ATTACK_TYPE_CODES",
    "ATTACK_TYPE_NAMES",
]
