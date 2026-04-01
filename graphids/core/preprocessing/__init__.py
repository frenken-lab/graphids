"""Preprocessing module.

Public API:
    from graphids.core.preprocessing import CANBusDataset
    ds = CANBusDataset(root="cache/set_01", raw_dir="data/train", split="train")

Re-exports for convenience:
    ATTACK_TYPE_CODES, ATTACK_TYPE_NAMES — CAN bus attack mappings
    TemporalDataModule, TemporalGrouper, etc. — lazy-imported (temporal disabled by default)
"""

from __future__ import annotations

from .datamodule import CANBusDataModule, FusionDataModule
from .datasets import ATTACK_TYPE_CODES, ATTACK_TYPE_NAMES, CANBusDataset

__all__ = [
    "CANBusDataModule",
    "CANBusDataset",
    "FusionDataModule",
    "TemporalDataModule",
    "ATTACK_TYPE_CODES",
    "ATTACK_TYPE_NAMES",
    "TemporalGrouper",
    "GraphSequence",
    "TemporalGraphDataset",
    "collate_temporal",
]


def __getattr__(name: str):
    """Lazy-load temporal symbols — avoids pulling in Lightning on every import."""
    _temporal_names = {"GraphSequence", "TemporalDataModule", "TemporalGraphDataset", "TemporalGrouper", "collate_temporal"}
    if name in _temporal_names:
        from .stages import temporal
        return getattr(temporal, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
