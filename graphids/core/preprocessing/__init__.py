"""Preprocessing module.

Public API:
    from graphids.core.preprocessing import CANBusDataset
    ds = CANBusDataset(root="cache/set_01", raw_dir="data/train", split="train")

Re-exports for convenience:
    get_batch_index, graph_attack_type   — graph utilities
    ATTACK_TYPE_CODES, ATTACK_TYPE_NAMES — CAN bus attack mappings
    TemporalGrouper, GraphSequence       — temporal grouping
    TemporalGraphDataset, collate_temporal — temporal dataset/collate
"""

from __future__ import annotations

from ._graph_utils import get_batch_index, graph_attack_type, graph_label
from ._temporal import GraphSequence, TemporalDataModule, TemporalGraphDataset, TemporalGrouper, collate_temporal
from .datamodule import CANBusDataModule, FusionDataModule, cache_predictions
from .datasets import ATTACK_TYPE_CODES, ATTACK_TYPE_NAMES, CANBusDataset

__all__ = [
    "CANBusDataModule",
    "CANBusDataset",
    "FusionDataModule",
    "TemporalDataModule",
    "cache_predictions",
    "ATTACK_TYPE_CODES",
    "ATTACK_TYPE_NAMES",
    "get_batch_index",
    "graph_attack_type",
    "graph_label",
    "TemporalGrouper",
    "GraphSequence",
    "TemporalGraphDataset",
    "collate_temporal",
]
