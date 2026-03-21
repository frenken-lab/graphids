"""Preprocessing module.

Public API:
    from graphids.core.preprocessing import CANBusDataset
    ds = CANBusDataset(root="cache/set_01", raw_dir="data/train", split="train")

Re-exports for convenience:
    get_batch_index, graph_attack_type   — graph utilities
    ATTACK_TYPE_CODES, ATTACK_TYPE_NAMES — CAN bus attack mappings
    TemporalGrouper, GraphSequence       — temporal grouping
"""

from __future__ import annotations

from ._graph_utils import get_batch_index, graph_attack_type
from ._temporal import GraphSequence, TemporalGrouper
from .datamodule import CANBusDataModule
from .datasets import ATTACK_TYPE_CODES, ATTACK_TYPE_NAMES, CANBusDataset

__all__ = [
    "CANBusDataModule",
    "CANBusDataset",
    "ATTACK_TYPE_CODES",
    "ATTACK_TYPE_NAMES",
    "get_batch_index",
    "graph_attack_type",
    "TemporalGrouper",
    "GraphSequence",
]
