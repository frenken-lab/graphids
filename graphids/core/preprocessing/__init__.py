"""Preprocessing module for CAN-Graph.

Architecture (Phase 3):
    schema.py       — IR column layout and validation
    vocabulary.py   — EntityVocabulary (ID ↔ dense index)
    engine.py       — Domain-agnostic graph construction (vectorized)
    dataset.py      — GraphDataset wrapper
    parallel.py     — Ray-parallel preprocessing driver
    adapters/       — Domain-specific adapters (CAN bus, network flow)
"""

from .dataset import GraphDataset
from .engine import GraphEngine
from .parallel import process_dataset
from .schema import CAN_BUS_SCHEMA, IRSchema
from .vocabulary import EntityVocabulary

__all__ = [
    "GraphDataset",
    "IRSchema",
    "CAN_BUS_SCHEMA",
    "EntityVocabulary",
    "GraphEngine",
    "process_dataset",
]
