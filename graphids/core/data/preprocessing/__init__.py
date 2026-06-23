"""Core graph preprocessing."""

from .materialization import GraphTables, build_graph_tables
from .pyg import graph_tables_to_pyg
from .representations import (
    GraphRepresentationCfg,
    SnapshotRepresentationCfg,
    SnapshotSequenceRepresentationCfg,
    representation_digest,
    representation_kind,
    representation_payload,
    representation_window_defaults,
)

__all__ = [
    "GraphTables",
    "build_graph_tables",
    "graph_tables_to_pyg",
    "GraphRepresentationCfg",
    "SnapshotRepresentationCfg",
    "SnapshotSequenceRepresentationCfg",
    "representation_digest",
    "representation_kind",
    "representation_payload",
    "representation_window_defaults",
]
