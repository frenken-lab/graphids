"""Core graph preprocessing."""

from .materialization import GraphTables, build_graph_tables
from .pyg import graph_tables_to_pyg
from .representations import (
    GraphRepresentationCfg,
    SnapshotRepresentationCfg,
    SnapshotSequenceRepresentationCfg,
    TemporalRepresentationCfg,
    representation_digest,
    representation_kind,
    representation_payload,
    representation_window_defaults,
)
from .temporal import (
    add_temporal_split_masks,
    assert_temporal_splits_disjoint,
    build_temporal_event_table,
    prepare_temporal_eval_table,
    split_temporal_train_val_tables,
    temporal_to_pyg,
)

__all__ = [
    "GraphTables",
    "build_graph_tables",
    "graph_tables_to_pyg",
    "GraphRepresentationCfg",
    "SnapshotRepresentationCfg",
    "SnapshotSequenceRepresentationCfg",
    "TemporalRepresentationCfg",
    "representation_digest",
    "representation_kind",
    "representation_payload",
    "representation_window_defaults",
    "add_temporal_split_masks",
    "assert_temporal_splits_disjoint",
    "build_temporal_event_table",
    "prepare_temporal_eval_table",
    "split_temporal_train_val_tables",
    "temporal_to_pyg",
]
