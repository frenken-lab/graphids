"""Core temporal preprocessing."""

from .representations import (
    RepresentationCfg,
    TemporalRepresentationCfg,
    representation_digest,
    representation_kind,
    representation_payload,
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
    "RepresentationCfg",
    "TemporalRepresentationCfg",
    "representation_digest",
    "representation_kind",
    "representation_payload",
    "add_temporal_split_masks",
    "assert_temporal_splits_disjoint",
    "build_temporal_event_table",
    "prepare_temporal_eval_table",
    "split_temporal_train_val_tables",
    "temporal_to_pyg",
]
