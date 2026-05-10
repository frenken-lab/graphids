"""Materialization contract tests for alternate representation branches."""

from __future__ import annotations

import numpy as np
import polars as pl

from graphids.core.data.datasets.can_bus import (
    BYTE_COLS,
    EDGE_BASE_COLS,
    EDGE_COL_ORDER,
    EDGE_STAT_EXPRS,
    LABEL_EXPRS,
    NODE_COL_ORDER,
    NODE_STAT_EXPRS,
)
from graphids.core.data.preprocessing.materialization import build_graph_tables
from graphids.core.data.preprocessing.segments import (
    EntitySegmentCfg,
    MultiScaleSegmentCfg,
    SequenceSegmentCfg,
)


def _frame(n_rows: int = 20) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "timestamp": np.arange(n_rows, dtype=np.float64),
            "node_id": pl.Series(([0, 1] * (n_rows // 2)), dtype=pl.Int64),
            **{c: np.zeros(n_rows, dtype=np.float32) for c in BYTE_COLS},
            "entropy": np.zeros(n_rows, dtype=np.float32),
            "attack": [0] * n_rows,
            "attack_type": [0] * n_rows,
        }
    )


def test_sequence_branch_tags_materialized_tables():
    tables = build_graph_tables(
        _frame(),
        node_stat_exprs=NODE_STAT_EXPRS,
        label_exprs=LABEL_EXPRS,
        edge_stat_exprs=EDGE_STAT_EXPRS,
        edge_base_cols=EDGE_BASE_COLS,
        edge_policy=None,
        graph_transforms=None,
        debug_artifacts_dir=None,
        segment_cfg=SequenceSegmentCfg(window_size=5, stride=5, sequence_length=3, sequence_stride=1),
    )
    assert "sequence_length" in tables.node_stats.columns
    assert "sequence_length" in tables.edge_df.columns
    assert "sequence_length" in tables.labels.columns


def test_multiscale_branch_tags_materialized_tables():
    tables = build_graph_tables(
        _frame(),
        node_stat_exprs=NODE_STAT_EXPRS,
        label_exprs=LABEL_EXPRS,
        edge_stat_exprs=EDGE_STAT_EXPRS,
        edge_base_cols=EDGE_BASE_COLS,
        edge_policy=None,
        graph_transforms=None,
        debug_artifacts_dir=None,
        segment_cfg=MultiScaleSegmentCfg(window_sizes=(5, 10), stride=5),
    )
    assert "scale_id" in tables.node_stats.columns
    assert "scale_window_size" in tables.node_stats.columns
    assert tables.node_stats.select("scale_id").n_unique() == 2


def test_entity_branch_tags_materialized_tables():
    tables = build_graph_tables(
        _frame(),
        node_stat_exprs=NODE_STAT_EXPRS,
        label_exprs=LABEL_EXPRS,
        edge_stat_exprs=EDGE_STAT_EXPRS,
        edge_base_cols=EDGE_BASE_COLS,
        edge_policy=None,
        graph_transforms=None,
        debug_artifacts_dir=None,
        segment_cfg=EntitySegmentCfg(
            anchor_column="node_id",
            anchor_value=0,
            history_window_size=5,
            future_window_size=0,
        ),
    )
    assert "anchor_column" in tables.node_stats.columns
    assert "anchor_value" in tables.node_stats.columns
