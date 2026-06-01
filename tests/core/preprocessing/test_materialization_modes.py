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
from graphids.core.data.preprocessing.pyg import graph_tables_to_pyg
from graphids.core.data.preprocessing.segments import (
    EntitySegmentCfg,
    MultiScaleSegmentCfg,
    SequenceSegmentCfg,
    WindowSegmentCfg,
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
    assert tables.node_stats.select("_wid").n_unique() == 2
    assert tables.labels.height == 2
    assert set(tables.node_stats.select("sequence_step").unique()["sequence_step"].to_list()) == {
        0,
        1,
        2,
    }
    assert "snapshot_wid" in tables.node_stats.columns
    assert "window_start_row" in tables.labels.columns
    assert "window_end_row" in tables.labels.columns
    assert tables.labels["window_start_row"].to_list() == [0, 5]
    assert tables.labels["window_end_row"].to_list() == [15, 20]
    assert "sequence_length" in tables.node_stats.columns
    assert "sequence_length" in tables.edge_df.columns
    assert "sequence_length" in tables.labels.columns
    assert "sequence_stride" in tables.labels.columns

    data, slices, num_graphs, _ = graph_tables_to_pyg(
        tables,
        node_col_order=NODE_COL_ORDER,
        edge_col_order=EDGE_COL_ORDER,
        label_exprs=LABEL_EXPRS,
    )
    assert num_graphs == 2
    assert data.sequence_id.tolist() == [0, 1]
    assert data.sequence_length.tolist() == [3, 3]
    assert data.sequence_stride.tolist() == [1, 1]
    assert data.target_snapshot_wid.tolist() == [10, 15]
    assert data.window_start_row.tolist() == [0, 5]
    assert data.window_end_row.tolist() == [15, 20]
    first_start, first_end = slices["node_sequence_step"][0], slices["node_sequence_step"][1]
    assert set(data.node_sequence_step[first_start:first_end].tolist()) == {0, 1, 2}


def test_sequence_labels_use_target_window_not_any_context_window():
    context_only_attack = _frame().with_columns(
        pl.when(pl.arange(0, pl.len()) < 5).then(pl.lit(1)).otherwise(pl.lit(0)).alias("attack")
    )
    tables = build_graph_tables(
        context_only_attack,
        node_stat_exprs=NODE_STAT_EXPRS,
        label_exprs=LABEL_EXPRS,
        edge_stat_exprs=EDGE_STAT_EXPRS,
        edge_base_cols=EDGE_BASE_COLS,
        edge_policy=None,
        graph_transforms=None,
        debug_artifacts_dir=None,
        segment_cfg=SequenceSegmentCfg(window_size=5, stride=5, sequence_length=3, sequence_stride=1),
    )
    assert tables.labels["y"].to_list() == [0, 0]

    target_attack = _frame().with_columns(
        pl.when((pl.arange(0, pl.len()) >= 10) & (pl.arange(0, pl.len()) < 15))
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias("attack")
    )
    tables = build_graph_tables(
        target_attack,
        node_stat_exprs=NODE_STAT_EXPRS,
        label_exprs=LABEL_EXPRS,
        edge_stat_exprs=EDGE_STAT_EXPRS,
        edge_base_cols=EDGE_BASE_COLS,
        edge_policy=None,
        graph_transforms=None,
        debug_artifacts_dir=None,
        segment_cfg=SequenceSegmentCfg(window_size=5, stride=5, sequence_length=3, sequence_stride=1),
    )
    assert tables.labels["y"].to_list() == [1, 0]


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


def test_window_metadata_tracks_source_boundary_counts():
    frame = _frame().with_columns(
        pl.when(pl.arange(0, pl.len()) < 10)
        .then(pl.lit("a"))
        .otherwise(pl.lit("b"))
        .alias("source_file"),
        pl.lit("train").alias("source_dir"),
    )
    tables = build_graph_tables(
        frame,
        node_stat_exprs=NODE_STAT_EXPRS,
        label_exprs=LABEL_EXPRS,
        edge_stat_exprs=EDGE_STAT_EXPRS,
        edge_base_cols=EDGE_BASE_COLS,
        edge_policy=None,
        graph_transforms=None,
        debug_artifacts_dir=None,
        segment_cfg=WindowSegmentCfg(window_size=20, stride=20),
    )

    assert tables.labels["source_dir_n_unique"].to_list() == [1]
    assert tables.labels["source_file_n_unique"].to_list() == [2]


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
