"""Regression tests for representation-aware split planning."""

from __future__ import annotations

import numpy as np
import polars as pl
import torch
from torch_geometric.data import Data

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
from graphids.core.data.preprocessing.representations import (
    SnapshotRepresentationCfg,
    SnapshotSequenceRepresentationCfg,
)
from graphids.core.data.preprocessing.segments import SequenceSegmentCfg
from graphids.core.data.preprocessing.splits import (
    audit_split_plan,
    build_blocked_split_plan,
    graph_touched_base_units,
    split_embargo_width,
)


def _frame(n_rows: int = 100) -> pl.DataFrame:
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


def test_snapshot_sequence_split_has_no_underlying_window_overlap():
    cfg = SnapshotSequenceRepresentationCfg(
        window_size=5,
        stride=5,
        sequence_length=3,
        sequence_stride=1,
    )
    tables = build_graph_tables(
        _frame(),
        node_stat_exprs=NODE_STAT_EXPRS,
        label_exprs=LABEL_EXPRS,
        edge_stat_exprs=EDGE_STAT_EXPRS,
        edge_base_cols=EDGE_BASE_COLS,
        edge_policy=None,
        graph_transforms=None,
        debug_artifacts_dir=None,
        segment_cfg=SequenceSegmentCfg(
            window_size=cfg.window_size,
            stride=cfg.stride,
            sequence_length=cfg.sequence_length,
            sequence_stride=cfg.sequence_stride,
        ),
    )
    data, slices, _, _ = graph_tables_to_pyg(
        tables,
        node_col_order=NODE_COL_ORDER,
        edge_col_order=EDGE_COL_ORDER,
        label_exprs=LABEL_EXPRS,
    )

    plan = build_blocked_split_plan(data, slices, cfg, val_fraction=0.2, seed=42)
    touched = graph_touched_base_units(data, slices)
    train_units = {
        unit for graph_idx in plan.train_idx.tolist() for unit in touched[graph_idx]
    }
    val_units = {
        unit for graph_idx in plan.val_idx.tolist() for unit in touched[graph_idx]
    }

    assert plan.embargo_width == 2
    assert len(plan.train_idx) > 0
    assert len(plan.val_idx) > 0
    assert train_units.isdisjoint(val_units)
    assert audit_split_plan(plan)["raw_interval_intersections"] == 0


def test_snapshot_overlap_uses_embargo_when_stride_is_smaller_than_window():
    cfg = SnapshotRepresentationCfg(window_size=10, stride=5)
    data = Data(
        x=torch.zeros((10, 1)),
        edge_index=torch.zeros((2, 10), dtype=torch.long),
        edge_attr=torch.zeros((10, 1)),
        y=torch.zeros(10, dtype=torch.long),
        graph_wid=torch.arange(10, dtype=torch.long),
    )
    slices = {
        "x": torch.arange(11, dtype=torch.long),
        "edge_index": torch.arange(11, dtype=torch.long),
        "edge_attr": torch.arange(11, dtype=torch.long),
        "y": torch.arange(11, dtype=torch.long),
        "graph_wid": torch.arange(11, dtype=torch.long),
    }

    plan = build_blocked_split_plan(data, slices, cfg, val_fraction=0.2, seed=42)

    assert split_embargo_width(cfg) == 1
    assert plan.train_idx.tolist() == list(range(7))
    assert plan.val_idx.tolist() == [8, 9]


def test_validation_uses_tail_block():
    cfg = SnapshotRepresentationCfg(window_size=5, stride=5)
    data = Data(
        x=torch.zeros((10, 1)),
        edge_index=torch.zeros((2, 10), dtype=torch.long),
        edge_attr=torch.zeros((10, 1)),
        y=torch.tensor([0, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=torch.long),
        graph_wid=torch.arange(10, dtype=torch.long),
    )
    slices = {
        "x": torch.arange(11, dtype=torch.long),
        "edge_index": torch.arange(11, dtype=torch.long),
        "edge_attr": torch.arange(11, dtype=torch.long),
        "y": torch.arange(11, dtype=torch.long),
        "graph_wid": torch.arange(11, dtype=torch.long),
    }

    plan = build_blocked_split_plan(data, slices, cfg, val_fraction=0.3, seed=42)

    assert plan.train_idx.tolist() == list(range(7))
    assert plan.val_idx.tolist() == [7, 8, 9]


def test_audit_reports_raw_interval_intersections():
    cfg = SnapshotRepresentationCfg(window_size=10, stride=5)
    data = Data(
        x=torch.zeros((4, 1)),
        edge_index=torch.zeros((2, 4), dtype=torch.long),
        edge_attr=torch.zeros((4, 1)),
        y=torch.zeros(4, dtype=torch.long),
        graph_wid=torch.arange(4, dtype=torch.long),
        window_start_row=torch.tensor([0, 5, 10, 15], dtype=torch.long),
        window_end_row=torch.tensor([10, 15, 20, 25], dtype=torch.long),
    )
    slices = {
        "x": torch.arange(5, dtype=torch.long),
        "edge_index": torch.arange(5, dtype=torch.long),
        "edge_attr": torch.arange(5, dtype=torch.long),
        "y": torch.arange(5, dtype=torch.long),
        "graph_wid": torch.arange(5, dtype=torch.long),
        "window_start_row": torch.arange(5, dtype=torch.long),
        "window_end_row": torch.arange(5, dtype=torch.long),
    }

    plan = build_blocked_split_plan(data, slices, cfg, val_fraction=0.5, seed=42)

    assert audit_split_plan(plan)["raw_interval_intersections"] == 0


def test_split_audit_reports_only_leakage_invariants():
    cfg = SnapshotRepresentationCfg(window_size=5, stride=5)
    data = Data(
        x=torch.zeros((6, 1)),
        edge_index=torch.zeros((2, 6), dtype=torch.long),
        edge_attr=torch.zeros((6, 1)),
        y=torch.zeros(6, dtype=torch.long),
        graph_wid=torch.arange(6, dtype=torch.long),
    )
    slices = {
        "x": torch.arange(7, dtype=torch.long),
        "edge_index": torch.arange(7, dtype=torch.long),
        "edge_attr": torch.arange(7, dtype=torch.long),
        "y": torch.arange(7, dtype=torch.long),
        "graph_wid": torch.arange(7, dtype=torch.long),
    }

    plan = build_blocked_split_plan(data, slices, cfg, val_fraction=0.5, seed=42)
    audit = audit_split_plan(plan)

    assert set(audit) == {
        "graph_index_overlap",
        "base_unit_overlap",
        "raw_interval_intersections",
    }
    assert audit == {
        "graph_index_overlap": 0,
        "base_unit_overlap": 0,
        "raw_interval_intersections": 0,
    }
