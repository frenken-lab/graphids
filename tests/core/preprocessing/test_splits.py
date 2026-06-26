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
from graphids.core.data.preprocessing.splits import (
    graph_touched_base_units,
    split_embargo_width,
    split_graph_indices,
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


def _touched_units(touched: list[tuple[int, ...]], idx: torch.Tensor) -> set[int]:
    return {unit for graph_idx in idx.tolist() for unit in touched[graph_idx]}


def _raw_intervals(data: Data, idx: torch.Tensor) -> list[tuple[int, int]]:
    return [
        (int(data.window_start_row[graph_idx]), int(data.window_end_row[graph_idx]))
        for graph_idx in idx.tolist()
    ]


def _has_interval_overlap(left: list[tuple[int, int]], right: list[tuple[int, int]]) -> bool:
    return any(start < right_end and right_start < end for start, end in left for right_start, right_end in right)


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
        representation_cfg=cfg,
    )
    data, slices, _, _ = graph_tables_to_pyg(
        tables,
        node_col_order=NODE_COL_ORDER,
        edge_col_order=EDGE_COL_ORDER,
        label_exprs=LABEL_EXPRS,
    )

    train_idx, val_idx = split_graph_indices(data, slices, cfg, val_fraction=0.2)
    touched = graph_touched_base_units(data, slices)
    train_units = _touched_units(touched, train_idx)
    val_units = _touched_units(touched, val_idx)

    assert split_embargo_width(cfg) == 2
    assert len(train_idx) > 0
    assert len(val_idx) > 0
    assert train_units.isdisjoint(val_units)
    assert not _has_interval_overlap(_raw_intervals(data, train_idx), _raw_intervals(data, val_idx))


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

    train_idx, val_idx = split_graph_indices(data, slices, cfg, val_fraction=0.2)

    assert split_embargo_width(cfg) == 1
    assert train_idx.tolist() == list(range(7))
    assert val_idx.tolist() == [8, 9]


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

    train_idx, val_idx = split_graph_indices(data, slices, cfg, val_fraction=0.3)

    assert train_idx.tolist() == list(range(7))
    assert val_idx.tolist() == [7, 8, 9]


def test_overlapping_raw_windows_are_separated_by_embargo():
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

    train_idx, val_idx = split_graph_indices(data, slices, cfg, val_fraction=0.5)

    assert not _has_interval_overlap(_raw_intervals(data, train_idx), _raw_intervals(data, val_idx))


def test_split_indices_are_disjoint_by_graph_and_base_unit():
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

    train_idx, val_idx = split_graph_indices(data, slices, cfg, val_fraction=0.5)
    touched = graph_touched_base_units(data, slices)

    assert set(train_idx.tolist()).isdisjoint(val_idx.tolist())
    assert _touched_units(touched, train_idx).isdisjoint(_touched_units(touched, val_idx))
