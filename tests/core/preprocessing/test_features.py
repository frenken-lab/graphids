"""Core graph-table feature invariant."""

from __future__ import annotations

import numpy as np
import polars as pl
import torch

from graphids.core.data.datasets.can_bus import (
    EDGE_BASE_COLS,
    EDGE_COL_ORDER,
    EDGE_STAT_EXPRS,
    LABEL_EXPRS,
    N_EDGE_FEATURES,
    N_NODE_FEATURES,
    NODE_COL_ORDER,
    NODE_STAT_EXPRS,
)
from graphids.core.data.preprocessing.materialization import build_graph_tables
from graphids.core.data.preprocessing.pyg import graph_tables_to_pyg
from graphids.core.data.preprocessing.representations import SnapshotRepresentationCfg


def test_graph_table_pipeline_preserves_core_feature_contract():
    n_rows = 50
    byte_data = {f"byte_{idx}": np.zeros(n_rows, dtype=np.float32) for idx in range(8)}
    byte_data["byte_0"][0] = 255.0
    df = pl.DataFrame(
        {
            "timestamp": np.arange(n_rows, dtype=np.float64),
            "node_id": pl.Series(([0, 1] * (n_rows // 2)), dtype=pl.Int64),
            **byte_data,
            "entropy": np.zeros(n_rows, dtype=np.float32),
            "attack": [0] * n_rows,
            "attack_type": [0] * n_rows,
        }
    )

    tables = build_graph_tables(
        df,
        node_stat_exprs=NODE_STAT_EXPRS,
        label_exprs=LABEL_EXPRS,
        edge_stat_exprs=EDGE_STAT_EXPRS,
        edge_base_cols=EDGE_BASE_COLS,
        representation_cfg=SnapshotRepresentationCfg(window_size=n_rows, stride=n_rows),
    )
    data, slices, num_graphs, n_raw_rows = graph_tables_to_pyg(
        tables,
        node_col_order=NODE_COL_ORDER,
        edge_col_order=EDGE_COL_ORDER,
        label_exprs=LABEL_EXPRS,
    )

    assert n_raw_rows == n_rows
    assert num_graphs == 1
    assert len(tables.node_stats) > 0
    assert len(tables.edge_df) > 0
    assert data.x.shape[1] == N_NODE_FEATURES
    assert data.edge_attr.shape[1] == N_EDGE_FEATURES
    assert data.edge_index.shape[0] == 2
    assert data.y.tolist() == [0]
    assert data.node_id.shape[0] == data.x.shape[0]
    assert data.edge_index.max() < data.x.shape[0]
    assert not torch.isnan(data.x).any()
    assert not torch.isnan(data.edge_attr).any()

    graph_edge_start, graph_edge_end = slices["edge_attr"][0], slices["edge_attr"][1]
    edge_attr = data.edge_attr[graph_edge_start:graph_edge_end]
    assert edge_attr[:, EDGE_COL_ORDER.index("edge_freq")].max() > 1
    assert data.x[:, NODE_COL_ORDER.index("skewness")].abs().max() <= 10.0
    assert data.x[:, NODE_COL_ORDER.index("kurtosis")].abs().max() <= 10.0
