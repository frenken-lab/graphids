"""Preprocessing feature tests over the invariant graph-table primitives."""

from __future__ import annotations

import numpy as np
import torch


def _get_graph(data, slices, idx):
    """Extract a single graph from pre-collated (data, slices) format."""
    from torch_geometric.data import Data

    ns, ne = slices["x"][idx], slices["x"][idx + 1]
    es, ee = slices["edge_index"][idx], slices["edge_index"][idx + 1]
    return Data(
        x=data.x[ns:ne],
        edge_index=data.edge_index[:, es:ee],
        edge_attr=data.edge_attr[es:ee],
        node_id=data.node_id[ns:ne],
        y=data.y[idx : idx + 1],
        attack_type=data.attack_type[idx : idx + 1],
    )


def _snapshot(window_size: int):
    from graphids.core.data.preprocessing.representations import (
        SnapshotRepresentationCfg,
    )

    return SnapshotRepresentationCfg(window_size=window_size, stride=window_size)


def _build_tables(df, representation_cfg):
    from graphids.core.data.datasets.can_bus import (
        EDGE_BASE_COLS,
        EDGE_STAT_EXPRS,
        LABEL_EXPRS,
        NODE_STAT_EXPRS,
    )
    from graphids.core.data.preprocessing.materialization import build_graph_tables

    return build_graph_tables(
        df,
        node_stat_exprs=NODE_STAT_EXPRS,
        label_exprs=LABEL_EXPRS,
        edge_stat_exprs=EDGE_STAT_EXPRS,
        edge_base_cols=EDGE_BASE_COLS,
        representation_cfg=representation_cfg,
    )


def _run(df, representation_cfg, *, node_col_order=None):
    from graphids.core.data.datasets.can_bus import (
        EDGE_COL_ORDER,
        LABEL_EXPRS,
        NODE_COL_ORDER,
    )
    from graphids.core.data.preprocessing.pyg import graph_tables_to_pyg

    tables = _build_tables(
        df,
        representation_cfg,
    )
    return graph_tables_to_pyg(
        tables,
        node_col_order=node_col_order or NODE_COL_ORDER,
        edge_col_order=EDGE_COL_ORDER,
        label_exprs=LABEL_EXPRS,
    )


def test_sliding_window_graphs_shapes_and_values():
    import polars as pl

    from graphids.core.data.datasets.can_bus import (
        EDGE_COL_ORDER,
        N_EDGE_FEATURES,
        N_NODE_FEATURES,
    )

    n_rows = 20
    rng = np.random.default_rng(0)
    node_ids = rng.integers(0, 4, n_rows)
    df = pl.DataFrame(
        {
            "timestamp": np.arange(n_rows, dtype=np.float64),
            "node_id": pl.Series(node_ids.tolist(), dtype=pl.Int64),
            **{f"byte_{i}": np.ones(n_rows, dtype=np.float32) * i for i in range(8)},
            "entropy": np.zeros(n_rows, dtype=np.float32),
            "attack": [0] * n_rows,
            "attack_type": [0] * n_rows,
        }
    )

    data, slices, num_graphs, _ = _run(df, _snapshot(n_rows))

    assert num_graphs > 0
    graph = _get_graph(data, slices, 0)
    assert graph.x.shape[1] == N_NODE_FEATURES
    assert graph.edge_attr.shape[1] == N_EDGE_FEATURES
    assert graph.edge_index.shape[0] == 2
    assert graph.y.item() == 0
    assert not torch.isnan(graph.x).any()
    assert not torch.isnan(graph.edge_attr).any()
    assert (graph.edge_attr[:, EDGE_COL_ORDER.index("iat")] == 1.0).all()
    for idx in range(8):
        col_idx = EDGE_COL_ORDER.index(f"byte_{idx}_diff")
        assert (graph.edge_attr[:, col_idx] == 0.0).all()


def test_sliding_window_graphs_edge_freq():
    import polars as pl

    from graphids.core.data.datasets.can_bus import EDGE_COL_ORDER

    df = pl.DataFrame(
        {
            "timestamp": np.arange(10, dtype=np.float64),
            "node_id": pl.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=pl.Int64),
            **{f"byte_{i}": np.zeros(10, dtype=np.float32) for i in range(8)},
            "entropy": np.zeros(10, dtype=np.float32),
            "attack": [0] * 10,
            "attack_type": [0] * 10,
        }
    )

    data, slices, num_graphs, _ = _run(df, _snapshot(10))

    assert num_graphs == 1
    graph = _get_graph(data, slices, 0)
    freq_idx = EDGE_COL_ORDER.index("edge_freq")
    assert (graph.edge_attr[:, freq_idx] > 0).all()
    assert graph.edge_attr[:, freq_idx].max() > 1


def test_skewness_kurtosis_clamped():
    import polars as pl

    from graphids.core.data.datasets.can_bus import NODE_COL_ORDER

    n_rows = 50
    rng = np.random.default_rng(99)
    byte_data = {f"byte_{idx}": np.zeros(n_rows, dtype=np.float32) for idx in range(8)}
    byte_data["byte_0"][0] = 255.0
    df = pl.DataFrame(
        {
            "timestamp": np.arange(n_rows, dtype=np.float64),
            "node_id": pl.Series(rng.integers(0, 3, n_rows).tolist(), dtype=pl.Int64),
            **byte_data,
            "entropy": np.zeros(n_rows, dtype=np.float32),
            "attack": [0] * n_rows,
            "attack_type": [0] * n_rows,
        }
    )

    data, slices, _, _ = _run(df, _snapshot(n_rows))
    graph = _get_graph(data, slices, 0)

    assert graph.x[:, NODE_COL_ORDER.index("skewness")].abs().max() <= 10.0
    assert graph.x[:, NODE_COL_ORDER.index("kurtosis")].abs().max() <= 10.0


def test_build_tables_returns_stage_tables():
    import polars as pl

    n_rows = 20
    df = pl.DataFrame(
        {
            "timestamp": np.arange(n_rows, dtype=np.float64),
            "node_id": pl.Series(([0, 1] * (n_rows // 2)), dtype=pl.Int64),
            **{f"byte_{i}": np.zeros(n_rows, dtype=np.float32) for i in range(8)},
            "entropy": np.zeros(n_rows, dtype=np.float32),
            "attack": [0] * n_rows,
            "attack_type": [0] * n_rows,
        }
    )

    tables = _build_tables(df, _snapshot(n_rows))

    assert tables.n_rows == n_rows
    assert len(tables.node_stats) > 0
    assert len(tables.edge_df) > 0
    assert len(tables.labels) > 0
