"""Preprocessing tests: feature computation produces correct shapes and values.

Tests exercise the production path (GraphPipeline) with CAN bus
schema constants. The pipeline is domain-agnostic; the CAN-specific Polars
expressions and column layouts are injected as parameters.
"""

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


def test_sliding_window_graphs_shapes_and_values():
    """GraphPipeline produces Data objects with correct shapes and edge features."""
    import polars as pl

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
    from graphids.core.data.preprocessing.pipeline import GraphPipeline
    from graphids.core.data.preprocessing.pipeline import run as run_pipeline

    n_rows = 20
    rng = np.random.default_rng(0)
    node_ids = rng.integers(0, 4, n_rows)
    df = pl.DataFrame(
        {
            "timestamp": np.arange(n_rows, dtype=np.float64),  # IAT = 1.0
            "node_id": pl.Series(node_ids.tolist(), dtype=pl.Int64),
            **{f"byte_{i}": np.ones(n_rows, dtype=np.float32) * i for i in range(8)},
            "entropy": np.zeros(n_rows, dtype=np.float32),
            "attack": [0] * n_rows,
            "attack_type": [0] * n_rows,
        }
    )
    pipeline = GraphPipeline(
        node_stat_exprs=NODE_STAT_EXPRS,
        edge_stat_exprs=EDGE_STAT_EXPRS,
        node_col_order=NODE_COL_ORDER,
        edge_col_order=EDGE_COL_ORDER,
        label_exprs=LABEL_EXPRS,
        edge_base_cols=EDGE_BASE_COLS,
        representation_cfg=_snapshot(n_rows),
    )
    data, slices, num_graphs, _ = run_pipeline(pipeline, df)
    assert num_graphs > 0
    g = _get_graph(data, slices, 0)
    assert g.x.shape[1] == N_NODE_FEATURES
    assert g.edge_attr.shape[1] == N_EDGE_FEATURES
    assert g.edge_index.shape[0] == 2
    assert g.y.item() == 0  # all-normal data
    assert not torch.isnan(g.x).any()
    assert not torch.isnan(g.edge_attr).any()
    # IAT column (index 0): consecutive timestamps differ by 1.0
    iat_idx = EDGE_COL_ORDER.index("iat")
    assert (g.edge_attr[:, iat_idx] == 1.0).all(), "IAT should be 1.0 for unit-spaced timestamps"
    # Byte diff columns: constant bytes → all diffs = 0
    for i in range(8):
        col_idx = EDGE_COL_ORDER.index(f"byte_{i}_diff")
        assert (g.edge_attr[:, col_idx] == 0.0).all(), (
            f"byte_{i}_diff should be 0 for constant bytes"
        )


def test_sliding_window_graphs_edge_freq():
    """edge_freq counts repeated (src, dst) pairs within a window."""
    import polars as pl

    from graphids.core.data.datasets.can_bus import (
        EDGE_BASE_COLS,
        EDGE_COL_ORDER,
        EDGE_STAT_EXPRS,
        LABEL_EXPRS,
        NODE_COL_ORDER,
        NODE_STAT_EXPRS,
    )
    from graphids.core.data.preprocessing.pipeline import GraphPipeline
    from graphids.core.data.preprocessing.pipeline import run as run_pipeline

    # 10 rows, 2 node IDs → many repeated (src, dst) pairs
    node_ids = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    df = pl.DataFrame(
        {
            "timestamp": np.arange(10, dtype=np.float64),
            "node_id": pl.Series(node_ids, dtype=pl.Int64),
            **{f"byte_{i}": np.zeros(10, dtype=np.float32) for i in range(8)},
            "entropy": np.zeros(10, dtype=np.float32),
            "attack": [0] * 10,
            "attack_type": [0] * 10,
        }
    )
    pipeline = GraphPipeline(
        node_stat_exprs=NODE_STAT_EXPRS,
        edge_stat_exprs=EDGE_STAT_EXPRS,
        node_col_order=NODE_COL_ORDER,
        edge_col_order=EDGE_COL_ORDER,
        label_exprs=LABEL_EXPRS,
        edge_base_cols=EDGE_BASE_COLS,
        representation_cfg=_snapshot(10),
    )
    data, slices, num_graphs, _ = run_pipeline(pipeline, df)
    assert num_graphs == 1
    g = _get_graph(data, slices, 0)
    freq_idx = EDGE_COL_ORDER.index("edge_freq")
    # Alternating 0,1,0,1... → edges are 0→1 and 1→0, each repeated multiple times
    assert (g.edge_attr[:, freq_idx] > 0).all(), "edge_freq should be positive"
    # With 10 alternating IDs: 5 edges 0→1, 4 edges 1→0 → freq should reflect counts
    assert g.edge_attr[:, freq_idx].max() > 1, "Repeated pairs should have edge_freq > 1"


def test_skewness_kurtosis_clamped():
    """Skewness/kurtosis clamped to [-10, 10] in production pipeline.

    INVARIANT: critical-constraints.md — fp16 max ~65504, unclamped skewness
    can reach 1e17 causing MSE overflow. The clamp is in NODE_STAT_EXPRS.
    """
    import polars as pl

    from graphids.core.data.datasets.can_bus import (
        EDGE_BASE_COLS,
        EDGE_COL_ORDER,
        EDGE_STAT_EXPRS,
        LABEL_EXPRS,
        NODE_COL_ORDER,
        NODE_STAT_EXPRS,
    )
    from graphids.core.data.preprocessing.pipeline import GraphPipeline
    from graphids.core.data.preprocessing.pipeline import run as run_pipeline

    # Extreme byte values: one constant column + one high-variance column
    # to provoke large skewness/kurtosis before clamping.
    n_rows = 50
    rng = np.random.default_rng(99)
    node_ids = rng.integers(0, 3, n_rows)
    byte_data = {f"byte_{i}": np.zeros(n_rows, dtype=np.float32) for i in range(8)}
    # byte_0: single spike → extreme skewness
    byte_data["byte_0"] = np.zeros(n_rows, dtype=np.float32)
    byte_data["byte_0"][0] = 255.0
    df = pl.DataFrame(
        {
            "timestamp": np.arange(n_rows, dtype=np.float64),
            "node_id": pl.Series(node_ids.tolist(), dtype=pl.Int64),
            **byte_data,
            "entropy": np.zeros(n_rows, dtype=np.float32),
            "attack": [0] * n_rows,
            "attack_type": [0] * n_rows,
        }
    )
    pipeline = GraphPipeline(
        node_stat_exprs=NODE_STAT_EXPRS,
        edge_stat_exprs=EDGE_STAT_EXPRS,
        node_col_order=NODE_COL_ORDER,
        edge_col_order=EDGE_COL_ORDER,
        label_exprs=LABEL_EXPRS,
        edge_base_cols=EDGE_BASE_COLS,
        representation_cfg=_snapshot(n_rows),
    )
    data, slices, num_graphs, _ = run_pipeline(pipeline, df)
    assert num_graphs > 0
    g = _get_graph(data, slices, 0)
    skew_idx = NODE_COL_ORDER.index("skewness")
    kurt_idx = NODE_COL_ORDER.index("kurtosis")
    assert g.x[:, skew_idx].abs().max() <= 10.0
    assert g.x[:, kurt_idx].abs().max() <= 10.0


def test_build_tables_and_debug_artifacts(tmp_path):
    """GraphPipeline.build_tables emits stage tables and optional parquet artifacts."""
    import polars as pl

    from graphids.core.data.datasets.can_bus import (
        EDGE_BASE_COLS,
        EDGE_COL_ORDER,
        EDGE_STAT_EXPRS,
        LABEL_EXPRS,
        NODE_COL_ORDER,
        NODE_STAT_EXPRS,
    )
    from graphids.core.data.preprocessing.pipeline import GraphPipeline
    from graphids.core.data.preprocessing.pipeline import (
        build_tables as build_pipeline_tables,
    )

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
    pipeline = GraphPipeline(
        node_stat_exprs=NODE_STAT_EXPRS,
        edge_stat_exprs=EDGE_STAT_EXPRS,
        node_col_order=NODE_COL_ORDER,
        edge_col_order=EDGE_COL_ORDER,
        label_exprs=LABEL_EXPRS,
        edge_base_cols=EDGE_BASE_COLS,
        debug_artifacts_dir=tmp_path / "artifacts",
        representation_cfg=_snapshot(n_rows),
    )
    tables = build_pipeline_tables(pipeline, df)
    assert tables.n_rows == n_rows
    assert len(tables.node_stats) > 0
    assert len(tables.edge_df) > 0
    assert len(tables.labels) > 0
    assert (tmp_path / "artifacts" / "01_windowed_rows.parquet").exists()
    assert (tmp_path / "artifacts" / "11_node_stats_localized.parquet").exists()


def test_edge_policy_explicit_direction():
    """Custom edge policy should control src/dst construction explicitly."""
    import polars as pl

    from graphids.core.data.datasets.can_bus import (
        EDGE_BASE_COLS,
        EDGE_COL_ORDER,
        EDGE_STAT_EXPRS,
        LABEL_EXPRS,
        NODE_COL_ORDER,
        NODE_STAT_EXPRS,
    )
    from graphids.core.data.preprocessing.edge_policy import temporal_edge_policy
    from graphids.core.data.preprocessing.pipeline import GraphPipeline
    from graphids.core.data.preprocessing.pipeline import (
        build_tables as build_pipeline_tables,
    )

    df = pl.DataFrame(
        {
            "timestamp": np.arange(6, dtype=np.float64),
            "node_id": pl.Series([0, 1, 2, 3, 4, 5], dtype=pl.Int64),
            **{f"byte_{i}": np.zeros(6, dtype=np.float32) for i in range(8)},
            "entropy": np.zeros(6, dtype=np.float32),
            "attack": [0] * 6,
            "attack_type": [0] * 6,
        }
    )
    pipeline = GraphPipeline(
        node_stat_exprs=NODE_STAT_EXPRS,
        edge_stat_exprs=EDGE_STAT_EXPRS,
        node_col_order=NODE_COL_ORDER,
        edge_col_order=EDGE_COL_ORDER,
        label_exprs=LABEL_EXPRS,
        edge_base_cols=EDGE_BASE_COLS,
        edge_policy=temporal_edge_policy(dst_shift=1),
        representation_cfg=_snapshot(6),
    )
    tables = build_pipeline_tables(pipeline, df)
    assert len(tables.edge_df) > 0
    assert (tables.edge_df["dst"] > tables.edge_df["src"]).all()


def test_secondary_graph_transforms_are_composable():
    """Secondary graph transforms can be composed for exploratory node stats."""
    import polars as pl

    from graphids.core.data.datasets.can_bus import (
        EDGE_BASE_COLS,
        EDGE_COL_ORDER,
        EDGE_STAT_EXPRS,
        LABEL_EXPRS,
        NODE_COL_ORDER,
        NODE_STAT_EXPRS,
    )
    from graphids.core.data.preprocessing.graph_ops import (
        default_graph_transforms,
        secondary_graph_transforms,
    )
    from graphids.core.data.preprocessing.pipeline import GraphPipeline
    from graphids.core.data.preprocessing.pipeline import (
        build_tables as build_pipeline_tables,
    )

    n_rows = 20
    df = pl.DataFrame(
        {
            "timestamp": np.arange(n_rows, dtype=np.float64),
            "node_id": pl.Series(([0, 1, 0, 2, 1] * 4), dtype=pl.Int64),
            **{f"byte_{i}": np.zeros(n_rows, dtype=np.float32) for i in range(8)},
            "entropy": np.zeros(n_rows, dtype=np.float32),
            "attack": [0] * n_rows,
            "attack_type": [0] * n_rows,
        }
    )
    pipeline = GraphPipeline(
        node_stat_exprs=NODE_STAT_EXPRS,
        edge_stat_exprs=EDGE_STAT_EXPRS,
        node_col_order=NODE_COL_ORDER + ["in_out_ratio", "neighbor_entropy"],
        edge_col_order=EDGE_COL_ORDER,
        label_exprs=LABEL_EXPRS,
        edge_base_cols=EDGE_BASE_COLS,
        graph_transforms=[*default_graph_transforms(), *secondary_graph_transforms()],
        representation_cfg=_snapshot(n_rows),
    )
    tables = build_pipeline_tables(pipeline, df)
    assert "in_out_ratio" in tables.node_stats.columns
    assert "neighbor_entropy" in tables.node_stats.columns
    assert not tables.node_stats["in_out_ratio"].is_null().any()
    assert not tables.node_stats["neighbor_entropy"].is_null().any()
