"""Preprocessing tests: feature computation produces correct shapes and values."""

from __future__ import annotations

import numpy as np
import torch


def _make_window(n_rows: int = 20, n_ids: int = 5):
    """Synthetic Polars DataFrame mimicking a CAN bus window."""
    import polars as pl
    rng = np.random.default_rng(42)
    node_ids = rng.integers(0, n_ids, n_rows)
    return pl.DataFrame({
        "timestamp": np.arange(n_rows, dtype=np.float64),
        "arb_id": [f"0x{i:03X}" for i in node_ids],
        "node_id": pl.Series(node_ids.tolist(), dtype=pl.Int64),
        "payload": ["AABBCCDD11223344"] * n_rows,
        **{f"byte_{i}": rng.uniform(0, 255, n_rows).astype(np.float32) for i in range(8)},
        "entropy": rng.uniform(0, 2, n_rows).astype(np.float32),
        "attack_type": [0] * n_rows,
    })


def test_node_features_shape():
    from graphids.core.data.datasets.can_bus import node_features, NODE_COL_ORDER
    x, node_ids = node_features(_make_window(80, 8), edge_index=np.array([[0, 1], [1, 2]]))
    n_active = node_ids.shape[0]
    assert x.shape == (n_active, len(NODE_COL_ORDER))
    assert node_ids.shape == (n_active,)
    assert not torch.isnan(x).any()


def test_node_features_skewness_clamped():
    from graphids.core.data.datasets.can_bus import node_features
    x, _ = node_features(_make_window(100, 3))
    # Skewness and kurtosis columns — verify clamping to +/-10
    # Only skewness and kurtosis are clamped — other features may exceed 10
    from graphids.core.data.datasets.can_bus import NODE_COL_ORDER
    skew_idx = NODE_COL_ORDER.index("skewness")
    kurt_idx = NODE_COL_ORDER.index("kurtosis")
    assert x[:, skew_idx].abs().max() <= 10.0
    assert x[:, kurt_idx].abs().max() <= 10.0


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
        y=data.y[idx:idx + 1],
        attack_type=data.attack_type[idx:idx + 1],
    )


def test_sliding_window_graphs_shapes_and_values():
    """sliding_window_graphs produces Data objects with correct shapes and edge features."""
    from graphids.core.data.graph_pipeline import sliding_window_graphs
    from graphids.core.data.datasets.can_bus import (
        N_EDGE_FEATURES, N_NODE_FEATURES, EDGE_COL_ORDER,
        NODE_STAT_EXPRS, EDGE_STAT_EXPRS, NODE_COL_ORDER,
        LABEL_EXPRS, EDGE_BASE_COLS,
    )
    import polars as pl

    n_rows = 20
    rng = np.random.default_rng(0)
    node_ids = rng.integers(0, 4, n_rows)
    df = pl.DataFrame({
        "timestamp": np.arange(n_rows, dtype=np.float64),  # IAT = 1.0
        "node_id": pl.Series(node_ids.tolist(), dtype=pl.Int64),
        **{f"byte_{i}": np.ones(n_rows, dtype=np.float32) * i for i in range(8)},
        "entropy": np.zeros(n_rows, dtype=np.float32),
        "attack": [0] * n_rows,
        "attack_type": [0] * n_rows,
    })
    data, slices, num_graphs = sliding_window_graphs(
        df, window_size=10, stride=5,
        node_stat_exprs=NODE_STAT_EXPRS, edge_stat_exprs=EDGE_STAT_EXPRS,
        node_col_order=NODE_COL_ORDER, edge_col_order=EDGE_COL_ORDER,
        label_exprs=LABEL_EXPRS, edge_base_cols=EDGE_BASE_COLS,
    )
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
        assert (g.edge_attr[:, col_idx] == 0.0).all(), f"byte_{i}_diff should be 0 for constant bytes"


def test_node_iat_features():
    """node_iat_mean/std computed from per-node timestamp diffs."""
    from graphids.core.data.datasets.can_bus import node_features, NODE_COL_ORDER
    window = _make_window(80, 4)
    x, _ = node_features(window, edge_index=np.array([[0, 1], [1, 2]]))
    iat_mean_idx = NODE_COL_ORDER.index("node_iat_mean")
    iat_std_idx = NODE_COL_ORDER.index("node_iat_std")
    assert (x[:, iat_mean_idx] >= 0).all()
    assert (x[:, iat_std_idx] >= 0).all()
    assert not torch.isnan(x[:, iat_mean_idx]).any()
    assert not torch.isnan(x[:, iat_std_idx]).any()


def test_degree_features():
    """in_degree/out_degree filled post-hoc from edge_index."""
    from graphids.core.data.datasets.can_bus import node_features, NODE_COL_ORDER
    window = _make_window(80, 4)
    ei = np.array([[0, 1], [1, 2]])
    x, _ = node_features(window, edge_index=ei)
    in_deg_idx = NODE_COL_ORDER.index("in_degree")
    out_deg_idx = NODE_COL_ORDER.index("out_degree")
    assert (x[:, in_deg_idx] >= 0).all()
    assert (x[:, out_deg_idx] >= 0).all()
    x_no_ei, _ = node_features(window, edge_index=None)
    assert (x_no_ei[:, in_deg_idx] == 0).all()
    assert (x_no_ei[:, out_deg_idx] == 0).all()


def test_sliding_window_graphs_edge_freq():
    """edge_freq counts repeated (src, dst) pairs within a window."""
    from graphids.core.data.graph_pipeline import sliding_window_graphs
    from graphids.core.data.datasets.can_bus import (
        EDGE_COL_ORDER, NODE_STAT_EXPRS, EDGE_STAT_EXPRS, NODE_COL_ORDER,
        LABEL_EXPRS, EDGE_BASE_COLS,
    )
    import polars as pl

    # 10 rows, 2 node IDs → many repeated (src, dst) pairs
    node_ids = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    df = pl.DataFrame({
        "timestamp": np.arange(10, dtype=np.float64),
        "node_id": pl.Series(node_ids, dtype=pl.Int64),
        **{f"byte_{i}": np.zeros(10, dtype=np.float32) for i in range(8)},
        "entropy": np.zeros(10, dtype=np.float32),
        "attack": [0] * 10,
        "attack_type": [0] * 10,
    })
    data, slices, num_graphs = sliding_window_graphs(
        df, window_size=10, stride=10,
        node_stat_exprs=NODE_STAT_EXPRS, edge_stat_exprs=EDGE_STAT_EXPRS,
        node_col_order=NODE_COL_ORDER, edge_col_order=EDGE_COL_ORDER,
        label_exprs=LABEL_EXPRS, edge_base_cols=EDGE_BASE_COLS,
    )
    assert num_graphs == 1
    g = _get_graph(data, slices, 0)
    freq_idx = EDGE_COL_ORDER.index("edge_freq")
    # Alternating 0,1,0,1... → edges are 0→1 and 1→0, each repeated multiple times
    assert (g.edge_attr[:, freq_idx] > 0).all(), "edge_freq should be positive"
    # With 10 alternating IDs: 5 edges 0→1, 4 edges 1→0 → freq should reflect counts
    assert g.edge_attr[:, freq_idx].max() > 1, "Repeated pairs should have edge_freq > 1"


class TestClusteringCoefficients:
    """clustering_coefficients: scipy sparse implementation vs NetworkX reference."""

    @staticmethod
    def _nx_reference(edge_index: np.ndarray, num_nodes: int) -> np.ndarray:
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        G.add_edges_from(zip(edge_index[0], edge_index[1]))
        cc = nx.clustering(G)
        return np.array([cc.get(i, 0.0) for i in range(num_nodes)], dtype=np.float32)

    def test_triangle(self):
        from graphids.core.data.datasets.can_bus import clustering_coefficients
        ei = np.array([[0, 1, 2], [1, 2, 0]])
        cc = clustering_coefficients(ei, 3)
        np.testing.assert_allclose(cc, [1.0, 1.0, 1.0], atol=1e-6)

    def test_path(self):
        from graphids.core.data.datasets.can_bus import clustering_coefficients
        ei = np.array([[0, 1], [1, 2]])
        cc = clustering_coefficients(ei, 3)
        np.testing.assert_allclose(cc, [0.0, 0.0, 0.0], atol=1e-6)

    def test_star(self):
        from graphids.core.data.datasets.can_bus import clustering_coefficients
        ei = np.array([[0, 0, 0], [1, 2, 3]])
        cc = clustering_coefficients(ei, 4)
        np.testing.assert_allclose(cc, [0.0, 0.0, 0.0, 0.0], atol=1e-6)

    def test_empty(self):
        from graphids.core.data.datasets.can_bus import clustering_coefficients
        cc = clustering_coefficients(np.zeros((2, 0), dtype=np.int64), 3)
        assert cc.shape == (3,)
        assert (cc == 0).all()

    def test_isolated_nodes(self):
        from graphids.core.data.datasets.can_bus import clustering_coefficients
        cc = clustering_coefficients(np.zeros((2, 0), dtype=np.int64), 0)
        assert cc.shape == (0,)

    def test_matches_networkx_random(self):
        from graphids.core.data.datasets.can_bus import clustering_coefficients
        rng = np.random.default_rng(123)
        for _ in range(20):
            n = rng.integers(5, 30)
            m = rng.integers(5, n * 2)
            src = rng.integers(0, n, m)
            dst = rng.integers(0, n, m)
            ei = np.stack([src, dst])
            cc_scipy = clustering_coefficients(ei, n)
            cc_nx = self._nx_reference(ei, n)
            np.testing.assert_allclose(cc_scipy, cc_nx, atol=1e-5,
                                       err_msg=f"Mismatch for n={n}, m={m}")


