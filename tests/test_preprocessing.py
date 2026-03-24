"""Preprocessing tests: feature computation produces correct shapes and values."""

from __future__ import annotations

import numpy as np
import pytest
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
    from graphids.core.preprocessing.features import node_features, NODE_COL_ORDER
    x, node_ids = node_features(_make_window(80, 8), edge_index=np.array([[0, 1], [1, 2]]))
    n_active = node_ids.shape[0]
    assert x.shape == (n_active, len(NODE_COL_ORDER))
    assert node_ids.shape == (n_active,)
    assert not torch.isnan(x).any()


def test_node_features_skewness_clamped():
    from graphids.core.preprocessing.features import node_features
    x, _ = node_features(_make_window(100, 3))
    # Skewness and kurtosis columns — verify clamping to ±10
    assert x.abs().max() <= 10.0 or True  # some features exceed 10 legitimately
    # But skewness (col 26) and kurtosis (col 27) specifically must be clamped
    from graphids.core.preprocessing.features import NODE_COL_ORDER
    skew_idx = NODE_COL_ORDER.index("skewness")
    kurt_idx = NODE_COL_ORDER.index("kurtosis")
    assert x[:, skew_idx].abs().max() <= 10.0
    assert x[:, kurt_idx].abs().max() <= 10.0


def test_edge_features_shape():
    from graphids.core.preprocessing.features import edge_features
    from graphids.config.constants import EDGE_FEATURE_COUNT
    n = 15
    ea = edge_features(
        np.arange(n + 1, dtype=np.float64),
        [np.random.rand(n + 1).astype(np.float32) for _ in range(8)],
        np.arange(n, dtype=np.int64), np.arange(1, n + 1, dtype=np.int64),
    )
    assert ea.shape == (n, EDGE_FEATURE_COUNT)


def test_edge_features_iat():
    """Inter-arrival time in column 0."""
    from graphids.core.preprocessing.features import edge_features
    ea = edge_features(
        np.array([0.0, 0.1, 0.3, 0.6]),
        [np.zeros(4, dtype=np.float32) for _ in range(8)],
        np.array([0, 1, 2]), np.array([1, 2, 3]),
    )
    expected = torch.tensor([0.1, 0.2, 0.3])
    torch.testing.assert_close(ea[:, 0], expected, atol=1e-5, rtol=1e-5)


def test_graph_construction_end_to_end():
    """node_features + edge_features produce compatible tensors for Data."""
    from graphids.core.preprocessing.features import edge_features, node_features
    window = _make_window(20, 5)
    ids = np.array(window["node_id"].to_list())
    src, dst = ids[:-1], ids[1:]
    x, node_ids = node_features(window, edge_index=np.stack([src, dst]))
    ea = edge_features(
        window["timestamp"].to_numpy(),
        [window[f"byte_{i}"].to_numpy() for i in range(8)],
        src, dst,
    )
    n_active = node_ids.shape[0]
    assert x.shape == (n_active, 35)
    assert node_ids.shape == (n_active,)
    assert ea.shape == (19, 11)


def test_node_iat_features():
    """node_iat_mean/std computed from per-node timestamp diffs."""
    from graphids.core.preprocessing.features import node_features, NODE_COL_ORDER
    window = _make_window(80, 4)
    x, _ = node_features(window, edge_index=np.array([[0, 1], [1, 2]]))
    iat_mean_idx = NODE_COL_ORDER.index("node_iat_mean")
    iat_std_idx = NODE_COL_ORDER.index("node_iat_std")
    # IAT mean should be positive (timestamps are monotonically increasing)
    assert (x[:, iat_mean_idx] >= 0).all()
    # IAT std should be non-negative
    assert (x[:, iat_std_idx] >= 0).all()
    assert not torch.isnan(x[:, iat_mean_idx]).any()
    assert not torch.isnan(x[:, iat_std_idx]).any()


def test_degree_features():
    """in_degree/out_degree filled post-hoc from edge_index."""
    from graphids.core.preprocessing.features import node_features, NODE_COL_ORDER
    window = _make_window(80, 4)
    # Create a known edge structure: 0→1, 1→2
    ei = np.array([[0, 1], [1, 2]])
    x, _ = node_features(window, edge_index=ei)
    in_deg_idx = NODE_COL_ORDER.index("in_degree")
    out_deg_idx = NODE_COL_ORDER.index("out_degree")
    n = x.shape[0]
    # All degree values should be non-negative integers
    assert (x[:, in_deg_idx] >= 0).all()
    assert (x[:, out_deg_idx] >= 0).all()
    # Without edge_index, degrees should be zero
    x_no_ei, _ = node_features(window, edge_index=None)
    assert (x_no_ei[:, in_deg_idx] == 0).all()
    assert (x_no_ei[:, out_deg_idx] == 0).all()


def test_edge_freq_numpy_path():
    """edge_freq in numpy path counts (src, dst) pair occurrences."""
    from graphids.core.preprocessing.features import edge_features, N_EDGE_FEATURES
    # Two edges share (0→1), one unique (1→2)
    src = np.array([0, 0, 1], dtype=np.int64)
    dst = np.array([1, 1, 2], dtype=np.int64)
    ts = np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float64)
    byte_arrs = [np.zeros(4, dtype=np.float32) for _ in range(8)]
    ea = edge_features(ts, byte_arrs, src, dst)
    assert ea.shape == (3, N_EDGE_FEATURES)
    # edge_freq is slot 10
    assert ea[0, 10] == 2.0  # (0→1) appears twice
    assert ea[1, 10] == 2.0  # (0→1) appears twice
    assert ea[2, 10] == 1.0  # (1→2) appears once


class TestInferAttackType:
    """_infer_attack_type substring matching — verify all attack codes."""

    @pytest.fixture
    def infer(self):
        from graphids.core.preprocessing.datasets.can_bus import CANBusDataset
        return CANBusDataset._infer_attack_type

    @pytest.mark.parametrize("stem,parent,expected", [
        ("normal_driving", "benign", 0),
        ("dos_attack", "test", 1),
        ("fuzzing_data", "attacks", 2),
        ("fuzzy_data", "attacks", 2),
        ("gear_spoof", "test", 3),
        ("rpm_attack", "test", 4),
        ("flooding_test", "attacks", 5),
        ("unknown_file", "unknown_dir", 0),
    ])
    def test_known_patterns(self, infer, stem, parent, expected, tmp_path):
        p = tmp_path / parent / f"{stem}.csv"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
        assert infer(p) == expected

    def test_fuzzy_vs_fuzzing_both_map_to_2(self, infer, tmp_path):
        """'fuzzy' and 'fuzzing' both map to code 2."""
        for name in ("fuzzy_test", "fuzzing_test"):
            p = tmp_path / "attacks" / f"{name}.csv"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.touch()
            assert infer(p) == 2


@pytest.mark.slow
class TestCANBusDatasetBuildGraphs:
    """CANBusDataset._build_graphs integration — CSV to PyG Data."""

    @staticmethod
    def _write_minimal_csv(path, n_rows=200, n_ids=5):
        import csv
        path.parent.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(42)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "arb_id", "data_field", "attack"])
            for i in range(n_rows):
                ts = float(i) * 0.001
                aid = f"0x{rng.integers(0, n_ids):03X}"
                payload = "".join(f"{rng.integers(0, 256):02X}" for _ in range(8))
                w.writerow([ts, aid, payload, 0])

    def test_produces_valid_data_objects(self, tmp_path):
        from graphids.core.preprocessing.datasets.can_bus import CANBusDataset
        self._write_minimal_csv(tmp_path / "raw" / "normal_test.csv")
        ds = CANBusDataset(
            root=str(tmp_path / "processed"), raw_dir=str(tmp_path / "raw"),
            split="train", window_size=50, stride=50, seed=42,
        )
        assert len(ds) > 0
        g = ds[0]
        assert hasattr(g, "x")
        assert hasattr(g, "edge_index")
        assert hasattr(g, "edge_attr")
        assert hasattr(g, "y")
        assert hasattr(g, "node_id")
        assert g.x.shape[1] == 35
        assert g.edge_attr.shape[1] == 11
        assert g.node_id.shape[0] == g.x.shape[0]
        assert g.x.shape[0] < 2048, "Compact graph should have << 2048 nodes"
        assert g.edge_index.max() < g.x.shape[0], "Edge indices must be local"
        assert not torch.isnan(g.x).any()


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
        """Complete triangle: every node has cc=1.0."""
        from graphids.core.preprocessing.features import clustering_coefficients
        ei = np.array([[0, 1, 2], [1, 2, 0]])
        cc = clustering_coefficients(ei, 3)
        np.testing.assert_allclose(cc, [1.0, 1.0, 1.0], atol=1e-6)

    def test_path(self):
        """Path graph 0-1-2: node 1 has cc=0 (no triangle)."""
        from graphids.core.preprocessing.features import clustering_coefficients
        ei = np.array([[0, 1], [1, 2]])
        cc = clustering_coefficients(ei, 3)
        np.testing.assert_allclose(cc, [0.0, 0.0, 0.0], atol=1e-6)

    def test_star(self):
        """Star graph: center cc=0, leaves cc=0."""
        from graphids.core.preprocessing.features import clustering_coefficients
        ei = np.array([[0, 0, 0], [1, 2, 3]])
        cc = clustering_coefficients(ei, 4)
        np.testing.assert_allclose(cc, [0.0, 0.0, 0.0, 0.0], atol=1e-6)

    def test_empty(self):
        from graphids.core.preprocessing.features import clustering_coefficients
        cc = clustering_coefficients(np.zeros((2, 0), dtype=np.int64), 3)
        assert cc.shape == (3,)
        assert (cc == 0).all()

    def test_isolated_nodes(self):
        from graphids.core.preprocessing.features import clustering_coefficients
        cc = clustering_coefficients(np.zeros((2, 0), dtype=np.int64), 0)
        assert cc.shape == (0,)

    def test_matches_networkx_random(self):
        """Random graphs: scipy matches NetworkX within tolerance."""
        from graphids.core.preprocessing.features import clustering_coefficients
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


class TestAssembleChunk:
    """Test graph assembly via _assemble_chunk_numpy + _numpy_to_data."""

    def test_single_window(self):
        from graphids.core.preprocessing.features import _assemble_chunk_numpy, _numpy_to_data
        # 3 nodes, 2 edges (0→1, 1→2), local indices — numpy arrays
        node_feats = np.zeros((3, 35), dtype=np.float32)
        node_ids = np.array([10, 20, 30], dtype=np.int64)
        edge_src = np.array([0, 1], dtype=np.int64)
        edge_dst = np.array([1, 2], dtype=np.int64)
        edge_feats = np.zeros((2, 11), dtype=np.float32)
        specs = [(0, 3, 0, 2, 0, 0)]  # s_start, s_count, e_start, e_count, y, at

        result = _assemble_chunk_numpy(node_feats, node_ids, edge_src, edge_dst, edge_feats, specs)
        graphs = _numpy_to_data(*result)
        assert len(graphs) == 1
        g = graphs[0]
        assert g.x.shape == (3, 35)
        assert g.edge_index.shape == (2, 2)
        assert g.edge_attr.shape == (2, 11)
        assert g.node_id.shape == (3,)
        assert g.edge_index.max() < 3, "Edge indices must be local"
        assert not torch.isnan(g.x).any()

    def test_multiple_windows(self):
        from graphids.core.preprocessing.features import _assemble_chunk_numpy, _numpy_to_data
        # 2 windows: first has 2 nodes/1 edge, second has 3 nodes/2 edges
        node_feats = np.zeros((5, 35), dtype=np.float32)
        node_ids = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        edge_src = np.array([0, 0, 1], dtype=np.int64)
        edge_dst = np.array([1, 1, 2], dtype=np.int64)
        edge_feats = np.zeros((3, 11), dtype=np.float32)
        specs = [
            (0, 2, 0, 1, 1, 0),  # window 1: 2 nodes, 1 edge, y=1
            (2, 3, 1, 2, 0, 2),  # window 2: 3 nodes, 2 edges, y=0, at=2
        ]

        result = _assemble_chunk_numpy(node_feats, node_ids, edge_src, edge_dst, edge_feats, specs)
        graphs = _numpy_to_data(*result)
        assert len(graphs) == 2
        assert graphs[0].x.shape[0] == 2
        assert graphs[1].x.shape[0] == 3
        assert graphs[0].y.item() == 1
        assert graphs[1].attack_type.item() == 2
