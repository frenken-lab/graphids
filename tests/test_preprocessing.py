"""Preprocessing tests: feature computation produces valid Data with correct dims."""

from __future__ import annotations

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Node features
# ---------------------------------------------------------------------------


class TestNodeFeatures:
    def _make_window(self, n_rows: int = 20, n_ids: int = 5):
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

    def test_node_features_shape(self):
        """node_features returns [num_nodes, 31] tensor."""
        from graphids.core.preprocessing.features import node_features

        window = self._make_window(n_rows=30, n_ids=8)
        num_nodes = 8
        edge_index = np.array([[0, 1, 2], [1, 2, 3]])
        x = node_features(window, num_nodes, edge_index=edge_index)
        assert x.shape == (num_nodes, 31)
        assert x.dtype == torch.float32

    def test_node_features_no_nans(self):
        """Feature tensor contains no NaN values."""
        from graphids.core.preprocessing.features import node_features

        window = self._make_window(n_rows=50, n_ids=10)
        x = node_features(window, 10)
        assert not torch.isnan(x).any(), "NaN detected in node features"

    def test_node_features_clustering_filled(self):
        """Column 28 (clustering_coeff) is populated when edge_index provided."""
        from graphids.core.preprocessing.features import node_features

        window = self._make_window(n_rows=20, n_ids=5)
        # Create edges that form a triangle: 0-1, 1-2, 2-0
        edge_index = np.array([[0, 1, 2, 1, 2, 0], [1, 2, 0, 0, 1, 2]])
        x = node_features(window, 5, edge_index=edge_index)
        # Nodes 0,1,2 should have non-zero clustering coefficients
        assert x[:3, 28].sum() > 0

    def test_skewness_kurtosis_clamped(self):
        """Skewness (col 26) and kurtosis (col 27) are clamped to [-10, 10]."""
        from graphids.core.preprocessing.features import node_features

        window = self._make_window(n_rows=100, n_ids=3)
        x = node_features(window, 3)
        assert x[:, 26].abs().max() <= 10.0, "Skewness not clamped"
        assert x[:, 27].abs().max() <= 10.0, "Kurtosis not clamped"


# ---------------------------------------------------------------------------
# Edge features
# ---------------------------------------------------------------------------


class TestEdgeFeatures:
    def test_edge_features_shape(self):
        """edge_features returns [num_edges, 12] tensor."""
        from graphids.core.preprocessing.features import edge_features

        n = 15
        timestamps = np.arange(n + 1, dtype=np.float64)  # n+1 messages -> n edges
        byte_arrays = [np.random.rand(n + 1).astype(np.float32) for _ in range(4)]
        src = np.arange(n, dtype=np.int64)
        dst = np.arange(1, n + 1, dtype=np.int64)

        ea = edge_features(timestamps, byte_arrays, src, dst)
        assert ea.shape == (n, 12)
        assert ea.dtype == torch.float32

    def test_edge_features_empty(self):
        """Empty edge arrays produce [0, 12] tensor."""
        from graphids.core.preprocessing.features import edge_features

        ea = edge_features(
            np.array([]), [np.array([]) for _ in range(4)],
            np.array([], dtype=np.int64), np.array([], dtype=np.int64),
        )
        assert ea.shape == (0, 12)

    def test_edge_features_iat_columns(self):
        """Inter-arrival time is in columns 0, 2, 3."""
        from graphids.core.preprocessing.features import edge_features

        timestamps = np.array([0.0, 0.1, 0.3, 0.6])
        byte_arrays = [np.zeros(4, dtype=np.float32) for _ in range(4)]
        src = np.array([0, 1, 2])
        dst = np.array([1, 2, 3])

        ea = edge_features(timestamps, byte_arrays, src, dst)
        expected_iat = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
        torch.testing.assert_close(ea[:, 0], expected_iat, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(ea[:, 2], expected_iat, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(ea[:, 3], expected_iat, atol=1e-5, rtol=1e-5)

    def test_edge_features_bidirectional(self):
        """Column 11 (bidirectional) is 1.0 when reverse edge exists."""
        from graphids.core.preprocessing.features import edge_features

        timestamps = np.array([0.0, 0.1, 0.2])
        byte_arrays = [np.zeros(3, dtype=np.float32) for _ in range(4)]
        # Edge 0->1 and edge 1->0 => bidirectional
        src = np.array([0, 1])
        dst = np.array([1, 0])

        ea = edge_features(timestamps, byte_arrays, src, dst)
        assert ea[0, 11] == 1.0  # 0->1 has reverse 1->0
        assert ea[1, 11] == 1.0  # 1->0 has reverse 0->1


# ---------------------------------------------------------------------------
# Vocab utility
# ---------------------------------------------------------------------------


class TestVocab:
    def test_vocab_from_column(self):
        """vocab_from_column produces dense 1-indexed mapping with OOV=0."""
        import polars as pl

        from graphids.core.preprocessing.utils import vocab_from_column

        series = pl.Series(["0xAA", "0xBB", "0xAA", "0xCC"])
        vocab, oov = vocab_from_column(series)
        assert oov == 0
        assert len(vocab) == 3  # AA, BB, CC
        assert all(v > 0 for v in vocab.values()), "Vocab indices should be 1-indexed"


# ---------------------------------------------------------------------------
# Graph construction (window_to_graph equivalent)
# ---------------------------------------------------------------------------


class TestGraphConstruction:
    def test_synthetic_graph_has_correct_attributes(self):
        """A manually constructed Data object has expected fields and shapes."""
        from graphids.core.preprocessing.features import edge_features, node_features

        import polars as pl

        rng = np.random.default_rng(99)
        n_rows, n_ids = 20, 5
        node_ids = rng.integers(0, n_ids, n_rows)
        window = pl.DataFrame({
            "timestamp": np.arange(n_rows, dtype=np.float64),
            "arb_id": [f"0x{i:03X}" for i in node_ids],
            "node_id": pl.Series(node_ids.tolist(), dtype=pl.Int64),
            "payload": ["AABB"] * n_rows,
            **{f"byte_{i}": rng.uniform(0, 255, n_rows).astype(np.float32) for i in range(8)},
            "entropy": rng.uniform(0, 2, n_rows).astype(np.float32),
            "attack_type": [0] * n_rows,
        })

        node_ids = np.array(window["node_id"].to_list())
        src, dst = node_ids[:-1], node_ids[1:]
        ei = np.stack([src, dst])

        x = node_features(window, n_ids, edge_index=ei)
        ea = edge_features(
            window["timestamp"].to_numpy(),
            [window[f"byte_{i}"].to_numpy() for i in range(4)],
            src, dst,
        )

        assert x.shape == (n_ids, 31), f"Expected (5, 31), got {x.shape}"
        assert ea.shape[0] == n_rows - 1
        assert ea.shape[1] == 12

        data = torch.from_numpy(ei)
        assert data.shape == (2, n_rows - 1)
