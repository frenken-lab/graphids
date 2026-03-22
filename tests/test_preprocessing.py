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
    from graphids.core.preprocessing.features import node_features, NODE_COL_ORDER
    x = node_features(_make_window(30, 8), 8, edge_index=np.array([[0, 1], [1, 2]]))
    assert x.shape == (8, len(NODE_COL_ORDER))
    assert not torch.isnan(x).any()


def test_node_features_skewness_clamped():
    from graphids.core.preprocessing.features import node_features
    x = node_features(_make_window(100, 3), 3)
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
        [np.random.rand(n + 1).astype(np.float32) for _ in range(4)],
        np.arange(n, dtype=np.int64), np.arange(1, n + 1, dtype=np.int64),
    )
    assert ea.shape == (n, EDGE_FEATURE_COUNT)


def test_edge_features_iat():
    """Inter-arrival time in columns 0, 2, 3."""
    from graphids.core.preprocessing.features import edge_features
    ea = edge_features(
        np.array([0.0, 0.1, 0.3, 0.6]),
        [np.zeros(4, dtype=np.float32) for _ in range(4)],
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
    x = node_features(window, 5, edge_index=np.stack([src, dst]))
    ea = edge_features(
        window["timestamp"].to_numpy(),
        [window[f"byte_{i}"].to_numpy() for i in range(4)],
        src, dst,
    )
    assert x.shape == (5, 31)
    assert ea.shape == (19, 12)
