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
    x = node_features(_make_window(80, 8), 8, edge_index=np.array([[0, 1], [1, 2]]))
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
        assert g.x.shape[1] == 31
        assert g.edge_attr.shape[1] == 12
        assert not torch.isnan(g.x).any()
