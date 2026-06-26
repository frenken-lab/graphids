"""Dataset construction and attack type inference tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch


class TestInferAttackType:
    """attack type substring matching -- verify all attack codes."""

    @pytest.fixture
    def infer(self):
        from graphids.core.data.datasets.can_bus import infer_attack_type

        return infer_attack_type

    @pytest.mark.parametrize(
        "stem,parent,expected",
        [
            ("normal_driving", "benign", 0),
            ("dos_attack", "test", 1),
            ("fuzzing_data", "attacks", 2),
            ("fuzzy_data", "attacks", 2),
            ("gear_spoof", "test", 3),
            ("rpm_attack", "test", 4),
            ("flooding_test", "attacks", 5),
            ("unknown_file", "unknown_dir", 0),
        ],
    )
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
    """CANBusDataset._build_graphs integration -- CSV to PyG Data."""

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
        from graphids.core.data.datasets.can_bus import CANBusDataset
        from graphids.core.data.preprocessing.vocab import persist_vocab, scan_arb_ids

        self._write_minimal_csv(tmp_path / "raw" / "train_01_attack_free" / "normal.csv")
        # Shared-vocab setup mirrors CANBusSource.build() — scan every
        # source_dir, build vocab, persist. This exercises the Stage-1
        # path end-to-end on synthetic data.
        vocab = {
            tok: i + 1
            for i, tok in enumerate(scan_arb_ids(tmp_path / "raw", ["train_01_attack_free"]))
        }
        digest = persist_vocab(vocab, tmp_path / "processed" / "vocab.json")
        ds = CANBusDataset(
            root=str(tmp_path / "processed"),
            raw_dir=str(tmp_path / "raw"),
            split="train",
            val_fraction=0.2,
            source_dirs=["train_01_attack_free"],
            shared_vocab=vocab,
            shared_vocab_digest=digest,
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
        assert g.x.shape[0] < 2048
        assert g.edge_index.max() < g.x.shape[0]
        assert not torch.isnan(g.x).any()
