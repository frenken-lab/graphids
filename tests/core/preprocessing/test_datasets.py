"""Dataset construction invariants."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import torch


def test_infer_attack_type_recognizes_representative_patterns(tmp_path):
    from graphids.core.data.datasets.can_bus import infer_attack_type

    cases = {
        "benign/normal_driving.csv": 0,
        "test/dos_attack.csv": 1,
        "attacks/fuzzing_data.csv": 2,
        "test/gear_spoof.csv": 3,
        "test/rpm_attack.csv": 4,
        "unknown/unknown_file.csv": 0,
    }
    for rel, expected in cases.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        assert infer_attack_type(path) == expected


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

    def test_vocab_digest_change_rebuilds_cached_tensors(self, tmp_path):
        from graphids.core.data.datasets.can_bus import CANBusDataset
        from graphids.core.data.preprocessing.representations import (
            SnapshotRepresentationCfg,
        )
        from graphids.core.data.preprocessing.vocab import vocab_digest

        raw_subdir = tmp_path / "raw" / "train"
        raw_subdir.mkdir(parents=True)
        pl.DataFrame(
            {
                "timestamp": np.arange(20, dtype=np.float64),
                "arb_id": ["0x001"] * 20,
                "data_field": ["AA" * 8] * 20,
                "attack": [0] * 20,
            }
        ).write_csv(raw_subdir / "normal.csv")

        common = dict(
            root=str(tmp_path / "processed"),
            raw_dir=str(tmp_path / "raw"),
            split="train",
            val_fraction=0.2,
            source_dirs=["train"],
            representation_cfg=SnapshotRepresentationCfg(window_size=10, stride=10),
        )
        first_vocab = {"0x001": 1}
        first = CANBusDataset(
            **common,
            shared_vocab=first_vocab,
            shared_vocab_digest=vocab_digest(first_vocab),
        )
        assert first._data.node_id.unique().tolist() == [1]

        second_vocab = {"0x001": 2}
        second = CANBusDataset(
            **common,
            shared_vocab=second_vocab,
            shared_vocab_digest=vocab_digest(second_vocab),
        )
        assert second._data.node_id.unique().tolist() == [2]
