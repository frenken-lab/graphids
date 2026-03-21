"""CAN bus dataset — self-contained InMemoryDataset subclass.

Owns the entire pipeline: scan CSVs (lazy) → parse hex → build vocabulary →
sliding windows via group_by_dynamic → compute features → PyG Data → cache.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch_geometric.data import Data, InMemoryDataset

from graphids.core.preprocessing.features import edge_features, node_features
from graphids.core.preprocessing.utils import atomic_save, nfs_lock, vocab_from_column

ATTACK_TYPE_CODES: dict[str, int] = {
    "normal": 0,
    "attack_free": 0,
    "benign": 0,
    "dos": 1,
    "fuzzy": 2,
    "gear": 3,
    "rpm": 4,
    "flooding": 5,
    "malfunction": 6,
}

ATTACK_TYPE_NAMES: dict[int, str] = {v: k for k, v in ATTACK_TYPE_CODES.items() if v != 0}
ATTACK_TYPE_NAMES[0] = "benign"


class CANBusDataset(InMemoryDataset):
    """CAN bus intrusion detection graph dataset.

    Each graph is one sliding window of CAN messages. Nodes are arbitration
    IDs, edges are temporal adjacency (shift-1).
    """

    def __init__(
        self,
        root: str | Path,
        raw_dir: str | Path,
        split: str = "train",
        val_fraction: float = 0.15,
        window_size: int = 50,
        stride: int = 25,
        seed: int = 42,
        transform=None,
        pre_transform=None,
    ):
        self.raw_data_dir = Path(raw_dir)
        self.split = split
        self.val_fraction = val_fraction
        self.window_size = window_size
        self.stride = stride
        self.seed = seed
        super().__init__(str(root), transform, pre_transform)
        self.load(self.processed_paths[0])
        self._load_num_arb_ids()

        if self.split in ("train", "val"):
            self._apply_train_val_split()

    @property
    def processed_file_names(self) -> list[str]:
        tag = "train" if self.split in ("train", "val") else "test"
        return [f"data_{tag}.pt"]

    def _load_num_arb_ids(self) -> None:
        meta_path = Path(self.processed_dir) / "num_arb_ids.txt"
        if meta_path.exists():
            self.num_arb_ids = int(meta_path.read_text().strip())
        else:
            # Fallback: max node count across all graphs (less precise but safe)
            self.num_arb_ids = max((g.x.shape[0] for g in self), default=0)

    def _apply_train_val_split(self) -> None:
        n = len(self)
        gen = torch.Generator().manual_seed(self.seed)
        perm = torch.randperm(n, generator=gen)
        n_val = int(n * self.val_fraction)
        self._indices = (perm[:n_val] if self.split == "val" else perm[n_val:]).tolist()

    # ── NFS-safe overrides ────────────────────────────────────────────

    def load(self, path: str, **kwargs):
        (self.data, self.slices) = torch.load(
            path, map_location="cpu", mmap=True, weights_only=False,
        )

    def save(self, data_list: list[Data], path: str) -> None:
        from torch_geometric.data import InMemoryDataset as _Base
        atomic_save(list(_Base.collate(data_list)), Path(path))

    def process(self) -> None:
        lock_path = Path(self.processed_dir) / ".lock"
        with nfs_lock(lock_path):
            if Path(self.processed_paths[0]).exists():
                return
            data_list, num_arb_ids = self._build_graphs()
            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]
            self.save(data_list, self.processed_paths[0])
            # Persist vocab size for embedding table construction
            (Path(self.processed_dir) / "num_arb_ids.txt").write_text(str(num_arb_ids))

    # ── pipeline ──────────────────────────────────────────────────────

    def _build_graphs(self) -> tuple[list[Data], int]:
        df = self._read_raw()

        # Vocabulary: vectorised .replace() instead of per-row Python lambda
        vocab, oov = vocab_from_column(df["arb_id"])
        df = df.with_columns(
            pl.col("arb_id").replace(vocab, default=oov).cast(pl.Int64).alias("node_id")
        )
        num_nodes = len(vocab) + 1

        # Add a row index for group_by_dynamic windowing
        df = df.with_row_index("_idx")

        # Sliding windows via group_by_dynamic on the row index.
        # every=stride, period=window_size gives the same windows as the
        # manual slice loop, but executed in Polars' Rust engine.
        groups = df.group_by_dynamic(
            "_idx",
            every=f"{self.stride}i",
            period=f"{self.window_size}i",
        )

        graphs: list[Data] = []
        for _key, window in groups:
            if len(window) < self.window_size:
                continue  # skip trailing partial window
            graphs.append(self._window_to_graph(window, num_nodes))
        return graphs, num_nodes

    def _read_raw(self) -> pl.DataFrame:
        """Lazy-scan CSVs, parse hex, compute entropy, tag attack types. Collect once."""
        frames = []
        for csv_path in sorted(self.raw_data_dir.rglob("*.csv")):
            at = ATTACK_TYPE_CODES.get(csv_path.parent.name.lower(), 1)
            lf = (
                pl.scan_csv(csv_path)
                .with_columns(pl.lit(at).alias("attack_type"))
            )
            frames.append(lf)

        combined = pl.concat(frames).sort("timestamp")

        # Parse hex payload → 8 byte columns (single with_columns, pushed into query plan)
        byte_exprs = [
            pl.col("payload")
            .str.slice(i * 2, 2)
            .str.to_integer(base=16, strict=False)
            .fill_null(0)
            .cast(pl.Float32)
            .alias(f"byte_{i}")
            for i in range(8)
        ]
        combined = combined.with_columns(byte_exprs)

        # Shannon entropy per message
        byte_cols = [pl.col(f"byte_{i}") for i in range(8)]
        row_sum = pl.sum_horizontal(byte_cols).clip(1e-12, None)
        entropy_terms = [
            pl.when(c > 0).then(-(c / row_sum) * (c / row_sum).log()).otherwise(0.0)
            for c in byte_cols
        ]
        combined = combined.with_columns(
            pl.sum_horizontal(entropy_terms).alias("entropy")
        )

        return combined.collect()

    def _window_to_graph(self, window: pl.DataFrame, num_nodes: int) -> Data:
        node_ids = window["node_id"].to_numpy()
        src, dst = node_ids[:-1], node_ids[1:]
        ei = np.stack([src, dst])

        timestamps = window["timestamp"].to_numpy()
        byte_arrays = [window[f"byte_{i}"].to_numpy() for i in range(4)]

        x = node_features(window, num_nodes, edge_index=ei)
        edge_attr = edge_features(timestamps, byte_arrays, src, dst)

        attack_col = window["attack_type"]
        has_attack = attack_col.max() > 0
        y = torch.tensor([1 if has_attack else 0], dtype=torch.long)

        if has_attack:
            at = int(attack_col.filter(attack_col > 0).mode().item())
        else:
            at = 0

        return Data(
            x=x,
            edge_index=torch.tensor(ei, dtype=torch.long),
            edge_attr=edge_attr,
            y=y,
            attack_type=torch.tensor([at], dtype=torch.long),
        )
