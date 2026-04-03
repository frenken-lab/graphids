"""CAN bus dataset — InMemoryDataset subclass.

Handles I/O (CSV scanning, hex parsing, vocabulary) and delegates the
general sliding-window-to-graph pipeline to features.sliding_window_graphs().
"""

from __future__ import annotations

import os
from pathlib import Path

import polars as pl
from graphids.log import get_logger
import torch
from torch_geometric.data import Data, InMemoryDataset

from graphids.core.preprocessing.features import parse_payload, sliding_window_graphs
from graphids.core.preprocessing.utils import atomic_save, nfs_lock, vocab_from_column

log = get_logger(__name__)

ATTACK_TYPE_CODES: dict[str, int] = {
    "normal": 0,
    "attack_free": 0,
    "benign": 0,
    "dos": 1,
    "fuzzy": 2,
    "fuzzing": 2,
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
        # Always derive from actual data — num_arb_ids.txt can be stale/corrupted
        self.num_arb_ids = int(self._data.node_id.max().item()) + 1

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

    def process(self) -> None:
        lock_path = Path(self.processed_dir) / ".lock"
        with nfs_lock(lock_path):
            marker = Path(self.processed_dir) / ".complete"
            if Path(self.processed_paths[0]).exists() and marker.exists():
                return
            data, slices, num_arb_ids, num_graphs = self._build_graphs()
            atomic_save([data, slices], Path(self.processed_paths[0]))
            (Path(self.processed_dir) / "num_arb_ids.txt").write_text(str(num_arb_ids))
            self._write_cache_metadata(slices, num_graphs)
            marker.write_text("ok")

    def _write_cache_metadata(self, slices: dict, num_graphs: int) -> None:
        """Write graph statistics to cache_metadata.json for DynamicBatchSampler."""
        import json
        import tempfile

        node_t = (slices["x"][1:] - slices["x"][:-1]).float()
        edge_t = (slices["edge_index"][1:] - slices["edge_index"][:-1]).float()

        meta = {
            "window_size": self.window_size,
            "stride": self.stride,
            "num_graphs": num_graphs,
            "graph_stats": {
                "node_count": {
                    "min": int(node_t.min().item()),
                    "max": int(node_t.max().item()),
                    "mean": float(node_t.mean().item()),
                    "p95": float(node_t.quantile(0.95).item()),
                    "p99": float(node_t.quantile(0.99).item()),
                },
                "edge_count": {
                    "min": int(edge_t.min().item()),
                    "max": int(edge_t.max().item()),
                    "mean": float(edge_t.mean().item()),
                    "p95": float(edge_t.quantile(0.95).item()),
                    "p99": float(edge_t.quantile(0.99).item()),
                },
            },
        }
        # NFS-safe atomic write: tmpfile → fsync → rename
        out_path = Path(self.root) / "cache_metadata.json"
        fd, tmp = tempfile.mkstemp(dir=out_path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(meta, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            fd = -1  # owned by fdopen now
            os.rename(tmp, out_path)
        except BaseException:
            if fd >= 0:
                os.close(fd)
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
        log.info("cache_metadata_written", path=str(out_path), num_graphs=num_graphs)

    # ── pipeline ──────────────────────────────────────────────────────

    def _build_graphs(self) -> tuple[Data, dict, int, int]:
        df = self._read_raw()
        log.info("raw_loaded", rows=len(df))

        # Vocabulary
        vocab, oov = vocab_from_column(df["arb_id"])
        num_arb_ids = len(vocab) + 1  # global vocab size for embedding table
        df = df.with_columns(
            pl.col("arb_id").replace_strict(vocab, default=oov).cast(pl.Int64).alias("node_id")
        )

        data, slices, num_graphs = sliding_window_graphs(df, self.window_size, self.stride)
        del df
        return data, slices, num_arb_ids, num_graphs

    @staticmethod
    def _infer_attack_type(csv_path: Path) -> int:
        """Infer attack type code from file/directory naming conventions."""
        parts = (csv_path.stem.lower() + " " + csv_path.parent.name.lower())
        for keyword, code in ATTACK_TYPE_CODES.items():
            if keyword in parts:
                return code
        return 0

    def _read_raw(self) -> pl.DataFrame:
        """Lazy-scan CSVs, parse hex, compute entropy, tag attack types. Collect once."""
        frames = []
        for csv_path in sorted(self.raw_data_dir.rglob("*.csv")):
            at = self._infer_attack_type(csv_path)
            lf = (
                pl.scan_csv(csv_path)
                .with_columns(pl.lit(at).alias("attack_type"))
            )
            frames.append(lf)

        combined = pl.concat(frames).sort("timestamp")

        # Normalize column names: HCRL CSVs use different names than our schema
        col_names = combined.collect_schema().names()
        renames = {}
        if "arbitration_id" in col_names:
            renames["arbitration_id"] = "arb_id"
        if "data_field" in col_names:
            renames["data_field"] = "payload"
        if renames:
            combined = combined.rename(renames)

        combined = parse_payload(combined)

        return combined.collect()
