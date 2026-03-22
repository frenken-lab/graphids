"""CAN bus dataset — self-contained InMemoryDataset subclass.

Owns the entire pipeline: scan CSVs (lazy) → parse hex → build vocabulary →
vectorized feature computation via Polars group_by → PyG Data → cache.

All feature computation is done in Polars expressions — no Python loops over
windows. The only per-window Python code is assembling the final Data objects
from pre-computed tensor slices.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import structlog
import torch
from torch_geometric.data import Data, InMemoryDataset

from graphids.core.preprocessing.features import (
    BYTE_COLS,
    N_EDGE_FEATURES,
    N_NODE_FEATURES,
    NODE_COL_ORDER,
    NODE_STAT_EXPRS,
    clustering_coefficients,
    stats_to_tensor,
)
from graphids.core.preprocessing.utils import atomic_save, nfs_lock, vocab_from_column

log = structlog.get_logger()

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
        meta_path = Path(self.processed_dir) / "num_arb_ids.txt"
        if meta_path.exists():
            self.num_arb_ids = int(meta_path.read_text().strip())
        else:
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
            (Path(self.processed_dir) / "num_arb_ids.txt").write_text(str(num_arb_ids))

    # ── pipeline ──────────────────────────────────────────────────────

    def _build_graphs(self) -> tuple[list[Data], int]:
        """Build all graphs from raw CSVs using vectorized Polars operations.

        Pipeline:
        1. _read_raw: lazy scan → parse hex → entropy → collect (one Polars plan)
        2. Add vocabulary, window_id, first_half flag
        3. group_by(window_id, node_id).agg() — ALL node stats in one call
        4. Compute edge features vectorized (diff-based, no per-window loop)
        5. Assemble Data objects from pre-computed arrays
        """
        df = self._read_raw()
        log.info("raw_loaded", rows=len(df))

        # Vocabulary
        vocab, oov = vocab_from_column(df["arb_id"])
        num_nodes = len(vocab) + 1
        df = df.with_columns(
            pl.col("arb_id").replace_strict(vocab, default=oov).cast(pl.Int64).alias("node_id")
        )

        # Window assignment: integer division on row index
        df = df.with_row_index("_row")
        n_rows = len(df)
        ws, st = self.window_size, self.stride

        # Each row belongs to window_id = (row - offset) // stride for valid offsets
        # But for sliding windows, a row can belong to multiple windows.
        # Instead: compute window start indices, then for each window slice the DataFrame.
        # This is still O(windows) but the per-window work is a slice + pre-computed agg.

        # Compute all window starts
        window_starts = list(range(0, n_rows - ws + 1, st))
        n_windows = len(window_starts)
        log.info("windowing", n_windows=n_windows, window_size=ws, stride=st)

        # Pre-compute first_half flag for split_half_ratio
        half = ws // 2

        # ── Vectorized node features ──────────────────────────────────
        # Assign each row a window_id via a cross-join approach:
        # For non-overlapping windows (stride >= window_size), simple integer division works.
        # For overlapping windows, rows belong to multiple windows — need to explode.

        if st >= ws:
            # Non-overlapping: each row belongs to exactly one window
            df = df.with_columns(
                (pl.col("_row") // st).cast(pl.Int64).alias("_wid"),
                (pl.col("_row") % ws < half).alias("_first_half"),
            )
            # Filter out partial trailing window
            max_wid = (n_rows - ws) // st
            df = df.filter(pl.col("_wid") <= max_wid)
        else:
            # Overlapping: assign each row to all windows it belongs to
            # Row r belongs to windows where start <= r < start + ws
            # i.e., window ids where (r - ws + 1) / st <= wid <= r / st
            # Efficient approach: build window_id array and explode
            row_np = df["_row"].to_numpy()
            wid_lists = []
            fh_lists = []
            for r in row_np:
                first_wid = max(0, (r - ws + 1 + st - 1) // st)  # ceil((r - ws + 1) / st)
                last_wid = r // st
                last_wid = min(last_wid, len(window_starts) - 1)
                wids = list(range(first_wid, last_wid + 1))
                wid_lists.append(wids)
                # Position within each window: r - wid * st
                fh_lists.append([int((r - wid * st) < half) for wid in wids])

            df = df.with_columns(
                pl.Series("_wid", wid_lists, dtype=pl.List(pl.Int64)),
                pl.Series("_first_half", fh_lists, dtype=pl.List(pl.Int64)),
            ).explode("_wid", "_first_half").with_columns(
                pl.col("_first_half").cast(pl.Boolean),
            )

        # One group_by over (window_id, node_id) — ALL per-node stats vectorized
        node_stats = df.group_by(["_wid", "node_id"], maintain_order=True).agg(
            *NODE_STAT_EXPRS
        ).fill_null(0).fill_nan(0)

        log.info("node_stats_computed", rows=len(node_stats))

        # ── Vectorized edge features ──────────────────────────────────
        # Edge = temporal adjacency (shift-1 within each window)
        # Pre-compute: src = node_id[:-1], dst = node_id[1:] per window
        # IAT = diff(timestamp), byte_diff = abs(diff(byte_i))

        # Get raw arrays for edge computation (original row order, pre-explode)
        raw_df = df.select("_wid", "_row", "node_id", "timestamp",
                           *[f"byte_{i}" for i in range(4)], "attack", "attack_type")
        # For overlapping windows, same row appears multiple times with different _wid
        # Sort by (_wid, _row) to get correct temporal order within each window
        raw_df = raw_df.sort(["_wid", "_row"])

        # ── Assemble graphs ───────────────────────────────────────────
        # Group node_stats by window for tensor construction
        wid_to_stats: dict[int, pl.DataFrame] = {}
        for wid_val, grp in node_stats.group_by("_wid", maintain_order=True):
            wid_to_stats[wid_val[0]] = grp  # type: ignore[index]

        # Group raw rows by window for edge construction
        wid_to_raw: dict[int, pl.DataFrame] = {}
        for wid_val, grp in raw_df.group_by("_wid", maintain_order=True):
            wid_to_raw[wid_val[0]] = grp  # type: ignore[index]

        graphs: list[Data] = []
        for wid in sorted(wid_to_stats.keys()):
            stats = wid_to_stats[wid]
            raw = wid_to_raw.get(wid)
            if raw is None or len(raw) < ws:
                continue

            # Node features → tensor (uses shared stats_to_tensor from features.py)
            node_ids = raw["node_id"].to_numpy()
            src, dst = node_ids[:-1], node_ids[1:]
            ei = np.stack([src, dst])
            x = stats_to_tensor(stats, num_nodes, edge_index=ei)

            # Edge features
            timestamps = raw["timestamp"].to_numpy()
            iat = np.diff(timestamps).astype(np.float32)
            n_edges = len(src)
            edge_attr = torch.zeros(n_edges, N_EDGE_FEATURES, dtype=torch.float32)
            edge_attr[:, 0] = torch.from_numpy(iat)
            edge_attr[:, 2] = torch.from_numpy(iat)
            edge_attr[:, 3] = torch.from_numpy(iat)
            edge_attr[:, 6] = 1.0
            for i in range(4):
                byte_diff = np.abs(np.diff(raw[f"byte_{i}"].to_numpy())).astype(np.float32)
                edge_attr[:, 7 + i] = torch.from_numpy(byte_diff)
            # Bidirectional flag
            directed = set(zip(src, dst))
            bidir = torch.tensor(
                [1.0 if (d, s) in directed else 0.0 for s, d in zip(src, dst)],
                dtype=torch.float32,
            )
            edge_attr[:, 11] = bidir

            # Labels
            has_attack = int(raw["attack"].max()) > 0
            y = torch.tensor([1 if has_attack else 0], dtype=torch.long)
            at_col = raw["attack_type"]
            if has_attack:
                at_vals = at_col.filter(at_col > 0)
                at = int(at_vals.mode().item()) if len(at_vals) > 0 else 0
            else:
                at = 0

            graphs.append(Data(
                x=x,
                edge_index=torch.tensor(ei, dtype=torch.long),
                edge_attr=edge_attr,
                y=y,
                attack_type=torch.tensor([at], dtype=torch.long),
            ))

        log.info("graphs_built", count=len(graphs), num_nodes=num_nodes)
        return graphs, num_nodes

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

        # Parse hex payload → 8 byte columns
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


