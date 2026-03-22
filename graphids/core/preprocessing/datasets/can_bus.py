"""CAN bus dataset — self-contained InMemoryDataset subclass.

Owns the entire pipeline: scan CSVs (lazy) → parse hex → build vocabulary →
vectorized feature computation via Polars group_by → PyG Data → cache.

Feature computation uses Polars expressions throughout:
- Node stats: single group_by(window_id, node_id).agg() across all windows
- Edge features: diff().over("_wid") for IAT and byte diffs, self-join for bidir
- Labels: group_by("_wid").agg() for attack presence and type
- Window assignment: int_ranges + explode (no Python loop)

The only per-window Python code is the final Data assembly and networkx
clustering coefficients (graph-structure-dependent, not vectorizable).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import structlog
import torch
from torch_geometric.data import Data, InMemoryDataset

from graphids.core.preprocessing.features import (
    N_EDGE_FEATURES,
    NODE_COL_ORDER,
    NODE_STAT_EXPRS,
    clustering_coefficients,
    edge_features,
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
            marker = Path(self.processed_dir) / ".complete"
            if Path(self.processed_paths[0]).exists() and marker.exists():
                return
            data_list, num_arb_ids = self._build_graphs()
            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]
            self.save(data_list, self.processed_paths[0])
            (Path(self.processed_dir) / "num_arb_ids.txt").write_text(str(num_arb_ids))
            marker.write_text("ok")

    # ── pipeline ──────────────────────────────────────────────────────

    def _build_graphs(self) -> tuple[list[Data], int]:
        df = self._read_raw()
        log.info("raw_loaded", rows=len(df))

        # Vocabulary
        vocab, oov = vocab_from_column(df["arb_id"])
        num_nodes = len(vocab) + 1
        df = df.with_columns(
            pl.col("arb_id").replace_strict(vocab, default=oov).cast(pl.Int64).alias("node_id")
        )

        # ── Window assignment (pure Polars, no Python loop) ───────────
        df = df.with_row_index("_row")
        n_rows = len(df)
        ws, st = self.window_size, self.stride
        half = ws // 2

        n_windows = max(0, (n_rows - ws) // st + 1)
        if n_windows == 0:
            log.warning("no_complete_windows", n_rows=n_rows, window_size=ws)
            return [], num_nodes
        max_wid = n_windows - 1
        log.info("windowing", n_windows=n_windows, window_size=ws, stride=st)

        if st >= ws:
            df = df.with_columns(
                (pl.col("_row") // st).cast(pl.Int64).alias("_wid"),
                (pl.col("_row") % ws < half).alias("_first_half"),
            ).filter(pl.col("_wid") <= max_wid)
        else:
            row = pl.col("_row")
            first_wid = ((row - ws + st) // st).clip(lower_bound=0)
            last_wid = (row // st).clip(upper_bound=max_wid)
            df = (
                df.with_columns(
                    pl.int_ranges(first_wid, last_wid + 1, dtype=pl.Int64).alias("_wid"),
                )
                .explode("_wid")
                .with_columns(
                    ((row - pl.col("_wid") * st) < half).alias("_first_half"),
                )
            )

        # ── Chunked processing ────────────────────────────────────────
        # Process windows in chunks to bound peak memory. Each chunk:
        # filter → stats + edges + labels (parallel in Polars) → assemble → free.
        CHUNK_SIZE = 500
        graphs: list[Data] = []

        for chunk_start in range(0, n_windows, CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, n_windows)
            chunk_df = df.filter(
                (pl.col("_wid") >= chunk_start) & (pl.col("_wid") < chunk_end)
            )

            chunk_graphs = self._process_chunk(chunk_df, num_nodes)
            graphs.extend(chunk_graphs)
            del chunk_df, chunk_graphs

            if chunk_end < n_windows:
                log.info("chunk_done", windows=f"{chunk_end}/{n_windows}")

        del df
        log.info("graphs_built", count=len(graphs), num_nodes=num_nodes)
        return graphs, num_nodes

    def _process_chunk(
        self, chunk_df: pl.DataFrame, num_nodes: int,
    ) -> list[Data]:
        """Build graphs for a chunk of windows. Frees intermediates eagerly."""
        # ── Node stats ────────────────────────────────────────────────
        node_stats = chunk_df.group_by(["_wid", "node_id"], maintain_order=True).agg(
            *NODE_STAT_EXPRS
        ).fill_null(0).fill_nan(0)

        # ── Edge features ─────────────────────────────────────────────
        edge_df = (
            chunk_df.select("_wid", "_row", "node_id", "timestamp",
                            *[f"byte_{i}" for i in range(4)])
            .sort(["_wid", "_row"])
            .with_columns(
                pl.col("node_id").alias("src"),
                pl.col("node_id").shift(-1).over("_wid").alias("dst"),
                pl.col("timestamp").diff().over("_wid").cast(pl.Float32).alias("iat"),
                *[
                    pl.col(f"byte_{i}").diff().abs().over("_wid").cast(pl.Float32)
                    .alias(f"byte_{i}_diff")
                    for i in range(4)
                ],
            )
            .filter(pl.col("iat").is_not_null())
        )
        # Bidirectional flag
        edge_pairs = edge_df.select("_wid", "src", "dst").unique()
        edge_df = (
            edge_df.join(
                edge_pairs.with_columns(pl.lit(True).alias("_rev")),
                left_on=["_wid", "dst", "src"],
                right_on=["_wid", "src", "dst"],
                how="left",
            )
            .with_columns(pl.col("_rev").fill_null(False).cast(pl.Float32).alias("bidir"))
            .drop("_rev")
        )
        del edge_pairs

        # ── Labels ────────────────────────────────────────────────────
        labels = chunk_df.group_by("_wid").agg(
            (pl.col("attack").max() > 0).cast(pl.Int64).alias("y"),
            pl.col("attack_type")
            .filter(pl.col("attack_type") > 0)
            .mode().first().fill_null(0).alias("at"),
        )
        label_dict: dict[int, tuple[int, int]] = {
            row[0]: (row[1], row[2]) for row in labels.iter_rows()
        }
        del labels

        # ── Partition + assemble ──────────────────────────────────────
        stats_parts = node_stats.partition_by("_wid", maintain_order=True, as_dict=True)
        del node_stats
        edge_parts = edge_df.partition_by("_wid", maintain_order=True, as_dict=True)
        del edge_df

        graphs: list[Data] = []
        for wid_key, stats in stats_parts.items():
            wid = wid_key[0] if isinstance(wid_key, tuple) else wid_key
            epart = edge_parts.get(wid_key)
            if epart is None:
                continue

            src = epart["src"].to_numpy()
            dst = epart["dst"].to_numpy()
            ei = np.stack([src, dst])

            x = stats_to_tensor(stats, num_nodes, edge_index=ei)

            n_edges = len(src)
            edge_attr = torch.zeros(n_edges, N_EDGE_FEATURES, dtype=torch.float32)
            iat = torch.from_numpy(epart["iat"].to_numpy().copy())
            edge_attr[:, 0] = iat
            edge_attr[:, 2] = iat
            edge_attr[:, 3] = iat
            edge_attr[:, 6] = 1.0
            for i in range(4):
                edge_attr[:, 7 + i] = torch.from_numpy(
                    epart[f"byte_{i}_diff"].to_numpy().copy()
                )
            edge_attr[:, 11] = torch.from_numpy(epart["bidir"].to_numpy().copy())

            y_val, at_val = label_dict.get(wid, (0, 0))
            graphs.append(Data(
                x=x,
                edge_index=torch.tensor(ei, dtype=torch.long),
                edge_attr=edge_attr,
                y=torch.tensor([y_val], dtype=torch.long),
                attack_type=torch.tensor([at_val], dtype=torch.long),
            ))

        del stats_parts, edge_parts
        return graphs

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
