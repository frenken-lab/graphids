"""CAN bus dataset — I/O, vocab, and feature schema.

Everything CAN-bus-specific lives here: hex payload parsing, byte-column
feature expressions, attack-type taxonomy, and the ``CANBusDataset`` adapter.
The general sliding-window pipeline lives in ``graph_pipeline.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import polars as pl
import torch
from graphids.log import get_logger
from torch import Tensor
from torch_geometric.data import Data, InMemoryDataset

from graphids.core.preprocessing.graph_pipeline import sliding_window_graphs
from graphids.core.preprocessing.io import atomic_save, nfs_lock, vocab_from_column

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Attack-type taxonomy (CAN bus specific)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# CAN feature schema — column layouts, Polars expressions, helper fns
# ---------------------------------------------------------------------------

BYTE_COLS = [f"byte_{i}" for i in range(8)]

# Column order defines tensor layout. Changing order changes model input.
NODE_COL_ORDER = (
    [f"{c}_mean" for c in BYTE_COLS]
    + [f"{c}_std" for c in BYTE_COLS]
    + [f"{c}_range" for c in BYTE_COLS]
    + ["msg_count", "entropy_mean", "skewness", "kurtosis",
       "clustering_coeff", "split_half_ratio", "change_rate",
       "node_iat_mean", "node_iat_std", "in_degree", "out_degree"]
)

N_NODE_FEATURES = len(NODE_COL_ORDER)
# Edge feature layout: iat + 8 byte diffs + bidirectional flag + freq.
EDGE_COL_ORDER = (
    "iat",
    *(f"byte_{i}_diff" for i in range(8)),
    "bidir",
    "edge_freq",
)

N_EDGE_FEATURES = len(EDGE_COL_ORDER)  # 11

# Column indices for post-hoc features filled from graph structure.
CC_IDX = NODE_COL_ORDER.index("clustering_coeff")
IN_DEG_IDX = NODE_COL_ORDER.index("in_degree")
OUT_DEG_IDX = NODE_COL_ORDER.index("out_degree")

# Polars aggregation expressions for per-node stats within a window.
# Used by group_by("node_id").agg() and group_by(["_wid", "node_id"]).agg().
# Requires columns: byte_0..7, entropy, _first_half (bool).
NODE_STAT_EXPRS: list[pl.Expr] = [
    *[pl.col(c).mean().alias(f"{c}_mean") for c in BYTE_COLS],
    *[pl.col(c).std().alias(f"{c}_std") for c in BYTE_COLS],
    *[(pl.col(c).max() - pl.col(c).min()).alias(f"{c}_range") for c in BYTE_COLS],
    pl.len().cast(pl.Float32).alias("msg_count"),
    pl.col("entropy").mean().alias("entropy_mean"),
    pl.mean_horizontal(*[pl.col(c).skew().fill_nan(0).clip(-10, 10) for c in BYTE_COLS]).alias("skewness"),
    pl.mean_horizontal(*[pl.col(c).kurtosis().fill_nan(0).clip(-10, 10) for c in BYTE_COLS]).alias("kurtosis"),
    pl.lit(0.0).alias("clustering_coeff"),  # filled per-window from graph structure
    pl.col("_first_half").mean().alias("split_half_ratio"),
    pl.mean_horizontal(*[(pl.col(c).diff().abs().drop_nulls() > 0).mean() for c in BYTE_COLS]).alias("change_rate"),
    pl.col("timestamp").diff().mean().cast(pl.Float32).alias("node_iat_mean"),
    pl.col("timestamp").diff().std().fill_nan(0).cast(pl.Float32).alias("node_iat_std"),
    pl.lit(0.0).alias("in_degree"),   # filled post-hoc from edge_index
    pl.lit(0.0).alias("out_degree"),  # filled post-hoc from edge_index
]

# Polars expressions for vectorized edge feature computation.
# Used by with_columns() after sort(["_wid", "_row"]).
# Requires columns: timestamp, byte_0..7, _wid.
# Note: bidir is computed separately via self-join (not expressible as a single expression).
EDGE_STAT_EXPRS: list[pl.Expr] = [
    pl.col("timestamp").diff().over("_wid").cast(pl.Float32).alias("iat"),
    *[
        pl.col(f"byte_{i}").diff().abs().over("_wid").cast(pl.Float32)
        .alias(f"byte_{i}_diff")
        for i in range(8)
    ],
]

# Label aggregations per window: y (binary attack) + attack_type (multiclass).
LABEL_EXPRS: list[pl.Expr] = [
    (pl.col("attack").max() > 0).cast(pl.Int64).alias("y"),
    pl.col("attack_type")
    .filter(pl.col("attack_type") > 0)
    .mode().first().fill_null(0).alias("attack_type"),
]

# Columns required by edge-feature computation (byte diffs need byte_0..7).
EDGE_BASE_COLS: list[str] = [f"byte_{i}" for i in range(8)]


def parse_payload(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Parse hex payload column into 8 byte columns + Shannon entropy.

    Expects a 'payload' column (16-char hex string). Adds byte_0..byte_7
    (Float32) and entropy (Float32). Passthrough if byte_0 already exists.
    """
    if "byte_0" in lf.collect_schema().names():
        return lf
    byte_exprs = [
        pl.col("payload").str.slice(i * 2, 2)
        .str.to_integer(base=16, strict=False)
        .fill_null(0).cast(pl.Float32).alias(f"byte_{i}")
        for i in range(8)
    ]
    lf = lf.with_columns(byte_exprs)
    byte_cols = [pl.col(f"byte_{i}") for i in range(8)]
    row_sum = pl.sum_horizontal(byte_cols).clip(1e-12, None)
    entropy_terms = [
        pl.when(c > 0).then(-(c / row_sum) * (c / row_sum).log()).otherwise(0.0)
        for c in byte_cols
    ]
    return lf.with_columns(pl.sum_horizontal(entropy_terms).alias("entropy"))


def clustering_coefficients(edge_index: np.ndarray, num_nodes: int) -> np.ndarray:
    """Clustering coefficient per node via NetworkX (C-optimized).

    NetworkX is the standard implementation for this metric. For our typical
    CAN bus graphs (20-30 nodes), it's ~0.65ms/call — equivalent to custom
    sparse matrix approaches, without maintaining custom math. Used by the
    per-window convenience path (tests and standalone), not by the bulk
    pipeline which computes clustering via Polars triangle counting.
    """
    import networkx as nx

    if num_nodes == 0 or edge_index.shape[1] == 0:
        return np.zeros(num_nodes, dtype=np.float32)

    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(zip(edge_index[0], edge_index[1]))
    cc = nx.clustering(G)
    return np.array([cc.get(i, 0.0) for i in range(num_nodes)], dtype=np.float32)


def stats_to_tensor(
    stats: pl.DataFrame, edge_index: np.ndarray | None = None,
) -> tuple[Tensor, Tensor]:
    """Convert per-node stats to compact [n_active, N_NODE_FEATURES] tensor.

    Returns (x, node_ids) where node_ids are global CAN ID indices.
    edge_index must use LOCAL indices (0..n_active-1).
    """
    n_active = len(stats)
    if n_active == 0:
        return torch.zeros(0, N_NODE_FEATURES, dtype=torch.float32), torch.zeros(0, dtype=torch.int64)

    node_ids = torch.from_numpy(stats["node_id"].cast(pl.Int64).to_numpy().copy())
    x = (
        stats.select(NODE_COL_ORDER)
        .cast({c: pl.Float32 for c in NODE_COL_ORDER})
        .fill_null(0).fill_nan(0)
        .to_torch(dtype=pl.Float32)
    )

    if edge_index is not None:
        x[:, CC_IDX] = torch.from_numpy(clustering_coefficients(edge_index, n_active))
        ei = edge_index.astype(np.intp)
        x[:, IN_DEG_IDX] = torch.from_numpy(np.bincount(ei[1], minlength=n_active).astype(np.float32))
        x[:, OUT_DEG_IDX] = torch.from_numpy(np.bincount(ei[0], minlength=n_active).astype(np.float32))

    return x, node_ids


def node_features(
    window: pl.DataFrame,
    edge_index: np.ndarray | None = None,
) -> tuple[Tensor, Tensor]:
    """Compute compact node features from a single window DataFrame.

    Returns (x, node_ids) — same contract as stats_to_tensor.
    This is the per-window convenience path used by tests and standalone
    usage. The vectorized batch path uses NODE_STAT_EXPRS directly.
    """
    half = len(window) // 2
    window = window.with_row_index("_row").with_columns(
        (pl.col("_row") < half).alias("_first_half")
    )
    stats = window.group_by("node_id").agg(*NODE_STAT_EXPRS).fill_null(0).fill_nan(0)
    return stats_to_tensor(stats, edge_index)


# ---------------------------------------------------------------------------
# CANBusDataset — InMemoryDataset adapter
# ---------------------------------------------------------------------------


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

    # ── Size tensors for NodeBudgetBatchSampler ───────────────────────
    # Derived from slices at zero I/O cost — the slice tensors are small
    # cumulative offsets (one int64 per graph + 1, ≈400KB for 50K graphs).
    # This lets the sampler walk sizes without reconstructing Data objects
    # per graph per epoch, which was 50K mmap reconstructions per epoch
    # under PyG's DynamicBatchSampler.__iter__.

    @property
    def num_nodes_per_graph(self) -> torch.Tensor:
        """Per-graph node counts for the current split (respects _indices)."""
        full = self.slices["x"][1:] - self.slices["x"][:-1]
        if self._indices is None:
            return full
        return full[torch.as_tensor(list(self._indices), dtype=torch.long)]

    @property
    def num_edges_per_graph(self) -> torch.Tensor:
        """Per-graph edge counts for the current split (respects _indices)."""
        full = self.slices["edge_index"][1:] - self.slices["edge_index"][:-1]
        if self._indices is None:
            return full
        return full[torch.as_tensor(list(self._indices), dtype=torch.long)]

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

        data, slices, num_graphs = sliding_window_graphs(
            df, self.window_size, self.stride,
            node_stat_exprs=NODE_STAT_EXPRS,
            edge_stat_exprs=EDGE_STAT_EXPRS,
            node_col_order=NODE_COL_ORDER,
            edge_col_order=EDGE_COL_ORDER,
            label_exprs=LABEL_EXPRS,
            edge_base_cols=EDGE_BASE_COLS,
        )
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
