"""CAN bus graph feature definitions and sliding-window graph construction.

Single source of truth for node and edge feature schemas, Polars expressions,
assembly functions, and the general sliding-window-to-graph pipeline.
Dataset adapters (e.g. can_bus.py) handle I/O and vocabulary, then call
sliding_window_graphs() for the general pipeline.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import structlog
import torch
from torch import Tensor
from torch_geometric.data import Data

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
# Edge feature layout: iat + 8 byte diffs + bidirectional flag.
EDGE_COL_ORDER = (
    "iat",
    *(f"byte_{i}_diff" for i in range(8)),
    "bidir",
    "edge_freq",
)

N_EDGE_FEATURES = len(EDGE_COL_ORDER)  # 10

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


def clustering_coefficients(edge_index: np.ndarray, num_nodes: int) -> np.ndarray:
    """Clustering coefficient per node via networkx (C-optimized)."""
    import networkx as nx

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
        cc_idx = NODE_COL_ORDER.index("clustering_coeff")
        x[:, cc_idx] = torch.from_numpy(clustering_coefficients(edge_index, n_active))
        in_deg_idx = NODE_COL_ORDER.index("in_degree")
        out_deg_idx = NODE_COL_ORDER.index("out_degree")
        ei = edge_index.astype(np.intp)
        x[:, in_deg_idx] = torch.from_numpy(np.bincount(ei[1], minlength=n_active).astype(np.float32))
        x[:, out_deg_idx] = torch.from_numpy(np.bincount(ei[0], minlength=n_active).astype(np.float32))

    return x, node_ids


def node_features(
    window: pl.DataFrame,
    edge_index: np.ndarray | None = None,
) -> tuple[Tensor, Tensor]:
    """Compute compact node features from a single window DataFrame.

    Returns (x, node_ids) — same contract as stats_to_tensor.
    This is the per-window path used by tests and standalone usage.
    The vectorized batch path in can_bus.py uses NODE_STAT_EXPRS directly.
    """
    half = len(window) // 2
    window = window.with_row_index("_row").with_columns(
        (pl.col("_row") < half).alias("_first_half")
    )
    stats = window.group_by("node_id").agg(*NODE_STAT_EXPRS).fill_null(0).fill_nan(0)
    return stats_to_tensor(stats, edge_index)


def add_bidir_flag(edge_df: pl.DataFrame) -> pl.DataFrame:
    """Add bidirectional flag to an edge DataFrame.

    For each edge (src→dst) in a window, bidir=1.0 if the reverse edge
    (dst→src) also exists in that window, else 0.0. Requires columns:
    _wid, src, dst. Returns the DataFrame with a 'bidir' Float32 column.
    """
    edge_pairs = edge_df.select("_wid", "src", "dst").unique()
    return (
        edge_df.join(
            edge_pairs.with_columns(pl.lit(True).alias("_rev")),
            left_on=["_wid", "dst", "src"],
            right_on=["_wid", "src", "dst"],
            how="left",
        )
        .with_columns(pl.col("_rev").fill_null(False).cast(pl.Float32).alias("bidir"))
        .drop("_rev")
    )


def edge_to_tensor(edge_df: pl.DataFrame) -> Tensor:
    """Convert an edge DataFrame with EDGE_COL_ORDER columns to a [n_edges, N_EDGE_FEATURES] tensor.

    Assembly function parallel to stats_to_tensor — used by the vectorized batch path.
    """
    return edge_df.select(list(EDGE_COL_ORDER)).fill_null(0).fill_nan(0).to_torch(
        dtype=pl.Float32,
    )


def edge_features(
    timestamps: np.ndarray,
    byte_arrays: list[np.ndarray],
    src: np.ndarray,
    dst: np.ndarray,
) -> Tensor:
    """Compute edge feature tensor from raw numpy arrays.

    Layout: iat | byte_0_diff..byte_7_diff | bidir | edge_freq
    Per-window entry point for any CAN bus dataset. The vectorized batch path
    in can_bus.py uses EDGE_STAT_EXPRS + edge_to_tensor() instead.
    """
    n = len(src)
    out = torch.zeros(n, N_EDGE_FEATURES, dtype=torch.float32)
    if n == 0:
        return out

    # IAT (slot 0)
    iat = torch.from_numpy(np.diff(timestamps).astype(np.float32))
    out[:, 0] = iat

    # Byte diffs (slots 1-8)
    for i in range(min(8, len(byte_arrays))):
        out[:, 1 + i] = torch.from_numpy(
            np.abs(np.diff(byte_arrays[i])).astype(np.float32)
        )

    # Bidirectional flag (slot 9)
    directed = set(zip(src, dst))
    bidir = torch.tensor(
        [1.0 if (d, s) in directed else 0.0 for s, d in zip(src, dst)],
        dtype=torch.float32,
    )
    out[:, 9] = bidir

    # Edge frequency (slot 10) — count of edges with same (src, dst) pair
    pairs = np.stack([src, dst], axis=1)
    _, inverse, counts = np.unique(pairs, axis=0, return_inverse=True, return_counts=True)
    out[:, 10] = torch.from_numpy(counts[inverse].astype(np.float32))

    return out


def sliding_window_graphs(
    df: pl.DataFrame,
    window_size: int,
    stride: int,
) -> list[Data]:
    """Convert a message DataFrame into PyG Data graphs via sliding windows.

    Produces compact graphs: x is [n_active, N_NODE_FEATURES] (only active nodes),
    edge_index uses local IDs (0..n_active-1), and node_id stores the
    global CAN ID indices for embedding lookup.

    Required columns: node_id (Int64), timestamp, byte_0..7, entropy,
    attack, attack_type.
    """
    log = structlog.get_logger()

    # ── Window assignment (pure Polars, no Python loop) ───────────
    df = df.with_row_index("_row")
    n_rows = len(df)
    ws, st = window_size, stride
    half = ws // 2

    n_windows = max(0, (n_rows - ws) // st + 1)
    if n_windows == 0:
        log.warning("no_complete_windows", n_rows=n_rows, window_size=ws)
        return []
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

    # ── Parallel feature computation (single scan of df) ──────────
    lf = df.lazy()

    stats_lf = (
        lf.group_by(["_wid", "node_id"], maintain_order=True)
        .agg(*NODE_STAT_EXPRS)
        .fill_null(0).fill_nan(0)
    )

    edges_base = (
        lf.select("_wid", "_row", "node_id", "timestamp",
                   *[f"byte_{i}" for i in range(8)])
        .sort(["_wid", "_row"])
        .with_columns(
            pl.col("node_id").alias("src"),
            pl.col("node_id").shift(-1).over("_wid").alias("dst"),
            *EDGE_STAT_EXPRS,
        )
        .filter(pl.col("iat").is_not_null() & pl.col("dst").is_not_null())
        .with_columns(
            pl.len().over(["_wid", "src", "dst"]).cast(pl.Float32).alias("edge_freq"),
        )
    )

    labels_lf = lf.group_by("_wid").agg(
        (pl.col("attack").max() > 0).cast(pl.Int64).alias("y"),
        pl.col("attack_type")
        .filter(pl.col("attack_type") > 0)
        .mode().first().fill_null(0).alias("at"),
    )

    node_stats, edge_df, labels = pl.collect_all([stats_lf, edges_base, labels_lf])
    del df, lf
    log.info("features_computed", stat_rows=len(node_stats), edge_rows=len(edge_df))

    # Bidirectional flag (needs materialized edge_df)
    edge_df = add_bidir_flag(edge_df)

    label_y = dict(zip(labels["_wid"].to_list(), labels["y"].to_list()))
    label_at = dict(zip(labels["_wid"].to_list(), labels["at"].to_list()))
    del labels

    # ── Slice boundaries via RLE ──────────────────────────────────
    def _rle_boundaries(frame: pl.DataFrame) -> tuple[list[int], list[int], list[int]]:
        rle = frame["_wid"].rle().struct.unnest()
        wids = rle["value"].to_list()
        counts = rle["len"].to_list()
        starts = (rle["len"].cum_sum() - rle["len"]).to_list()
        return wids, starts, counts

    s_wids, s_starts, s_counts = _rle_boundaries(node_stats)
    e_wids, e_starts, e_counts = _rle_boundaries(edge_df)
    e_lookup: dict[int, tuple[int, int]] = dict(zip(e_wids, zip(e_starts, e_counts)))

    # ── Build graphs ──────────────────────────────────────────────
    graphs: list[Data] = []
    for i, wid in enumerate(s_wids):
        e_entry = e_lookup.get(wid)
        if e_entry is None:
            continue

        ss, sc = s_starts[i], s_counts[i]
        es, ec = e_entry

        stats_slice = node_stats[ss:ss + sc]
        edge_slice = edge_df[es:es + ec]

        # Global→local ID remapping (Polars replace_strict — no Python row loop)
        node_ids_list = stats_slice["node_id"].to_list()
        global_to_local = {gid: i for i, gid in enumerate(node_ids_list)}
        edge_slice = edge_slice.with_columns(
            pl.col("src").replace_strict(global_to_local).cast(pl.Int64).alias("src"),
            pl.col("dst").replace_strict(global_to_local).cast(pl.Int64).alias("dst"),
        )

        ei = edge_slice.select("src", "dst").to_torch(dtype=pl.Int64).T
        x, node_ids = stats_to_tensor(stats_slice, edge_index=ei.numpy())
        edge_attr = edge_to_tensor(edge_slice)

        y_val = label_y.get(wid, 0)
        at_val = label_at.get(wid, 0)
        graphs.append(Data(
            x=x,
            edge_index=ei,
            edge_attr=edge_attr,
            node_id=node_ids,
            y=torch.tensor([y_val], dtype=torch.long),
            attack_type=torch.tensor([at_val], dtype=torch.long),
        ))

    del node_stats, edge_df
    log.info("graphs_built", count=len(graphs))
    return graphs
