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


def clustering_coefficients(edge_index: np.ndarray, num_nodes: int) -> np.ndarray:
    """Clustering coefficient per node via NetworkX (C-optimized).

    NetworkX is the standard implementation for this metric. For our typical
    CAN bus graphs (20-30 nodes), it's ~0.65ms/call — equivalent to custom
    sparse matrix approaches, without maintaining custom math.
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

    # Sequential collection reduces peak memory by ~20-30GB vs collect_all().
    # All three lazy frames reference `lf` → `df`, so df/lf must survive until
    # the last collect. labels is tiny; collect it first.
    labels = labels_lf.collect()
    del labels_lf
    node_stats = stats_lf.collect()
    del stats_lf
    edge_df = edges_base.collect()
    del edges_base, df, lf
    log.info("features_computed", stat_rows=len(node_stats), edge_rows=len(edge_df))

    # Bidirectional flag (needs materialized edge_df)
    edge_df = add_bidir_flag(edge_df)

    # ── Graph structure features (vectorized Polars) ──────────────
    # In/out degree from directed edges
    in_deg = edge_df.group_by(["_wid", "dst"]).agg(
        pl.len().cast(pl.Float32).alias("in_degree")
    ).rename({"dst": "node_id"})
    out_deg = edge_df.group_by(["_wid", "src"]).agg(
        pl.len().cast(pl.Float32).alias("out_degree")
    ).rename({"src": "node_id"})

    # Clustering coefficient via triangle counting on undirected edges
    edge_pairs = pl.concat([
        edge_df.select("_wid", pl.col("src").alias("u"), pl.col("dst").alias("v")),
        edge_df.select("_wid", pl.col("dst").alias("u"), pl.col("src").alias("v")),
    ]).unique(["_wid", "u", "v"])

    # 2-paths: u─v─w
    two_paths = edge_pairs.join(
        edge_pairs.select("_wid", pl.col("u").alias("_mid"), pl.col("v").alias("w")),
        left_on=["_wid", "v"], right_on=["_wid", "_mid"], how="inner",
    ).filter(pl.col("u") != pl.col("w"))

    # Close triangles: keep 2-paths where u─w edge exists
    tri = two_paths.join(
        edge_pairs, left_on=["_wid", "u", "w"], right_on=["_wid", "u", "v"], how="semi",
    )
    del two_paths

    tri_per_node = tri.group_by(["_wid", "u"]).agg(
        (pl.len() / 2).cast(pl.Float32).alias("_tri")
    ).rename({"u": "node_id"})
    del tri

    undirected_deg = edge_pairs.group_by(["_wid", "u"]).agg(
        pl.len().cast(pl.Float32).alias("_undeg")
    ).rename({"u": "node_id"})
    del edge_pairs

    cc = (
        tri_per_node.join(undirected_deg, on=["_wid", "node_id"], how="right")
        .fill_null(0)
        .with_columns(
            pl.when(pl.col("_undeg") > 1)
            .then(2.0 * pl.col("_tri") / (pl.col("_undeg") * (pl.col("_undeg") - 1)))
            .otherwise(0.0)
            .cast(pl.Float32)
            .alias("clustering_coeff")
        )
        .select("_wid", "node_id", "clustering_coeff")
    )
    del tri_per_node, undirected_deg

    # Overwrite placeholder columns in node_stats
    node_stats = (
        node_stats
        .join(cc, on=["_wid", "node_id"], how="left", suffix="_computed")
        .join(in_deg, on=["_wid", "node_id"], how="left", suffix="_computed")
        .join(out_deg, on=["_wid", "node_id"], how="left", suffix="_computed")
        .with_columns(
            pl.col("clustering_coeff_computed").fill_null(0.0).alias("clustering_coeff"),
            pl.col("in_degree_computed").fill_null(0.0).alias("in_degree"),
            pl.col("out_degree_computed").fill_null(0.0).alias("out_degree"),
        )
        .drop("clustering_coeff_computed", "in_degree_computed", "out_degree_computed")
    )
    del cc, in_deg, out_deg
    log.info("graph_structure_features_computed")

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

    # ── Vectorized local ID computation (Polars) ─────────────────
    # Assign 0-based local IDs per window matching RLE row order,
    # then join to edge_df so src/dst get local indices in bulk.
    local_ids = np.concatenate([np.arange(c, dtype=np.int64) for c in s_counts])
    node_stats = node_stats.with_columns(pl.Series("_local_id", local_ids))

    id_map = node_stats.select("_wid", "node_id", "_local_id")
    edge_df = (
        edge_df
        .join(
            id_map.rename({"node_id": "src", "_local_id": "src_local"}),
            on=["_wid", "src"], how="left",
        )
        .join(
            id_map.rename({"node_id": "dst", "_local_id": "dst_local"}),
            on=["_wid", "dst"], how="left",
        )
    )
    del id_map

    # Recompute edge boundaries after join (left join preserves row order)
    e_wids, e_starts, e_counts = _rle_boundaries(edge_df)
    e_lookup: dict[int, tuple[int, int]] = dict(zip(e_wids, zip(e_starts, e_counts)))

    # ── Bulk Polars → torch handoff ──────────────────────────────
    all_node_feats = (
        node_stats.select(NODE_COL_ORDER)
        .fill_null(0).fill_nan(0)
        .to_torch(dtype=pl.Float32)
    )
    all_node_ids = torch.from_numpy(
        node_stats["node_id"].cast(pl.Int64).to_numpy().copy()
    )
    all_edge_src = torch.from_numpy(
        edge_df["src_local"].cast(pl.Int64).to_numpy().copy()
    )
    all_edge_dst = torch.from_numpy(
        edge_df["dst_local"].cast(pl.Int64).to_numpy().copy()
    )
    all_edge_feats = (
        edge_df.select(list(EDGE_COL_ORDER))
        .fill_null(0).fill_nan(0)
        .to_torch(dtype=pl.Float32)
    )
    del node_stats, edge_df

    # ── Build Data objects (direct slicing) ──────────────────────
    graphs: list[Data] = []
    for i, wid in enumerate(s_wids):
        e_entry = e_lookup.get(wid)
        if e_entry is None:
            continue
        ss, sc = s_starts[i], s_counts[i]
        es, ec = e_entry
        graphs.append(Data(
            x=all_node_feats[ss:ss + sc].clone(),
            edge_index=torch.stack([all_edge_src[es:es + ec], all_edge_dst[es:es + ec]]),
            edge_attr=all_edge_feats[es:es + ec].clone(),
            node_id=all_node_ids[ss:ss + sc].clone(),
            y=torch.tensor([label_y.get(wid, 0)], dtype=torch.long),
            attack_type=torch.tensor([label_at.get(wid, 0)], dtype=torch.long),
        ))
    log.info("graphs_built", count=len(graphs))
    return graphs
