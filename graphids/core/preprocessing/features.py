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



def _assemble_chunk_numpy(
    node_feats: np.ndarray,
    node_ids: np.ndarray,
    edge_src: np.ndarray,
    edge_dst: np.ndarray,
    edge_feats: np.ndarray,
    window_specs: list[tuple[int, int, int, int, int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           list[tuple[int, int, int, int, int, int]]]:
    """Build graph arrays from pre-materialized numpy — no torch, no Data objects.

    Runs in worker processes. Returns pure numpy arrays so IPC uses standard
    pickle (raw memcpy) instead of torch's mmap/file_system mechanism.

    Returns
    -------
    (node_feats_out, node_ids_out, edge_src_out, edge_dst_out, edge_feats_out, specs_out)
        Concatenated numpy arrays for all windows in the chunk, with specs_out
        containing (s_start, s_count, e_start, e_count, y_val, at_val) tuples
        using LOCAL offsets into the returned arrays.
    """
    nf_parts: list[np.ndarray] = []
    ni_parts: list[np.ndarray] = []
    es_parts: list[np.ndarray] = []
    ed_parts: list[np.ndarray] = []
    ef_parts: list[np.ndarray] = []
    specs_out: list[tuple[int, int, int, int, int, int]] = []

    s_offset = 0
    e_offset = 0

    for ss, sc, es, ec, y_val, at_val in window_specs:
        nf = node_feats[ss:ss + sc].copy()
        nids = node_ids[ss:ss + sc].copy()
        esrc = edge_src[es:es + ec].copy()
        edst = edge_dst[es:es + ec].copy()
        ef = edge_feats[es:es + ec].copy()

        # Clustering coefficients (networkx, already numpy-native)
        ei_np = np.stack([esrc, edst]) if ec > 0 else np.empty((2, 0), dtype=np.int64)
        nf[:, CC_IDX] = clustering_coefficients(ei_np, sc)

        # In-degree and out-degree via np.bincount (no torch import needed)
        nf[:, IN_DEG_IDX] = np.bincount(edst, minlength=sc).astype(np.float32) if ec > 0 else 0.0
        nf[:, OUT_DEG_IDX] = np.bincount(esrc, minlength=sc).astype(np.float32) if ec > 0 else 0.0

        nf_parts.append(nf)
        ni_parts.append(nids)
        es_parts.append(esrc)
        ed_parts.append(edst)
        ef_parts.append(ef)
        specs_out.append((s_offset, sc, e_offset, ec, y_val, at_val))

        s_offset += sc
        e_offset += ec

    # Concatenate all parts into contiguous arrays
    if nf_parts:
        return (
            np.concatenate(nf_parts),
            np.concatenate(ni_parts),
            np.concatenate(es_parts) if es_parts and e_offset > 0 else np.empty(0, dtype=edge_src.dtype),
            np.concatenate(ed_parts) if ed_parts and e_offset > 0 else np.empty(0, dtype=edge_dst.dtype),
            np.concatenate(ef_parts) if ef_parts and e_offset > 0 else np.empty((0, edge_feats.shape[1]), dtype=edge_feats.dtype),
            specs_out,
        )
    # Empty chunk fallback
    return (
        np.empty((0, node_feats.shape[1]), dtype=node_feats.dtype),
        np.empty(0, dtype=node_ids.dtype),
        np.empty(0, dtype=edge_src.dtype),
        np.empty(0, dtype=edge_dst.dtype),
        np.empty((0, edge_feats.shape[1]), dtype=edge_feats.dtype),
        specs_out,
    )


def _numpy_to_data(
    node_feats: np.ndarray,
    node_ids: np.ndarray,
    edge_src: np.ndarray,
    edge_dst: np.ndarray,
    edge_feats: np.ndarray,
    specs: list[tuple[int, int, int, int, int, int]],
) -> list[Data]:
    """Convert numpy arrays returned by _assemble_chunk_numpy into PyG Data objects.

    Runs in the parent process where torch IPC is not involved.
    Each tuple in specs: (s_start, s_count, e_start, e_count, y_val, at_val)
    with offsets local to the passed arrays.
    """
    graphs: list[Data] = []
    for ss, sc, es, ec, y_val, at_val in specs:
        nf = torch.from_numpy(node_feats[ss:ss + sc].copy())
        nids = torch.from_numpy(node_ids[ss:ss + sc].copy())
        ei = torch.stack([
            torch.from_numpy(edge_src[es:es + ec].copy()),
            torch.from_numpy(edge_dst[es:es + ec].copy()),
        ])
        graphs.append(Data(
            x=nf,
            edge_index=ei,
            edge_attr=torch.from_numpy(edge_feats[es:es + ec].copy()),
            node_id=nids,
            y=torch.tensor([y_val], dtype=torch.long),
            attack_type=torch.tensor([at_val], dtype=torch.long),
        ))
    return graphs


def _assemble_graphs(
    all_node_feats: Tensor,
    all_node_ids: Tensor,
    all_edge_src: Tensor,
    all_edge_dst: Tensor,
    all_edge_feats: Tensor,
    s_wids: list[int],
    s_starts: list[int],
    s_counts: list[int],
    e_lookup: dict[int, tuple[int, int]],
    label_y: dict[int, int],
    label_at: dict[int, int],
) -> list[Data]:
    """Dispatch graph assembly — sequential or parallel via ProcessPoolExecutor."""
    import os

    log = structlog.get_logger()

    # Build flat window specs: (s_start, s_count, e_start, e_count, y_val, at_val)
    win_specs: list[tuple[int, int, int, int, int, int]] = []
    for i, wid in enumerate(s_wids):
        e_entry = e_lookup.get(wid)
        if e_entry is None:
            continue
        ss, sc = s_starts[i], s_counts[i]
        es, ec = e_entry
        win_specs.append((ss, sc, es, ec, label_y.get(wid, 0), label_at.get(wid, 0)))

    # Default to min(cpu_count, 8) workers: fork is cheap (~ms startup, no
    # tensor pickling) and preprocessing is CPU-only (no CUDA).
    default_workers = min(os.cpu_count() or 1, 8)
    n_workers = int(os.environ.get("KD_GAT_GRAPH_WORKERS", default_workers))
    chunk_size = max(500, len(win_specs) // (n_workers * 16)) if n_workers > 1 else len(win_specs)

    # Convert to numpy once — _assemble_chunk_numpy works with numpy arrays.
    nf_np = all_node_feats.numpy()
    ni_np = all_node_ids.numpy()
    es_np = all_edge_src.numpy()
    ed_np = all_edge_dst.numpy()
    ef_np = all_edge_feats.numpy()
    del all_node_feats, all_node_ids, all_edge_src, all_edge_dst, all_edge_feats

    if n_workers <= 1 or len(win_specs) <= chunk_size:
        log.info("graph_assembly_start", n_windows=len(win_specs), n_workers=1, mode="sequential")
        result = _assemble_chunk_numpy(nf_np, ni_np, es_np, ed_np, ef_np, win_specs)
        return _numpy_to_data(*result)

    # ── Parallel path (fork — safe because preprocessing is CPU-only) ──
    # Workers return numpy arrays (standard pickle/memcpy IPC), not torch
    # Data objects (which would trigger torch's mmap/file_system IPC).
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    log.info("graph_assembly_start", n_windows=len(win_specs),
             n_workers=n_workers, chunk_size=chunk_size, mode="parallel")

    ctx = mp.get_context("fork")
    futures = []
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
        for chunk_start in range(0, len(win_specs), chunk_size):
            chunk = win_specs[chunk_start:chunk_start + chunk_size]
            # Compute contiguous array slice bounds for this chunk
            s_lo = min(ss for ss, *_ in chunk)
            s_hi = max(ss + sc for ss, sc, *_ in chunk)
            e_lo = min(es for _, _, es, *_ in chunk)
            e_hi = max(es + ec for _, _, es, ec, *_ in chunk)
            # Rebase offsets to chunk-local
            local_specs = [
                (ss - s_lo, sc, es - e_lo, ec, y, at)
                for ss, sc, es, ec, y, at in chunk
            ]
            futures.append(pool.submit(
                _assemble_chunk_numpy,
                nf_np[s_lo:s_hi].copy(),
                ni_np[s_lo:s_hi].copy(),
                es_np[e_lo:e_hi].copy(),
                ed_np[e_lo:e_hi].copy(),
                ef_np[e_lo:e_hi].copy(),
                local_specs,
            ))

    # Collect numpy results in submission order, convert to Data in parent
    graphs: list[Data] = []
    for future in futures:
        nf_out, ni_out, es_out, ed_out, ef_out, specs_out = future.result()
        graphs.extend(_numpy_to_data(nf_out, ni_out, es_out, ed_out, ef_out, specs_out))
    return graphs


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

    # ── Build graphs (torch tensor slicing) ──────────────────────
    graphs = _assemble_graphs(
        all_node_feats, all_node_ids,
        all_edge_src, all_edge_dst, all_edge_feats,
        s_wids, s_starts, s_counts, e_lookup,
        label_y, label_at,
    )
    log.info("graphs_built", count=len(graphs))
    return graphs
