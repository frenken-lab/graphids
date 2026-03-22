"""Feature definitions for CAN bus graph windows.

Single source of truth for node and edge feature expressions.
Both the per-window path (node_features/edge_features functions)
and the vectorized batch path (can_bus.py _build_graphs) use these.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import torch
from torch import Tensor

BYTE_COLS = [f"byte_{i}" for i in range(8)]

# Column order defines tensor layout. Changing order changes model input.
NODE_COL_ORDER = (
    [f"{c}_mean" for c in BYTE_COLS]
    + [f"{c}_std" for c in BYTE_COLS]
    + [f"{c}_range" for c in BYTE_COLS]
    + ["msg_count", "entropy_mean", "skewness", "kurtosis",
       "clustering_coeff", "split_half_ratio", "change_rate"]
)

N_NODE_FEATURES = len(NODE_COL_ORDER)
N_EDGE_FEATURES = 12

# Polars aggregation expressions for per-node stats within a window.
# Used by group_by("node_id").agg() and group_by(["_wid", "node_id"]).agg().
# Requires columns: byte_0..7, entropy, _first_half (bool).
NODE_STAT_EXPRS: list[pl.Expr] = [
    *[pl.col(c).mean().alias(f"{c}_mean") for c in BYTE_COLS],
    *[pl.col(c).std().alias(f"{c}_std") for c in BYTE_COLS],
    *[(pl.col(c).max() - pl.col(c).min()).alias(f"{c}_range") for c in BYTE_COLS],
    pl.len().cast(pl.Float32).alias("msg_count"),
    pl.col("entropy").mean().alias("entropy_mean"),
    pl.col("byte_0").skew().fill_nan(0).clip(-10, 10).alias("skewness"),
    pl.col("byte_0").kurtosis().fill_nan(0).clip(-10, 10).alias("kurtosis"),
    pl.lit(0.0).alias("clustering_coeff"),  # filled per-window from graph structure
    pl.col("_first_half").mean().alias("split_half_ratio"),
    (pl.col("byte_0").diff().abs().drop_nulls() > 0).mean().alias("change_rate"),
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
    stats: pl.DataFrame, num_nodes: int, edge_index: np.ndarray | None = None,
) -> Tensor:
    """Convert a per-node stats DataFrame to a [num_nodes, 31] feature tensor.

    Stats must have 'node_id' column + all NODE_COL_ORDER columns.
    """
    out = torch.zeros(num_nodes, N_NODE_FEATURES, dtype=torch.float32)
    if stats.is_empty():
        return out

    ids = stats["node_id"].to_numpy().copy()
    feat_df = stats.select(NODE_COL_ORDER).cast({c: pl.Float32 for c in NODE_COL_ORDER})
    feat_tensor = feat_df.to_torch(dtype=pl.Float32)
    out[torch.from_numpy(ids)] = feat_tensor

    if edge_index is not None:
        cc_idx = NODE_COL_ORDER.index("clustering_coeff")
        out[:, cc_idx] = torch.from_numpy(clustering_coefficients(edge_index, num_nodes))

    return out


def node_features(
    window: pl.DataFrame,
    num_nodes: int,
    edge_index: np.ndarray | None = None,
) -> Tensor:
    """Compute 31-D node feature matrix from a single window DataFrame.

    This is the per-window path used by tests and standalone usage.
    The vectorized batch path in can_bus.py uses NODE_STAT_EXPRS directly.
    """
    half = len(window) // 2
    window = window.with_row_index("_row").with_columns(
        (pl.col("_row") < half).alias("_first_half")
    )
    stats = window.group_by("node_id").agg(*NODE_STAT_EXPRS).fill_null(0).fill_nan(0)
    return stats_to_tensor(stats, num_nodes, edge_index)


def edge_features(
    timestamps: np.ndarray,
    byte_arrays: list[np.ndarray],
    src: np.ndarray,
    dst: np.ndarray,
) -> Tensor:
    """Compute 12-D edge feature tensor from raw numpy arrays.

    iat | 0 | iat | iat | 0 | 0 | 1 | byte_diff(4) | bidirectional
    """
    n = len(src)
    out = torch.zeros(n, N_EDGE_FEATURES, dtype=torch.float32)
    if n == 0:
        return out

    iat = torch.from_numpy(np.diff(timestamps).astype(np.float32))
    out[:, 0] = iat
    out[:, 2] = iat
    out[:, 3] = iat
    out[:, 6] = 1.0

    for i in range(min(4, len(byte_arrays))):
        out[:, 7 + i] = torch.from_numpy(
            np.abs(np.diff(byte_arrays[i])).astype(np.float32)
        )

    directed = set(zip(src, dst))
    bidir = torch.tensor(
        [1.0 if (d, s) in directed else 0.0 for s, d in zip(src, dst)],
        dtype=torch.float32,
    )
    out[:, 11] = bidir

    return out
