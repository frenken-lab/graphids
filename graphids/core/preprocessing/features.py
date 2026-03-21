"""Polars-based feature computation for CAN bus graph windows.

31-D node features, 12-D edge features.
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


def _clustering_coefficients(edge_index: np.ndarray, num_nodes: int) -> np.ndarray:
    """Clustering coefficient per node via networkx (C-optimized)."""
    import networkx as nx

    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(zip(edge_index[0], edge_index[1]))
    cc = nx.clustering(G)
    return np.array([cc.get(i, 0.0) for i in range(num_nodes)], dtype=np.float32)


def node_features(
    window: pl.DataFrame,
    num_nodes: int,
    edge_index: np.ndarray | None = None,
) -> Tensor:
    """Compute 31-D node feature matrix from a single window.

    byte_mean(8) | byte_std(8) | byte_range(8)
    | msg_count | entropy | skewness | kurtosis
    | clustering_coeff | split_half_ratio | change_rate
    """
    half = len(window) // 2

    window = window.with_row_index("_row").with_columns(
        (pl.col("_row") < half).alias("_first_half")
    )

    stats = window.group_by("node_id").agg(
        *[pl.col(c).mean().alias(f"{c}_mean") for c in BYTE_COLS],
        *[pl.col(c).std().alias(f"{c}_std") for c in BYTE_COLS],
        *[(pl.col(c).max() - pl.col(c).min()).alias(f"{c}_range") for c in BYTE_COLS],
        pl.len().cast(pl.Float32).alias("msg_count"),
        pl.col("entropy").mean().alias("entropy_mean"),
        pl.col("byte_0").skew().alias("skewness"),
        pl.col("byte_0").kurtosis().alias("kurtosis"),
        pl.lit(0.0).alias("clustering_coeff"),  # placeholder, filled below
        pl.col("_first_half").mean().alias("split_half_ratio"),
        (pl.col("byte_0").diff().abs().drop_nulls() > 0).mean().alias("change_rate"),
    ).fill_null(0)

    out = torch.zeros(num_nodes, 31, dtype=torch.float32)
    if stats.is_empty():
        return out

    ids = stats["node_id"].to_numpy()

    # Clamp skewness/kurtosis before export
    stats = stats.with_columns(
        pl.col("skewness").clip(-10, 10),
        pl.col("kurtosis").clip(-10, 10),
    )

    # Select feature columns in order → to_torch() → dense tensor scatter
    feature_df = stats.select(NODE_COL_ORDER).cast({col: pl.Float32 for col in NODE_COL_ORDER})
    tensor_dict = feature_df.to_torch(dtype=pl.Float32)  # dict of 1-D tensors
    for col_idx, name in enumerate(NODE_COL_ORDER):
        out[ids, col_idx] = tensor_dict[name]

    # Clustering coefficient needs graph structure — overwrite placeholder
    if edge_index is not None:
        out[:, 28] = torch.from_numpy(_clustering_coefficients(edge_index, num_nodes))

    return out


def edge_features(
    timestamps: np.ndarray,
    byte_arrays: list[np.ndarray],
    src: np.ndarray,
    dst: np.ndarray,
) -> Tensor:
    """Compute 12-D edge feature tensor from raw numpy arrays.

    iat | 0 | iat | iat | 0 | 0 | 1 | byte_diff(4) | bidirectional
    Each edge in shift-1 adjacency is unique — no groupby.
    """
    n = len(src)
    out = torch.zeros(n, 12, dtype=torch.float32)
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
