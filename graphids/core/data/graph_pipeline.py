"""Dataset-agnostic sliding-window → graph pipeline.

Converts a timestamped message DataFrame into a collection of PyG ``Data``
graphs, one per sliding window. Domain-specific adapters (e.g.
``datasets/can_bus.py``) supply their own Polars expressions and column
layouts; this module handles windowing, graph construction, and tensor
packing with no knowledge of the underlying protocol.

Pipeline steps (inside ``sliding_window_graphs``):

1. **Window assignment** — map each row to one or more overlapping windows.
2. **Feature aggregation** — per-node stats, per-edge stats, per-window
   labels via Polars lazy frames (single scan of the data).
3. **Sequential collection** — collect lazy frames one at a time to bound
   peak memory (~20-30 GB savings vs ``collect_all``).
4. **Bidirectional edge flag** — mark edges whose reverse also exists.
5. **Graph structure** — clustering coefficient (triangle counting) and
   in/out degree, computed entirely in Polars.
6. **Label mapping** — per-window ``y`` (binary) and auxiliary labels.
7. **Slice boundaries** — RLE on window ID for zero-copy offset computation.
8. **Local ID assignment** — 0-based node indices per window, joined to edges.
9. **Polars → torch** — bulk tensor conversion (no per-graph Python loop).
10. **Pre-collation** — build ``(Data, slices)`` directly from bulk tensors
    (avoids the 3× memory cost of ``list[Data]`` → ``collate`` → save).
"""

from __future__ import annotations

import numpy as np
import polars as pl
import torch
from torch_geometric.data import Data

from graphids.log import get_logger


def _rle_boundaries(frame: pl.DataFrame) -> tuple[list[int], list[int], list[int]]:
    """Run-length encode the ``_wid`` column into (wids, starts, counts).

    Used to derive per-window slice boundaries without groupby overhead.
    The ``_wid`` column must be sorted (guaranteed by pipeline ordering).
    """
    rle = frame["_wid"].rle().struct.unnest()
    wids = rle["value"].to_list()
    counts = rle["len"].to_list()
    starts = (rle["len"].cum_sum() - rle["len"]).to_list()
    return wids, starts, counts


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


def sliding_window_graphs(
    df: pl.DataFrame,
    window_size: int,
    stride: int,
    *,
    node_stat_exprs: list[pl.Expr],
    edge_stat_exprs: list[pl.Expr],
    node_col_order: list[str],
    edge_col_order: tuple[str, ...],
    label_exprs: list[pl.Expr],
    edge_base_cols: list[str],
) -> tuple[Data, dict, int]:
    """Convert a message DataFrame into pre-collated (Data, slices, num_graphs).

    Returns the InMemoryDataset collated format directly from bulk tensors,
    avoiding the triple-copy of list[Data] → collate → save. Peak memory
    is ~1x the final tensor size instead of ~3x.

    Each graph is one sliding window. Nodes are active per-window entities
    (identified by the ``node_id`` column), edges are temporal adjacency
    (shift-1), with local IDs (0..n_active-1). ``node_id`` stores global
    indices for embedding lookup.

    Required columns in ``df``:
        ``node_id`` (Int64), ``timestamp``, plus whatever ``node_stat_exprs``,
        ``edge_stat_exprs``, ``label_exprs``, and ``edge_base_cols`` reference.

    Parameters describing the dataset schema:
        ``node_stat_exprs``: Polars aggregations for per-node features. Must
            include a ``_first_half`` reference if the dataset cares about it.
        ``edge_stat_exprs``: Polars expressions for edge features (iat, byte
            diffs, etc.) applied after ``sort(["_wid", "_row"])``.
        ``node_col_order``: final column order for the node feature tensor.
        ``edge_col_order``: final column order for the edge feature tensor.
        ``label_exprs``: per-window aggregations yielding label columns. The
            first expression must be aliased ``y``; additional expressions
            are attached as auxiliary attributes on the ``Data`` object.
        ``edge_base_cols``: extra columns required for edge feature
            computation (e.g. byte_0..7 for CAN byte diffs).
    """
    log = get_logger(__name__)

    # ── Step 1: Window assignment ──────────────────────────────────
    # Map each row to one or more sliding windows. When stride < window_size
    # windows overlap: a single row appears in multiple windows (explode).
    df = df.with_row_index("_row")
    n_rows = len(df)
    ws, st = window_size, stride
    half = ws // 2

    n_windows = max(0, (n_rows - ws) // st + 1)
    if n_windows == 0:
        log.warning("no_complete_windows", n_rows=n_rows, window_size=ws)
        return Data(), {}, 0
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

    # ── Step 2: Feature aggregation ────────────────────────────────
    # Build three lazy frames from one scan: node stats (group_by node),
    # edge features (shift-1 temporal adjacency), and per-window labels.
    lf = df.lazy()

    stats_lf = (
        lf.group_by(["_wid", "node_id"], maintain_order=True)
        .agg(*node_stat_exprs)
        .fill_null(0)
        .fill_nan(0)
    )

    edges_base = (
        lf.select("_wid", "_row", "node_id", "timestamp", *edge_base_cols)
        .sort(["_wid", "_row"])
        .with_columns(
            pl.col("node_id").alias("src"),
            pl.col("node_id").shift(-1).over("_wid").alias("dst"),
            *edge_stat_exprs,
        )
        .filter(pl.col("iat").is_not_null() & pl.col("dst").is_not_null())
        .with_columns(
            pl.len().over(["_wid", "src", "dst"]).cast(pl.Float32).alias("edge_freq"),
        )
    )

    labels_lf = lf.group_by("_wid").agg(*label_exprs)

    # ── Step 3: Sequential collection ──────────────────────────────
    # Collect lazy frames one at a time to bound peak memory (~20-30 GB
    # savings vs collect_all). Order matters: labels is tiny so collect
    # first; df/lf must survive until the last collect.
    labels = labels_lf.collect()
    del labels_lf
    node_stats = stats_lf.collect()
    del stats_lf
    edge_df = edges_base.collect()
    del edges_base, df, lf
    log.info("features_computed", stat_rows=len(node_stats), edge_rows=len(edge_df))

    # ── Step 4: Bidirectional edge flag ────────────────────────────
    # For each directed edge, mark whether the reverse also exists.
    edge_df = add_bidir_flag(edge_df)

    # ── Step 5: Graph structure — clustering coefficient + degree ─
    # Computed entirely in Polars (no NetworkX). Triangle counting on
    # undirected edges gives clustering coefficient; directed edge
    # group_by gives in/out degree. Results overwrite placeholder
    # zeros emitted by node_stat_exprs.
    # In/out degree from directed edges
    in_deg = (
        edge_df.group_by(["_wid", "dst"])
        .agg(pl.len().cast(pl.Float32).alias("in_degree"))
        .rename({"dst": "node_id"})
    )
    out_deg = (
        edge_df.group_by(["_wid", "src"])
        .agg(pl.len().cast(pl.Float32).alias("out_degree"))
        .rename({"src": "node_id"})
    )

    # Clustering coefficient via triangle counting on undirected edges
    edge_pairs = pl.concat(
        [
            edge_df.select("_wid", pl.col("src").alias("u"), pl.col("dst").alias("v")),
            edge_df.select("_wid", pl.col("dst").alias("u"), pl.col("src").alias("v")),
        ]
    ).unique(["_wid", "u", "v"])

    # 2-paths: u─v─w
    two_paths = edge_pairs.join(
        edge_pairs.select("_wid", pl.col("u").alias("_mid"), pl.col("v").alias("w")),
        left_on=["_wid", "v"],
        right_on=["_wid", "_mid"],
        how="inner",
    ).filter(pl.col("u") != pl.col("w"))

    # Close triangles: keep 2-paths where u─w edge exists
    tri = two_paths.join(
        edge_pairs,
        left_on=["_wid", "u", "w"],
        right_on=["_wid", "u", "v"],
        how="semi",
    )
    del two_paths

    tri_per_node = (
        tri.group_by(["_wid", "u"])
        .agg((pl.len() / 2).cast(pl.Float32).alias("_tri"))
        .rename({"u": "node_id"})
    )
    del tri

    undirected_deg = (
        edge_pairs.group_by(["_wid", "u"])
        .agg(pl.len().cast(pl.Float32).alias("_undeg"))
        .rename({"u": "node_id"})
    )
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
        node_stats.join(cc, on=["_wid", "node_id"], how="left", suffix="_computed")
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

    # ── Step 6: Label mapping ──────────────────────────────────────
    # First label expr must be aliased "y"; additional exprs become
    # extra tensor attributes on the Data object.
    label_names = [e.meta.output_name() for e in label_exprs]
    assert label_names[0] == "y", f"first label expr must be aliased 'y', got {label_names[0]!r}"
    label_maps: dict[str, dict[int, int]] = {}
    for name in label_names:
        label_maps[name] = dict(zip(labels["_wid"].to_list(), labels[name].to_list()))
    del labels

    # ── Step 7: Slice boundaries via RLE ────────────────────────────
    # Run-length encode _wid to get per-window (start, count) offsets
    # into the bulk node_stats and edge_df DataFrames.
    s_wids, s_starts, s_counts = _rle_boundaries(node_stats)

    # ── Step 8: Local ID assignment ──────────────────────────────
    # Assign 0-based local node IDs per window (0..n_active-1),
    # then join to edge_df so src/dst get local indices in bulk.
    local_ids = np.concatenate([np.arange(c, dtype=np.int64) for c in s_counts])
    node_stats = node_stats.with_columns(pl.Series("_local_id", local_ids))

    id_map = node_stats.select("_wid", "node_id", "_local_id")
    edge_df = edge_df.join(
        id_map.rename({"node_id": "src", "_local_id": "src_local"}),
        on=["_wid", "src"],
        how="left",
    ).join(
        id_map.rename({"node_id": "dst", "_local_id": "dst_local"}),
        on=["_wid", "dst"],
        how="left",
    )
    del id_map

    # Recompute edge boundaries after join (left join preserves row order)
    e_wids, e_starts, e_counts = _rle_boundaries(edge_df)
    e_lookup: dict[int, tuple[int, int]] = dict(zip(e_wids, zip(e_starts, e_counts)))

    # ── Step 9: Polars → torch ─────────────────────────────────────
    # Bulk tensor conversion — no per-graph Python loop.
    all_node_feats = (
        node_stats.select(node_col_order).fill_null(0).fill_nan(0).to_torch(dtype=pl.Float32)
    )
    all_node_ids = torch.from_numpy(node_stats["node_id"].cast(pl.Int64).to_numpy().copy())
    all_edge_src = torch.from_numpy(edge_df["src_local"].cast(pl.Int64).to_numpy().copy())
    all_edge_dst = torch.from_numpy(edge_df["dst_local"].cast(pl.Int64).to_numpy().copy())
    all_edge_feats = (
        edge_df.select(list(edge_col_order)).fill_null(0).fill_nan(0).to_torch(dtype=pl.Float32)
    )
    del node_stats, edge_df

    # ── Step 10: Pre-collation ─────────────────────────────────────
    # Build (Data, slices) directly from the bulk tensors — the same
    # format InMemoryDataset.collate() produces, but without the 3×
    # memory cost of list[Data] → collate → save. Graphs are presorted
    # by node count so DynamicBatchSampler page-faults sequentially.
    keep = [i for i, wid in enumerate(s_wids) if wid in e_lookup]
    keep.sort(key=lambda i: s_counts[i])  # presort by node count
    kept_wids = [s_wids[i] for i in keep]
    num_graphs = len(kept_wids)
    if num_graphs == 0:
        log.warning("no_graphs_with_edges")
        return Data(), {}, 0

    # Node slices: cumulative node counts for kept windows
    node_counts = [s_counts[i] for i in keep]
    node_offsets = [s_starts[i] for i in keep]
    node_cumsum = torch.zeros(num_graphs + 1, dtype=torch.long)
    for j, c in enumerate(node_counts):
        node_cumsum[j + 1] = node_cumsum[j] + c

    # Edge slices: cumulative edge counts for kept windows
    edge_counts = [e_lookup[s_wids[i]][1] for i in keep]
    edge_offsets = [e_lookup[s_wids[i]][0] for i in keep]
    edge_cumsum = torch.zeros(num_graphs + 1, dtype=torch.long)
    for j, c in enumerate(edge_counts):
        edge_cumsum[j + 1] = edge_cumsum[j] + c

    # Gather node tensors (contiguous, in kept-window order)
    node_indices = torch.cat(
        [torch.arange(node_offsets[j], node_offsets[j] + node_counts[j]) for j in range(num_graphs)]
    )
    cat_x = all_node_feats[node_indices]
    cat_node_id = all_node_ids[node_indices]
    del all_node_feats, all_node_ids

    # Gather edge tensors
    edge_indices = torch.cat(
        [torch.arange(edge_offsets[j], edge_offsets[j] + edge_counts[j]) for j in range(num_graphs)]
    )
    cat_edge_src = all_edge_src[edge_indices]
    cat_edge_dst = all_edge_dst[edge_indices]
    cat_edge_attr = all_edge_feats[edge_indices]
    del all_edge_src, all_edge_dst, all_edge_feats

    cat_edge_index = torch.stack([cat_edge_src, cat_edge_dst])
    del cat_edge_src, cat_edge_dst

    # Per-graph labels (one tensor per label expression)
    graph_idx = torch.arange(num_graphs + 1, dtype=torch.long)
    label_tensors: dict[str, torch.Tensor] = {
        name: torch.tensor([label_maps[name].get(w, 0) for w in kept_wids], dtype=torch.long)
        for name in label_names
    }

    data = Data(
        x=cat_x,
        edge_index=cat_edge_index,
        edge_attr=cat_edge_attr,
        node_id=cat_node_id,
        **label_tensors,
    )
    slices = {
        "x": node_cumsum,
        "edge_index": edge_cumsum,
        "edge_attr": edge_cumsum,
        "node_id": node_cumsum,
        **{name: graph_idx for name in label_names},
    }
    log.info("graphs_built", count=num_graphs)
    return data, slices, num_graphs
