"""Sliding-window CSV → PyG Data graphs (v2 composition).

Schema-driven (GraphSchema's Polars exprs + col orders) — same contract
as v1 with three structural changes:

1. ``polars.DataFrame.to_torch(dtype=...)`` everywhere instead of
   ``to_numpy().copy()`` + ``torch.from_numpy``. Polars handles the
   contiguous copy and dtype coercion in one call.
2. ``_slices_from_counts`` helper: per-group counts → cumulative slice
   tensor. Used by both x/node_id and edge_index/edge_attr.
3. The pipeline class is a thin schema-binding layer over module-level
   functions. ``_aggregate``, ``_add_bidir``, ``_graph_structure``,
   ``_build_pyg_data`` were instance methods only because they read
   ``self.<schema_attr>`` — making the schema bits explicit kwargs lets
   them stand alone and be tested in isolation.

Triangle-count clustering coefficient stays Polars (no PyG primitive;
per-graph NX loop over ~50K windows is 10-100× slower at preprocess time).
"""

from __future__ import annotations

import polars as pl
import torch
from structlog import get_logger
from torch_geometric.data import Data

log = get_logger(__name__)


def _slices_from_counts(counts: pl.Series) -> torch.Tensor:
    """Per-group counts → cumulative slice tensor ``[0, c0, c0+c1, ...]``."""
    return torch.cat(
        [
            torch.zeros(1, dtype=torch.long),
            torch.from_numpy(counts.to_numpy()).cumsum(0).to(torch.long),
        ]
    )


def _add_bidir(edge_df: pl.DataFrame) -> pl.DataFrame:
    """``bidir=1.0`` if the reverse edge exists in the same window."""
    pairs = edge_df.select("_wid", "src", "dst").unique()
    return (
        edge_df.join(
            pairs.with_columns(pl.lit(True).alias("_rev")),
            left_on=["_wid", "dst", "src"],
            right_on=["_wid", "src", "dst"],
            how="left",
        )
        .with_columns(pl.col("_rev").fill_null(False).cast(pl.Float32).alias("bidir"))
        .drop("_rev")
    )


def _graph_structure(node_stats: pl.DataFrame, edge_df: pl.DataFrame) -> pl.DataFrame:
    """Clustering coefficient (triangle counting) + in/out degree."""
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

    pairs = pl.concat(
        [
            edge_df.select("_wid", pl.col("src").alias("u"), pl.col("dst").alias("v")),
            edge_df.select("_wid", pl.col("dst").alias("u"), pl.col("src").alias("v")),
        ]
    ).unique(["_wid", "u", "v"])

    triangles = (
        pairs.join(
            pairs.select("_wid", pl.col("u").alias("_mid"), pl.col("v").alias("w")),
            left_on=["_wid", "v"],
            right_on=["_wid", "_mid"],
            how="inner",
        )
        .filter(pl.col("u") != pl.col("w"))
        .join(pairs, left_on=["_wid", "u", "w"], right_on=["_wid", "u", "v"], how="semi")
        .group_by(["_wid", "u"])
        .agg((pl.len() / 2).cast(pl.Float32).alias("_tri"))
        .rename({"u": "node_id"})
    )
    udeg = (
        pairs.group_by(["_wid", "u"])
        .agg(pl.len().cast(pl.Float32).alias("_undeg"))
        .rename({"u": "node_id"})
    )
    cc = (
        triangles.join(udeg, on=["_wid", "node_id"], how="right")
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
    return (
        node_stats.update(cc, on=["_wid", "node_id"])
        .update(in_deg, on=["_wid", "node_id"])
        .update(out_deg, on=["_wid", "node_id"])
    )


class GraphPipeline:
    """Sliding-window → PyG Data graphs. Schema-driven."""

    def __init__(
        self,
        *,
        node_stat_exprs: list[pl.Expr],
        edge_stat_exprs: list[pl.Expr],
        node_col_order: list[str],
        edge_col_order: tuple[str, ...],
        label_exprs: list[pl.Expr],
        edge_base_cols: list[str],
    ):
        self.node_stat_exprs = node_stat_exprs
        self.edge_stat_exprs = edge_stat_exprs
        self.node_col_order = node_col_order
        self.edge_col_order = edge_col_order
        self.label_exprs = label_exprs
        self.edge_base_cols = edge_base_cols
        self.label_names = [e.meta.output_name() for e in label_exprs]
        assert self.label_names[0] == "y", f"first label expr must alias 'y', got {self.label_names[0]!r}"

    def run(
        self,
        df: pl.DataFrame,
        window_size: int,
        stride: int,
    ) -> tuple[Data, dict, int, int]:
        df = df.with_row_index("_row").with_columns(pl.col("_row").cast(pl.Int64))
        n_rows = len(df)
        n_windows = max(0, (n_rows - window_size) // stride + 1)
        if n_windows == 0:
            log.warning("no_complete_windows", n_rows=n_rows, window_size=window_size)
            return Data(), {}, 0, n_rows

        log.info("windowing", n_windows=n_windows, window=window_size, stride=stride)
        half = window_size // 2
        max_wid = (n_windows - 1) * stride
        df = df.with_columns((pl.col("_row") % window_size < half).alias("_first_half"))
        lf = df.lazy().sort("_row")

        # Fused single-scan aggregation
        dyn = dict(every=f"{stride}i", period=f"{window_size}i", closed="left")
        node_lf = (
            lf.group_by_dynamic("_row", **dyn, group_by="node_id")
            .agg(*self.node_stat_exprs)
            .fill_null(0)
            .fill_nan(0)
            .rename({"_row": "_wid"})
        )
        labels_lf = (
            lf.group_by_dynamic("_row", **dyn).agg(*self.label_exprs).rename({"_row": "_wid"})
        )
        edge_agg = [
            pl.col("node_id").alias("src"),
            pl.col("node_id").shift(-1).alias("dst"),
            *self.edge_stat_exprs,
        ]
        edge_cols = ["src", "dst"] + [e.meta.output_name() for e in self.edge_stat_exprs]
        edge_lf = (
            lf.select("_row", "node_id", "timestamp", *self.edge_base_cols)
            .group_by_dynamic("_row", **dyn)
            .agg(*edge_agg)
            .rename({"_row": "_wid"})
            .explode(edge_cols)
            .filter(pl.col("dst").is_not_null() & pl.col("iat").is_not_null())
            .with_columns(
                pl.len().over(["_wid", "src", "dst"]).cast(pl.Float32).alias("edge_freq")
            )
        )
        node_stats, labels, edge_df = pl.collect_all([node_lf, labels_lf, edge_lf])
        log.info("features_computed", stats=len(node_stats), edges=len(edge_df))
        del df, lf

        node_stats = node_stats.filter(pl.col("_wid") <= max_wid)
        edge_df = edge_df.filter(pl.col("_wid") <= max_wid)
        labels = labels.filter(pl.col("_wid") <= max_wid)

        edge_df = _add_bidir(edge_df)
        node_stats = _graph_structure(node_stats, edge_df)

        # Keep only windows with edges
        node_stats = node_stats.filter(pl.col("_wid").is_in(edge_df["_wid"].unique()))
        if len(node_stats) == 0:
            log.warning("no_graphs_with_edges")
            return Data(), {}, 0, n_rows

        # Pre-sort by node count → sequential page faults for sampler
        wid_sizes = node_stats.group_by("_wid").agg(pl.len().alias("_n"))
        node_stats = node_stats.join(wid_sizes, on="_wid").sort(["_n", "_wid"])
        edge_df = edge_df.join(wid_sizes, on="_wid").sort(["_n", "_wid"])

        # Local IDs per window
        node_stats = node_stats.with_columns(
            (pl.cum_count("node_id").over("_wid") - 1).cast(pl.Int64).alias("_local_id")
        )
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

        # Polars → torch via to_torch (no manual to_numpy().copy())
        x = (
            node_stats.select(self.node_col_order)
            .fill_null(0)
            .fill_nan(0)
            .to_torch(dtype=pl.Float32)
        )
        node_id = node_stats.select("node_id").to_torch(dtype=pl.Int64).squeeze(-1)
        edge_index = (
            edge_df.select("src_local", "dst_local").to_torch(dtype=pl.Int64).t().contiguous()
        )
        edge_attr = (
            edge_df.select(list(self.edge_col_order))
            .fill_null(0)
            .fill_nan(0)
            .to_torch(dtype=pl.Float32)
        )

        kept_wids = node_stats.group_by("_wid", maintain_order=True).first().select("_wid")
        num_graphs = len(kept_wids)
        node_counts = node_stats.group_by("_wid", maintain_order=True).len()["len"]
        edge_counts = edge_df.group_by("_wid", maintain_order=True).len()["len"]
        node_slice = _slices_from_counts(node_counts)
        edge_slice = _slices_from_counts(edge_counts)
        graph_idx = torch.arange(num_graphs + 1, dtype=torch.long)

        labels_aligned = kept_wids.join(labels, on="_wid", how="left").fill_null(0)
        label_tensors = {
            n: labels_aligned.select(n).to_torch(dtype=pl.Int64).squeeze(-1)
            for n in self.label_names
        }

        data = Data(
            x=x, edge_index=edge_index, edge_attr=edge_attr, node_id=node_id, **label_tensors
        )
        slices = {
            "x": node_slice,
            "edge_index": edge_slice,
            "edge_attr": edge_slice,
            "node_id": node_slice,
            **{n: graph_idx for n in self.label_names},
        }
        log.info("graphs_built", count=num_graphs)
        return data, slices, num_graphs, n_rows
