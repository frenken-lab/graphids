"""Composable graph transforms over node/edge preprocessing tables."""

from __future__ import annotations

import polars as pl


def _add_edge_frequency(node_stats: pl.DataFrame, edge_df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    return node_stats, edge_df.with_columns(
        pl.len().over(["_wid", "src", "dst"]).cast(pl.Float32).alias("edge_freq")
    )


def _add_bidir(node_stats: pl.DataFrame, edge_df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    pairs = edge_df.select("_wid", "src", "dst").unique()
    return node_stats, (
        edge_df.join(
            pairs.with_columns(pl.lit(True).alias("_rev")),
            left_on=["_wid", "dst", "src"],
            right_on=["_wid", "src", "dst"],
            how="left",
        )
        .with_columns(pl.col("_rev").fill_null(False).cast(pl.Float32).alias("bidir"))
        .drop("_rev")
    )


def _add_graph_topology(node_stats: pl.DataFrame, edge_df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
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
    node_stats = (
        node_stats.update(cc, on=["_wid", "node_id"])
        .update(in_deg, on=["_wid", "node_id"])
        .update(out_deg, on=["_wid", "node_id"])
    )
    return node_stats, edge_df


def apply_default_graph_transforms(
    node_stats: pl.DataFrame,
    edge_df: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Add the graph features consumed by the CAN schema."""
    node_stats, edge_df = _add_edge_frequency(node_stats, edge_df)
    node_stats, edge_df = _add_bidir(node_stats, edge_df)
    return _add_graph_topology(node_stats, edge_df)
