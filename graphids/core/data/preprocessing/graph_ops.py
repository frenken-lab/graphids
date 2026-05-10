"""Composable graph transforms over node/edge preprocessing tables."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True)
class GraphTransform:
    """A declarative graph transform with explicit input/output columns."""

    name: str
    requires: tuple[str, ...]
    produces: tuple[str, ...]
    fn: Callable[[pl.DataFrame, pl.DataFrame], tuple[pl.DataFrame, pl.DataFrame]]

    def apply(self, node_stats: pl.DataFrame, edge_df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
        available = set(node_stats.columns) | set(edge_df.columns)
        missing = [c for c in self.requires if c not in available]
        if missing:
            raise ValueError(f"transform {self.name!r} missing required columns: {missing!r}")
        next_node, next_edge = self.fn(node_stats, edge_df)
        produced = set(next_node.columns) | set(next_edge.columns)
        missing_out = [c for c in self.produces if c not in produced]
        if missing_out:
            raise ValueError(f"transform {self.name!r} did not produce columns: {missing_out!r}")
        return next_node, next_edge


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


def _add_secondary_node_stats(node_stats: pl.DataFrame, edge_df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    total = pl.col("in_degree") + pl.col("out_degree")
    p_out = pl.when(total > 0).then(pl.col("out_degree") / total).otherwise(0.0)
    p_in = pl.when(total > 0).then(pl.col("in_degree") / total).otherwise(0.0)
    node_stats = node_stats.with_columns(
        pl.when(pl.col("in_degree") > 0)
        .then(pl.col("out_degree") / pl.col("in_degree"))
        .otherwise(pl.col("out_degree"))
        .cast(pl.Float32)
        .alias("in_out_ratio"),
        pl.when(total > 0)
        .then(-pl.when(p_out > 0).then(p_out * p_out.log()).otherwise(0.0) - pl.when(p_in > 0).then(p_in * p_in.log()).otherwise(0.0))
        .otherwise(0.0)
        .cast(pl.Float32)
        .alias("neighbor_entropy"),
    )
    return node_stats, edge_df


def default_graph_transforms() -> list[GraphTransform]:
    """Default graph transforms used in cache builds."""
    return [
        GraphTransform(
            name="edge_frequency",
            requires=("_wid", "src", "dst"),
            produces=("edge_freq",),
            fn=_add_edge_frequency,
        ),
        GraphTransform(
            name="bidir",
            requires=("_wid", "src", "dst"),
            produces=("bidir",),
            fn=_add_bidir,
        ),
        GraphTransform(
            name="topology",
            requires=("_wid", "node_id", "src", "dst"),
            produces=("clustering_coeff", "in_degree", "out_degree"),
            fn=_add_graph_topology,
        ),
    ]


def secondary_graph_transforms() -> list[GraphTransform]:
    """Additional exploratory graph transforms used in feature tests."""
    return [
        GraphTransform(
            name="secondary_node_stats",
            requires=("in_degree", "out_degree"),
            produces=("in_out_ratio", "neighbor_entropy"),
            fn=_add_secondary_node_stats,
        )
    ]
