"""Composable graph transforms over node/edge preprocessing tables."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import polars as pl


@dataclass(frozen=True)
class GraphTransform:
    """A declarative graph transform with explicit input/output columns."""

    name: str
    applies_to: Literal["node", "edge", "graph"]
    requires: tuple[str, ...]
    produces: tuple[str, ...]
    fn: Callable[[pl.DataFrame, pl.DataFrame], tuple[pl.DataFrame, pl.DataFrame]]

    def apply(self, node_stats: pl.DataFrame, edge_df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
        missing = [c for c in self.requires if c not in set(node_stats.columns) | set(edge_df.columns)]
        if missing:
            raise ValueError(f"transform {self.name!r} missing required columns: {missing!r}")
        next_node, next_edge = self.fn(node_stats, edge_df)
        missing_out = [c for c in self.produces if c not in set(next_node.columns) | set(next_edge.columns)]
        if missing_out:
            raise ValueError(f"transform {self.name!r} did not produce columns: {missing_out!r}")
        return next_node, next_edge


def _add_edge_frequency(node_stats: pl.DataFrame, edge_df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    return node_stats, edge_df.with_columns(
        pl.len().over(["_wid", "src", "dst"]).cast(pl.Float32).alias("edge_freq")
    )


def _add_bidir(edge_df: pl.DataFrame) -> pl.DataFrame:
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


def _add_bidir_transform(node_stats: pl.DataFrame, edge_df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    return node_stats, _add_bidir(edge_df)


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


def _add_in_out_ratio(node_stats: pl.DataFrame, edge_df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    node_stats = node_stats.with_columns(
        pl.when(pl.col("out_degree") > 0)
        .then(pl.col("in_degree") / pl.col("out_degree"))
        .otherwise(0.0)
        .cast(pl.Float32)
        .alias("in_out_ratio")
    )
    return node_stats, edge_df


def _add_neighbor_entropy(node_stats: pl.DataFrame, edge_df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    counts = edge_df.group_by(["_wid", "src", "dst"]).agg(pl.len().cast(pl.Float32).alias("_cnt"))
    totals = counts.group_by(["_wid", "src"]).agg(pl.sum("_cnt").alias("_tot"))
    entropy = (
        counts.join(totals, on=["_wid", "src"], how="left")
        .with_columns((pl.col("_cnt") / pl.col("_tot")).alias("_p"))
        .with_columns(
            pl.when(pl.col("_p") > 0)
            .then(-(pl.col("_p") * pl.col("_p").log()))
            .otherwise(0.0)
            .cast(pl.Float32)
            .alias("_term")
        )
        .group_by(["_wid", "src"])
        .agg(pl.sum("_term").cast(pl.Float32).alias("neighbor_entropy"))
        .rename({"src": "node_id"})
    )
    node_stats = node_stats.join(entropy, on=["_wid", "node_id"], how="left").with_columns(
        pl.col("neighbor_entropy").fill_null(0.0).cast(pl.Float32)
    )
    return node_stats, edge_df


def default_graph_transforms() -> list[GraphTransform]:
    """Default graph transforms used in cache builds."""
    return [
        GraphTransform(
            name="edge_frequency",
            applies_to="edge",
            requires=("_wid", "src", "dst"),
            produces=("edge_freq",),
            fn=_add_edge_frequency,
        ),
        GraphTransform(
            name="bidir",
            applies_to="edge",
            requires=("_wid", "src", "dst"),
            produces=("bidir",),
            fn=_add_bidir_transform,
        ),
        GraphTransform(
            name="topology",
            applies_to="graph",
            requires=("_wid", "node_id", "src", "dst"),
            produces=("clustering_coeff", "in_degree", "out_degree"),
            fn=_add_graph_topology,
        ),
    ]


def secondary_graph_transforms() -> list[GraphTransform]:
    """Optional graph transforms for additional node-level statistics."""
    return [
        GraphTransform(
            name="in_out_ratio",
            applies_to="node",
            requires=("in_degree", "out_degree"),
            produces=("in_out_ratio",),
            fn=_add_in_out_ratio,
        ),
        GraphTransform(
            name="neighbor_entropy",
            applies_to="node",
            requires=("_wid", "node_id", "src", "dst"),
            produces=("neighbor_entropy",),
            fn=_add_neighbor_entropy,
        ),
    ]
