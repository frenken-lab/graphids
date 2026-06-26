"""Legacy raw CAN rows to windowed graph tables."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl
from structlog import get_logger

from graphids.core.data.preprocessing.graph_ops import apply_default_graph_transforms
from graphids.core.data.preprocessing.representations import (
    GraphRepresentationCfg,
    SnapshotRepresentationCfg,
    SnapshotSequenceRepresentationCfg,
)

log = get_logger(__name__)


@dataclass(frozen=True)
class GraphTables:
    node_stats: pl.DataFrame
    edge_df: pl.DataFrame
    labels: pl.DataFrame
    n_rows: int


def _empty(n_rows: int) -> GraphTables:
    return GraphTables(pl.DataFrame(), pl.DataFrame(), pl.DataFrame(), n_rows)


def _window_meta_exprs(window_size: int, stride: int) -> list[pl.Expr]:
    return [
        pl.col("_wid").cast(pl.Int64).alias("window_start_row"),
        (pl.col("_wid") + window_size).cast(pl.Int64).alias("window_end_row"),
        (pl.col("_wid") // stride).cast(pl.Int64).alias("window_ordinal"),
    ]


def _aggregate_nodes_and_labels(
    rows: pl.DataFrame,
    *,
    window_size: int,
    stride: int,
    node_stat_exprs: list[pl.Expr],
    label_exprs: list[pl.Expr],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    lf = rows.lazy().sort("_row")
    dyn = dict(every=f"{stride}i", period=f"{window_size}i", closed="left")
    node_lf = (
        lf.group_by_dynamic("_row", **dyn, group_by="node_id")
        .agg(*node_stat_exprs)
        .fill_null(0)
        .fill_nan(0)
        .rename({"_row": "_wid"})
        .with_columns(*_window_meta_exprs(window_size, stride))
    )
    label_lf = (
        lf.group_by_dynamic("_row", **dyn)
        .agg(*label_exprs)
        .rename({"_row": "_wid"})
        .with_columns(*_window_meta_exprs(window_size, stride))
    )
    return tuple(pl.collect_all([node_lf, label_lf]))  # type: ignore[return-value]


def _generate_edges(
    rows: pl.DataFrame,
    *,
    window_size: int,
    stride: int,
    edge_stat_exprs: list[pl.Expr],
    edge_base_cols: list[str],
) -> pl.DataFrame:
    lf = rows.lazy().sort("_row")
    dyn = dict(every=f"{stride}i", period=f"{window_size}i", closed="left")
    edge_names = [expr.meta.output_name() for expr in edge_stat_exprs]
    base_cols = ["_row", "node_id", "timestamp"]
    base_cols.extend(c for c in edge_base_cols if c not in base_cols)

    edge_df = (
        lf.select(*base_cols)
        .group_by_dynamic("_row", **dyn)
        .agg(
            pl.col("node_id").alias("src"),
            pl.col("node_id").shift(1).alias("dst"),
            *edge_stat_exprs,
        )
        .rename({"_row": "_wid"})
        .explode("src", "dst", *edge_names)
        .with_columns(*_window_meta_exprs(window_size, stride))
    ).collect()
    return edge_df.filter(
        pl.col("dst").is_not_null(),
        *(pl.col(c).is_not_null() for c in edge_names),
    )


def _complete_windows(tables: GraphTables, *, max_wid: int) -> GraphTables:
    return GraphTables(
        tables.node_stats.filter(pl.col("_wid") <= max_wid),
        tables.edge_df.filter(pl.col("_wid") <= max_wid),
        tables.labels.filter(pl.col("_wid") <= max_wid),
        tables.n_rows,
    )


def _keep_windows_with_edges(tables: GraphTables) -> GraphTables:
    edge_wids = tables.edge_df.select("_wid").unique()
    node_stats = tables.node_stats.join(edge_wids, on="_wid", how="semi")
    labels = tables.labels.join(node_stats.select("_wid").unique(), on="_wid", how="semi")
    return GraphTables(node_stats, tables.edge_df, labels, tables.n_rows)


def _localize_ids(tables: GraphTables) -> GraphTables:
    node_stats, edge_df = tables.node_stats, tables.edge_df
    wid_sizes = node_stats.group_by("_wid").agg(pl.len().alias("_n"))
    node_stats = node_stats.join(wid_sizes, on="_wid").sort(["_n", "_wid"])
    edge_df = edge_df.join(wid_sizes, on="_wid").sort(["_n", "_wid"])
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
    return GraphTables(node_stats, edge_df, tables.labels, tables.n_rows)


def _snapshot_tables(
    df: pl.DataFrame,
    *,
    window_size: int,
    stride: int,
    node_stat_exprs: list[pl.Expr],
    label_exprs: list[pl.Expr],
    edge_stat_exprs: list[pl.Expr],
    edge_base_cols: list[str],
) -> GraphTables:
    rows = (
        df.with_row_index("_row")
        .with_columns(pl.col("_row").cast(pl.Int64))
        .with_columns((pl.col("_row") % window_size < (window_size // 2)).alias("_first_half"))
    )
    n_rows = len(rows)
    n_windows = max(0, (n_rows - window_size) // stride + 1)
    if n_windows == 0:
        log.warning("no_complete_windows", n_rows=n_rows, window_size=window_size)
        return _empty(n_rows)

    node_stats, labels = _aggregate_nodes_and_labels(
        rows,
        window_size=window_size,
        stride=stride,
        node_stat_exprs=node_stat_exprs,
        label_exprs=label_exprs,
    )
    edge_df = _generate_edges(
        rows,
        window_size=window_size,
        stride=stride,
        edge_stat_exprs=edge_stat_exprs,
        edge_base_cols=edge_base_cols,
    )
    tables = GraphTables(node_stats, edge_df, labels, n_rows)
    tables = _complete_windows(tables, max_wid=(n_windows - 1) * stride)
    node_stats, edge_df = apply_default_graph_transforms(tables.node_stats, tables.edge_df)
    tables = GraphTables(node_stats, edge_df, tables.labels, tables.n_rows)
    tables = _keep_windows_with_edges(tables)
    if tables.node_stats.is_empty():
        log.warning("no_graphs_with_edges")
        return _empty(n_rows)
    return _localize_ids(tables)


def _sequence_tables(base: GraphTables, cfg: SnapshotSequenceRepresentationCfg) -> GraphTables:
    window_ids = base.node_stats.select("_wid").unique().sort("_wid")["_wid"].to_list()
    if len(window_ids) < cfg.sequence_length:
        return _empty(base.n_rows)

    sequence_rows = [
        {
            "sequence_id": sid,
            "sequence_step": step,
            "sequence_length": cfg.sequence_length,
            "sequence_stride": cfg.sequence_stride,
            "snapshot_wid": int(window_ids[start + step]),
        }
        for sid, start in enumerate(
            range(0, len(window_ids) - cfg.sequence_length + 1, cfg.sequence_stride)
        )
        for step in range(cfg.sequence_length)
    ]
    if not sequence_rows:
        return _empty(base.n_rows)

    sequence_map = pl.DataFrame(sequence_rows).with_columns(
        pl.col("sequence_id").cast(pl.Int64),
        pl.col("sequence_step").cast(pl.Int64),
        pl.col("sequence_length").cast(pl.Int64),
        pl.col("sequence_stride").cast(pl.Int64),
        pl.col("snapshot_wid").cast(pl.Int64),
    )
    node_counts = base.node_stats.group_by("_wid").agg(pl.len().alias("_node_count")).rename(
        {"_wid": "snapshot_wid"}
    )
    sequence_map = (
        sequence_map.join(node_counts, on="snapshot_wid", how="inner")
        .sort(["sequence_id", "sequence_step"])
        .with_columns(
            (pl.col("_node_count").cum_sum().over("sequence_id") - pl.col("_node_count"))
            .cast(pl.Int64)
            .alias("node_offset")
        )
    )

    node_stats = (
        sequence_map.join(base.node_stats.rename({"_wid": "snapshot_wid"}), on="snapshot_wid")
        .with_columns(
            pl.col("sequence_id").alias("_wid"),
            (pl.col("_local_id") + pl.col("node_offset")).cast(pl.Int64).alias("_local_id"),
        )
        .drop("node_offset", "_node_count")
        .sort(["sequence_id", "sequence_step", "_local_id"])
    )
    edge_df = (
        sequence_map.join(base.edge_df.rename({"_wid": "snapshot_wid"}), on="snapshot_wid")
        .with_columns(
            pl.col("sequence_id").alias("_wid"),
            (pl.col("src_local") + pl.col("node_offset")).cast(pl.Int64).alias("src_local"),
            (pl.col("dst_local") + pl.col("node_offset")).cast(pl.Int64).alias("dst_local"),
        )
        .drop("node_offset", "_node_count")
        .sort(["sequence_id", "sequence_step", "src_local", "dst_local"])
    )

    target = sequence_map.filter(pl.col("sequence_step") == cfg.sequence_length - 1).select(
        "sequence_id",
        "sequence_length",
        "sequence_stride",
        pl.col("snapshot_wid").alias("target_snapshot_wid"),
    )
    base_labels = base.labels.rename({"_wid": "target_snapshot_wid"})
    labels = target.join(base_labels, on="target_snapshot_wid", how="left")
    context_meta = (
        sequence_map.select("sequence_id", "snapshot_wid")
        .join(base.labels.rename({"_wid": "snapshot_wid"}), on="snapshot_wid", how="left")
        .group_by("sequence_id", maintain_order=True)
        .agg(
            pl.col("window_start_row").min().alias("window_start_row"),
            pl.col("window_end_row").max().alias("window_end_row"),
            pl.col("window_ordinal").min().alias("window_ordinal"),
        )
    )
    labels = (
        labels.join(context_meta, on="sequence_id", how="left", suffix="_context")
        .with_columns(pl.col("sequence_id").alias("_wid"))
        .drop("window_start_row", "window_end_row", "window_ordinal")
        .rename(
            {
                "window_start_row_context": "window_start_row",
                "window_end_row_context": "window_end_row",
                "window_ordinal_context": "window_ordinal",
            }
        )
    )
    return GraphTables(node_stats, edge_df, labels, base.n_rows)


def build_graph_tables(
    df: pl.DataFrame,
    *,
    node_stat_exprs: list[pl.Expr],
    label_exprs: list[pl.Expr],
    edge_stat_exprs: list[pl.Expr],
    edge_base_cols: list[str],
    representation_cfg: GraphRepresentationCfg,
) -> GraphTables:
    if isinstance(representation_cfg, SnapshotRepresentationCfg | SnapshotSequenceRepresentationCfg):
        base = _snapshot_tables(
            df,
            window_size=representation_cfg.window_size,
            stride=representation_cfg.stride,
            node_stat_exprs=node_stat_exprs,
            label_exprs=label_exprs,
            edge_stat_exprs=edge_stat_exprs,
            edge_base_cols=edge_base_cols,
        )
    else:
        raise TypeError(f"unsupported representation config: {type(representation_cfg)!r}")

    if isinstance(representation_cfg, SnapshotSequenceRepresentationCfg):
        return base if base.node_stats.is_empty() else _sequence_tables(base, representation_cfg)
    return base
