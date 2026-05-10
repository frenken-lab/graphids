"""Graph table materialization from windowed preprocessing primitives."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from structlog import get_logger

from graphids.core.data.preprocessing.edge_policy import EdgePolicy, temporal_edge_policy
from graphids.core.data.preprocessing.graph_ops import GraphTransform, default_graph_transforms
from graphids.core.data.preprocessing.segments import (
    EntitySegmentCfg,
    MultiScaleSegmentCfg,
    SequenceSegmentCfg,
    WindowSegmentCfg,
    WindowSegmenter,
    WindowedRows,
)

log = get_logger(__name__)


@dataclass(frozen=True)
class AggregatedTables:
    node_stats: pl.DataFrame
    edge_df: pl.DataFrame
    labels: pl.DataFrame


@dataclass(frozen=True)
class GraphTables:
    node_stats: pl.DataFrame
    edge_df: pl.DataFrame
    labels: pl.DataFrame
    n_rows: int


def _stage_summary(table: pl.DataFrame) -> dict[str, object]:
    summary: dict[str, object] = {
        "rows": table.height,
        "columns": table.width,
        "column_names": table.columns,
    }
    if "src" in table.columns:
        summary["src_unique"] = int(table.select(pl.col("src").n_unique()).item())
    if "dst" in table.columns:
        summary["dst_unique"] = int(table.select(pl.col("dst").n_unique()).item())
    if "_wid" in table.columns:
        summary["windows"] = int(table.select(pl.col("_wid").n_unique()).item())
    return summary


def _dump_stage(debug_dir: Path | None, stage: str, table: pl.DataFrame) -> None:
    if debug_dir is None:
        return
    debug_dir.mkdir(parents=True, exist_ok=True)
    table.write_parquet(debug_dir / f"{stage}.parquet")
    (debug_dir / f"{stage}.summary.json").write_text(
        json.dumps(_stage_summary(table), indent=2),
        encoding="utf-8",
    )


def _aggregate_nodes_labels(
    windowed: WindowedRows,
    *,
    node_stat_exprs: list[pl.Expr],
    label_exprs: list[pl.Expr],
    window_size: int,
    stride: int,
    debug_dir: Path | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    lf = windowed.rows.lazy().sort("_row")
    dyn = dict(every=f"{stride}i", period=f"{window_size}i", closed="left")
    node_lf = (
        lf.group_by_dynamic("_row", **dyn, group_by="node_id")
        .agg(*node_stat_exprs)
        .fill_null(0)
        .fill_nan(0)
        .rename({"_row": "_wid"})
    )
    labels_lf = lf.group_by_dynamic("_row", **dyn).agg(*label_exprs).rename({"_row": "_wid"})
    node_stats, labels = pl.collect_all([node_lf, labels_lf])
    _dump_stage(debug_dir, "02_node_stats_raw", node_stats)
    _dump_stage(debug_dir, "03_labels_raw", labels)
    return node_stats, labels


def _generate_edges(
    windowed: WindowedRows,
    *,
    edge_policy: EdgePolicy,
    edge_stat_exprs: list[pl.Expr],
    edge_base_cols: list[str],
    edge_feature_names: list[str],
    window_size: int,
    stride: int,
    debug_dir: Path | None = None,
) -> pl.DataFrame:
    lf = windowed.rows.lazy().sort("_row")
    dyn = dict(every=f"{stride}i", period=f"{window_size}i", closed="left")
    edge_agg = [
        pl.col(edge_policy.src_col).alias(edge_policy.src_alias),
        pl.col(edge_policy.dst_col).shift(-edge_policy.dst_shift).alias(edge_policy.dst_alias),
        *edge_stat_exprs,
    ]
    edge_cols = [edge_policy.src_alias, edge_policy.dst_alias, *edge_feature_names]
    base_select = ["_row", edge_policy.src_col]
    if edge_policy.dst_col != edge_policy.src_col:
        base_select.append(edge_policy.dst_col)
    if "timestamp" not in base_select:
        base_select.append("timestamp")
    base_select.extend(c for c in edge_base_cols if c not in base_select)
    edge_df = (
        lf.select(*base_select)
        .group_by_dynamic("_row", **dyn)
        .agg(*edge_agg)
        .rename({"_row": "_wid"})
        .explode(edge_cols)
        .rename({edge_policy.src_alias: "src", edge_policy.dst_alias: "dst"})
    ).collect()
    filters = [pl.col("dst").is_not_null()]
    filters.extend(pl.col(c).is_not_null() for c in edge_feature_names)
    edge_df = edge_df.filter(pl.all_horizontal(*filters))
    _dump_stage(debug_dir, "04_edges_generated", edge_df)
    return edge_df


def _aggregate(
    windowed: WindowedRows,
    *,
    node_stat_exprs: list[pl.Expr],
    label_exprs: list[pl.Expr],
    edge_policy: EdgePolicy,
    edge_stat_exprs: list[pl.Expr],
    edge_base_cols: list[str],
    edge_feature_names: list[str],
    window_size: int,
    stride: int,
    debug_dir: Path | None = None,
) -> AggregatedTables:
    node_stats, labels = _aggregate_nodes_labels(
        windowed,
        node_stat_exprs=node_stat_exprs,
        label_exprs=label_exprs,
        window_size=window_size,
        stride=stride,
        debug_dir=debug_dir,
    )
    edge_df = _generate_edges(
        windowed,
        edge_policy=edge_policy,
        edge_stat_exprs=edge_stat_exprs,
        edge_base_cols=edge_base_cols,
        edge_feature_names=edge_feature_names,
        window_size=window_size,
        stride=stride,
        debug_dir=debug_dir,
    )
    log.info("features_computed", stats=len(node_stats), edges=len(edge_df))
    return AggregatedTables(node_stats=node_stats, edge_df=edge_df, labels=labels)


def _trim_complete_windows(
    tables: AggregatedTables,
    *,
    max_wid: int,
    debug_dir: Path | None = None,
) -> AggregatedTables:
    node_stats = tables.node_stats.filter(pl.col("_wid") <= max_wid)
    edge_df = tables.edge_df.filter(pl.col("_wid") <= max_wid)
    labels = tables.labels.filter(pl.col("_wid") <= max_wid)
    _dump_stage(debug_dir, "05_node_stats_trimmed", node_stats)
    _dump_stage(debug_dir, "06_edges_trimmed", edge_df)
    _dump_stage(debug_dir, "07_labels_trimmed", labels)
    return AggregatedTables(node_stats=node_stats, edge_df=edge_df, labels=labels)


def _apply_graph_transforms(
    tables: AggregatedTables,
    *,
    graph_transforms: list[GraphTransform],
    debug_dir: Path | None = None,
) -> AggregatedTables:
    node_stats, edge_df = tables.node_stats, tables.edge_df
    for i, transform in enumerate(graph_transforms, start=1):
        node_stats, edge_df = transform.apply(node_stats, edge_df)
        _dump_stage(debug_dir, f"08_transform_{i:02d}_{transform.name}_node", node_stats)
        _dump_stage(debug_dir, f"08_transform_{i:02d}_{transform.name}_edge", edge_df)
    return AggregatedTables(node_stats=node_stats, edge_df=edge_df, labels=tables.labels)


def _keep_windows_with_edges(
    tables: AggregatedTables,
    *,
    debug_dir: Path | None = None,
) -> AggregatedTables:
    edge_wids = tables.edge_df.select("_wid").unique()
    node_stats = tables.node_stats.join(edge_wids, on="_wid", how="semi")
    labels = tables.labels.join(node_stats.select("_wid").unique(), on="_wid", how="semi")
    _dump_stage(debug_dir, "09_node_stats_with_edges", node_stats)
    _dump_stage(debug_dir, "10_labels_with_edges", labels)
    return AggregatedTables(node_stats=node_stats, edge_df=tables.edge_df, labels=labels)


def _localize_ids(
    tables: AggregatedTables,
    *,
    debug_dir: Path | None = None,
) -> AggregatedTables:
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
    _dump_stage(debug_dir, "11_node_stats_localized", node_stats)
    _dump_stage(debug_dir, "12_edges_localized", edge_df)
    return AggregatedTables(node_stats=node_stats, edge_df=edge_df, labels=tables.labels)


def _with_tags(table: pl.DataFrame, tags: dict[str, object]) -> pl.DataFrame:
    if table.is_empty() or not tags:
        return table
    return table.with_columns([pl.lit(value).alias(name) for name, value in tags.items()])


def _offset_wids(tables: AggregatedTables, offset: int) -> AggregatedTables:
    if offset == 0:
        return tables
    return AggregatedTables(
        node_stats=tables.node_stats.with_columns((pl.col("_wid") + offset).alias("_wid")),
        edge_df=tables.edge_df.with_columns((pl.col("_wid") + offset).alias("_wid")),
        labels=tables.labels.with_columns((pl.col("_wid") + offset).alias("_wid")),
    )


def _build_graph_tables_windowed(
    df: pl.DataFrame,
    *,
    window_size: int,
    stride: int,
    node_stat_exprs: list[pl.Expr],
    label_exprs: list[pl.Expr],
    edge_policy: EdgePolicy | None = None,
    edge_stat_exprs: list[pl.Expr],
    edge_base_cols: list[str],
    graph_transforms: list[GraphTransform] | None = None,
    debug_artifacts_dir: str | Path | None = None,
    tags: dict[str, object] | None = None,
) -> GraphTables:
    debug_dir = Path(debug_artifacts_dir) if debug_artifacts_dir else None
    segment = WindowSegmenter(window_size, stride).segment(df)
    _dump_stage(debug_dir, "01_windowed_rows", segment.rows)
    if segment.n_windows == 0:
        log.warning("no_complete_windows", n_rows=segment.n_rows, window_size=window_size)
        return GraphTables(pl.DataFrame(), pl.DataFrame(), pl.DataFrame(), segment.n_rows)

    edge_policy = edge_policy or temporal_edge_policy()
    graph_transforms = graph_transforms or default_graph_transforms()
    log.info(
        "edge_policy",
        name=edge_policy.name,
        src_col=edge_policy.src_col,
        dst_col=edge_policy.dst_col,
        dst_shift=edge_policy.dst_shift,
    )
    edge_feature_names = [e.meta.output_name() for e in edge_stat_exprs]
    tables = _aggregate(
        segment,
        node_stat_exprs=node_stat_exprs,
        label_exprs=label_exprs,
        edge_policy=edge_policy,
        edge_stat_exprs=edge_stat_exprs,
        edge_base_cols=edge_base_cols,
        edge_feature_names=edge_feature_names,
        window_size=window_size,
        stride=stride,
        debug_dir=debug_dir,
    )
    tables = _trim_complete_windows(tables, max_wid=segment.max_wid, debug_dir=debug_dir)
    tables = _apply_graph_transforms(tables, graph_transforms=graph_transforms, debug_dir=debug_dir)
    tables = _keep_windows_with_edges(tables, debug_dir=debug_dir)
    if tables.node_stats.is_empty():
        log.warning("no_graphs_with_edges")
        return GraphTables(pl.DataFrame(), pl.DataFrame(), pl.DataFrame(), segment.n_rows)
    tables = _localize_ids(tables, debug_dir=debug_dir)
    if tags:
        tables = AggregatedTables(
            node_stats=_with_tags(tables.node_stats, tags),
            edge_df=_with_tags(tables.edge_df, tags),
            labels=_with_tags(tables.labels, tags),
        )
    return GraphTables(
        node_stats=tables.node_stats,
        edge_df=tables.edge_df,
        labels=tables.labels,
        n_rows=segment.n_rows,
    )


def build_graph_tables(
    df: pl.DataFrame,
    *,
    node_stat_exprs: list[pl.Expr],
    label_exprs: list[pl.Expr],
    edge_policy: EdgePolicy | None = None,
    edge_stat_exprs: list[pl.Expr],
    edge_base_cols: list[str],
    graph_transforms: list[GraphTransform] | None = None,
    debug_artifacts_dir: str | Path | None = None,
    segment_cfg: WindowSegmentCfg | SequenceSegmentCfg | MultiScaleSegmentCfg | EntitySegmentCfg | None = None,
) -> GraphTables:
    """Compose the graph preprocessing primitives into staged graph tables."""
    if segment_cfg is None:
        raise ValueError("build_graph_tables requires an explicit segment_cfg")
    if isinstance(segment_cfg, MultiScaleSegmentCfg):
        out: list[GraphTables] = []
        offset = 0
        for scale_id, scale_window_size in enumerate(segment_cfg.window_sizes):
            tables = _build_graph_tables_windowed(
                df,
                window_size=scale_window_size,
                stride=segment_cfg.stride,
                node_stat_exprs=node_stat_exprs,
                label_exprs=label_exprs,
                edge_policy=edge_policy,
                edge_stat_exprs=edge_stat_exprs,
                edge_base_cols=edge_base_cols,
                graph_transforms=graph_transforms,
                debug_artifacts_dir=debug_artifacts_dir,
                tags={"scale_id": scale_id, "scale_window_size": scale_window_size},
            )
            if tables.node_stats.is_empty():
                continue
            out.append(GraphTables(
                node_stats=tables.node_stats.with_columns((pl.col("_wid") + offset).alias("_wid")),
                edge_df=tables.edge_df.with_columns((pl.col("_wid") + offset).alias("_wid")),
                labels=tables.labels.with_columns((pl.col("_wid") + offset).alias("_wid")),
                n_rows=tables.n_rows,
            ))
            offset += int(tables.node_stats.select(pl.col("_wid").max()).item()) + 1
        if not out:
            return GraphTables(pl.DataFrame(), pl.DataFrame(), pl.DataFrame(), len(df))
        return GraphTables(
            node_stats=pl.concat([t.node_stats for t in out], how="vertical"),
            edge_df=pl.concat([t.edge_df for t in out], how="vertical"),
            labels=pl.concat([t.labels for t in out], how="vertical"),
            n_rows=len(df),
        )

    if isinstance(segment_cfg, SequenceSegmentCfg):
        return _build_graph_tables_windowed(
            df,
            window_size=segment_cfg.window_size + (segment_cfg.sequence_length - 1) * segment_cfg.sequence_stride * segment_cfg.stride,
            stride=max(1, segment_cfg.sequence_stride * segment_cfg.stride),
            node_stat_exprs=node_stat_exprs,
            label_exprs=label_exprs,
            edge_policy=edge_policy,
            edge_stat_exprs=edge_stat_exprs,
            edge_base_cols=edge_base_cols,
            graph_transforms=graph_transforms,
            debug_artifacts_dir=debug_artifacts_dir,
            tags={
                "sequence_length": segment_cfg.sequence_length,
                "sequence_stride": segment_cfg.sequence_stride,
            },
        )

    if isinstance(segment_cfg, EntitySegmentCfg):
        return _build_graph_tables_windowed(
            df,
            window_size=segment_cfg.history_window_size + segment_cfg.future_window_size + 1,
            stride=max(1, segment_cfg.future_window_size or 1),
            node_stat_exprs=node_stat_exprs,
            label_exprs=label_exprs,
            edge_policy=edge_policy,
            edge_stat_exprs=edge_stat_exprs,
            edge_base_cols=edge_base_cols,
            graph_transforms=graph_transforms,
            debug_artifacts_dir=debug_artifacts_dir,
            tags={
                "anchor_column": segment_cfg.anchor_column,
                "anchor_value": segment_cfg.anchor_value if segment_cfg.anchor_value is not None else "",
            },
        )

    if isinstance(segment_cfg, WindowSegmentCfg):
        return _build_graph_tables_windowed(
            df,
            window_size=segment_cfg.window_size,
            stride=segment_cfg.stride,
            node_stat_exprs=node_stat_exprs,
            label_exprs=label_exprs,
            edge_policy=edge_policy,
            edge_stat_exprs=edge_stat_exprs,
            edge_base_cols=edge_base_cols,
            graph_transforms=graph_transforms,
            debug_artifacts_dir=debug_artifacts_dir,
        )
    raise TypeError(f"unsupported segment config: {type(segment_cfg)!r}")
