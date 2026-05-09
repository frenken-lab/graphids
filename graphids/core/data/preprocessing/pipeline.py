"""Sliding-window tables and tensor packing for graph cache builds.

Schema-driven (GraphSchema's Polars expressions + column orders).

Main APIs:
- ``build_tables(df, window_size, stride)`` for staged DataFrame outputs
- ``run(df, window_size, stride)`` for pre-collated PyG cache tensors
- ``inspect(df, window_size, stride)`` to retrieve intermediate stages
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import torch
from structlog import get_logger
from torch_geometric.data import Data

from graphids.core.data.preprocessing.edge_policy import (
    EdgePolicy,
    temporal_edge_policy,
)
from graphids.core.data.preprocessing.graph_ops import (
    GraphTransform,
    default_graph_transforms,
)

log = get_logger(__name__)


def _slices_from_counts(counts: pl.Series) -> torch.Tensor:
    return torch.cat(
        [
            torch.zeros(1, dtype=torch.long),
            torch.from_numpy(counts.to_numpy()).cumsum(0).to(torch.long),
        ]
    )


@dataclass(frozen=True)
class WindowedRows:
    rows: pl.DataFrame
    n_rows: int
    n_windows: int
    max_wid: int


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


class GraphPipeline:
    """Composable time-series -> graph table -> tensor preprocessing pipeline."""

    def __init__(
        self,
        *,
        node_stat_exprs: list[pl.Expr],
        edge_stat_exprs: list[pl.Expr],
        node_col_order: list[str],
        edge_col_order: tuple[str, ...],
        label_exprs: list[pl.Expr],
        edge_base_cols: list[str],
        edge_policy: EdgePolicy | None = None,
        graph_transforms: list[GraphTransform] | None = None,
        debug_artifacts_dir: str | Path | None = None,
    ):
        self.node_stat_exprs = node_stat_exprs
        self.edge_stat_exprs = edge_stat_exprs
        self.node_col_order = node_col_order
        self.edge_col_order = edge_col_order
        self.label_exprs = label_exprs
        self.edge_base_cols = edge_base_cols
        self.edge_policy = edge_policy or temporal_edge_policy()
        self.graph_transforms = graph_transforms or default_graph_transforms()
        self.debug_artifacts_dir = Path(debug_artifacts_dir) if debug_artifacts_dir else None
        self.label_names = [e.meta.output_name() for e in label_exprs]
        self.edge_feature_names = [e.meta.output_name() for e in edge_stat_exprs]
        assert self.label_names[0] == "y", f"first label expr must alias 'y', got {self.label_names[0]!r}"

    def _stage_summary(self, table: pl.DataFrame) -> dict[str, object]:
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

    def _dump_stage(self, stage: str, table: pl.DataFrame) -> None:
        if self.debug_artifacts_dir is None:
            return
        self.debug_artifacts_dir.mkdir(parents=True, exist_ok=True)
        table.write_parquet(self.debug_artifacts_dir / f"{stage}.parquet")
        (self.debug_artifacts_dir / f"{stage}.summary.json").write_text(
            json.dumps(self._stage_summary(table), indent=2),
            encoding="utf-8",
        )

    def _assign_windows(self, df: pl.DataFrame, window_size: int, stride: int) -> WindowedRows:
        rows = df.with_row_index("_row").with_columns(pl.col("_row").cast(pl.Int64))
        n_rows = len(rows)
        n_windows = max(0, (n_rows - window_size) // stride + 1)
        max_wid = (n_windows - 1) * stride
        rows = rows.with_columns((pl.col("_row") % window_size < (window_size // 2)).alias("_first_half"))
        self._dump_stage("01_windowed_rows", rows)
        return WindowedRows(rows=rows, n_rows=n_rows, n_windows=n_windows, max_wid=max_wid)

    def _aggregate_nodes_labels(self, windowed: WindowedRows, *, window_size: int, stride: int) -> tuple[pl.DataFrame, pl.DataFrame]:
        lf = windowed.rows.lazy().sort("_row")
        dyn = dict(every=f"{stride}i", period=f"{window_size}i", closed="left")
        node_lf = (
            lf.group_by_dynamic("_row", **dyn, group_by="node_id")
            .agg(*self.node_stat_exprs)
            .fill_null(0)
            .fill_nan(0)
            .rename({"_row": "_wid"})
        )
        labels_lf = lf.group_by_dynamic("_row", **dyn).agg(*self.label_exprs).rename({"_row": "_wid"})
        node_stats, labels = pl.collect_all([node_lf, labels_lf])
        self._dump_stage("02_node_stats_raw", node_stats)
        self._dump_stage("03_labels_raw", labels)
        return node_stats, labels

    def _generate_edges(self, windowed: WindowedRows, *, window_size: int, stride: int) -> pl.DataFrame:
        lf = windowed.rows.lazy().sort("_row")
        dyn = dict(every=f"{stride}i", period=f"{window_size}i", closed="left")
        edge_agg = [
            pl.col(self.edge_policy.src_col).alias(self.edge_policy.src_alias),
            pl.col(self.edge_policy.dst_col)
            .shift(self.edge_policy.dst_shift)
            .alias(self.edge_policy.dst_alias),
            *self.edge_stat_exprs,
        ]
        edge_cols = [self.edge_policy.src_alias, self.edge_policy.dst_alias, *self.edge_feature_names]
        base_select = ["_row", self.edge_policy.src_col]
        if self.edge_policy.dst_col != self.edge_policy.src_col:
            base_select.append(self.edge_policy.dst_col)
        if "timestamp" not in base_select:
            base_select.append("timestamp")
        base_select.extend(c for c in self.edge_base_cols if c not in base_select)
        edge_df = (
            lf.select(*base_select)
            .group_by_dynamic("_row", **dyn)
            .agg(*edge_agg)
            .rename({"_row": "_wid"})
            .explode(edge_cols)
            .rename({self.edge_policy.src_alias: "src", self.edge_policy.dst_alias: "dst"})
        ).collect()
        filters = [pl.col("dst").is_not_null()]
        filters.extend(pl.col(c).is_not_null() for c in self.edge_feature_names)
        edge_df = edge_df.filter(pl.all_horizontal(*filters))
        self._dump_stage("04_edges_generated", edge_df)
        return edge_df

    def _aggregate(self, windowed: WindowedRows, *, window_size: int, stride: int) -> AggregatedTables:
        node_stats, labels = self._aggregate_nodes_labels(windowed, window_size=window_size, stride=stride)
        edge_df = self._generate_edges(windowed, window_size=window_size, stride=stride)
        log.info("features_computed", stats=len(node_stats), edges=len(edge_df))
        return AggregatedTables(node_stats=node_stats, edge_df=edge_df, labels=labels)

    def _trim_complete_windows(self, tables: AggregatedTables, *, max_wid: int) -> AggregatedTables:
        node_stats = tables.node_stats.filter(pl.col("_wid") <= max_wid)
        edge_df = tables.edge_df.filter(pl.col("_wid") <= max_wid)
        labels = tables.labels.filter(pl.col("_wid") <= max_wid)
        self._dump_stage("05_node_stats_trimmed", node_stats)
        self._dump_stage("06_edges_trimmed", edge_df)
        self._dump_stage("07_labels_trimmed", labels)
        return AggregatedTables(node_stats=node_stats, edge_df=edge_df, labels=labels)

    def _apply_graph_transforms(self, tables: AggregatedTables) -> AggregatedTables:
        node_stats, edge_df = tables.node_stats, tables.edge_df
        for i, transform in enumerate(self.graph_transforms, start=1):
            node_stats, edge_df = transform.apply(node_stats, edge_df)
            self._dump_stage(f"08_transform_{i:02d}_{transform.name}_node", node_stats)
            self._dump_stage(f"08_transform_{i:02d}_{transform.name}_edge", edge_df)
        return AggregatedTables(node_stats=node_stats, edge_df=edge_df, labels=tables.labels)

    def _keep_windows_with_edges(self, tables: AggregatedTables) -> AggregatedTables:
        node_stats = tables.node_stats.filter(pl.col("_wid").is_in(tables.edge_df["_wid"].unique()))
        labels = tables.labels.filter(pl.col("_wid").is_in(node_stats["_wid"].unique()))
        self._dump_stage("09_node_stats_with_edges", node_stats)
        self._dump_stage("10_labels_with_edges", labels)
        return AggregatedTables(node_stats=node_stats, edge_df=tables.edge_df, labels=labels)

    def _localize_ids(self, tables: AggregatedTables) -> AggregatedTables:
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
        self._dump_stage("11_node_stats_localized", node_stats)
        self._dump_stage("12_edges_localized", edge_df)
        return AggregatedTables(node_stats=node_stats, edge_df=edge_df, labels=tables.labels)

    def _build_tables_internal(
        self,
        df: pl.DataFrame,
        *,
        window_size: int,
        stride: int,
        collect_stages: bool,
    ) -> tuple[GraphTables, dict[str, pl.DataFrame]]:
        stage_views: dict[str, pl.DataFrame] = {}
        windowed = self._assign_windows(df, window_size, stride)
        if collect_stages:
            stage_views["windowed_rows"] = windowed.rows
        if windowed.n_windows == 0:
            log.warning("no_complete_windows", n_rows=windowed.n_rows, window_size=window_size)
            return (
                GraphTables(pl.DataFrame(), pl.DataFrame(), pl.DataFrame(), windowed.n_rows),
                stage_views,
            )

        log.info(
            "edge_policy",
            name=self.edge_policy.name,
            src_col=self.edge_policy.src_col,
            dst_col=self.edge_policy.dst_col,
            dst_shift=self.edge_policy.dst_shift,
        )
        tables = self._aggregate(windowed, window_size=window_size, stride=stride)
        if collect_stages:
            stage_views["aggregated_node_stats"] = tables.node_stats
            stage_views["aggregated_edges"] = tables.edge_df
            stage_views["aggregated_labels"] = tables.labels
        tables = self._trim_complete_windows(tables, max_wid=windowed.max_wid)
        tables = self._apply_graph_transforms(tables)
        if collect_stages:
            stage_views["transformed_node_stats"] = tables.node_stats
            stage_views["transformed_edges"] = tables.edge_df
        tables = self._keep_windows_with_edges(tables)
        if tables.node_stats.is_empty():
            log.warning("no_graphs_with_edges")
            return (
                GraphTables(pl.DataFrame(), pl.DataFrame(), pl.DataFrame(), windowed.n_rows),
                stage_views,
            )
        tables = self._localize_ids(tables)
        if collect_stages:
            stage_views["localized_node_stats"] = tables.node_stats
            stage_views["localized_edges"] = tables.edge_df
            stage_views["labels"] = tables.labels
        return (
            GraphTables(
                node_stats=tables.node_stats,
                edge_df=tables.edge_df,
                labels=tables.labels,
                n_rows=windowed.n_rows,
            ),
            stage_views,
        )

    def build_tables(self, df: pl.DataFrame, window_size: int, stride: int) -> GraphTables:
        tables, _ = self._build_tables_internal(
            df,
            window_size=window_size,
            stride=stride,
            collect_stages=False,
        )
        return tables

    def inspect(self, df: pl.DataFrame, window_size: int, stride: int) -> dict[str, pl.DataFrame]:
        _, stages = self._build_tables_internal(
            df,
            window_size=window_size,
            stride=stride,
            collect_stages=True,
        )
        return stages

    def _to_pyg(self, tables: GraphTables) -> tuple[Data, dict, int, int]:
        x = (
            tables.node_stats.select(self.node_col_order)
            .fill_null(0)
            .fill_nan(0)
            .to_torch(dtype=pl.Float32)
        )
        node_id = tables.node_stats.select("node_id").to_torch(dtype=pl.Int64).squeeze(-1)
        edge_index = (
            tables.edge_df.select("src_local", "dst_local").to_torch(dtype=pl.Int64).t().contiguous()
        )
        edge_attr = (
            tables.edge_df.select(list(self.edge_col_order))
            .fill_null(0)
            .fill_nan(0)
            .to_torch(dtype=pl.Float32)
        )
        kept_wids = tables.node_stats.group_by("_wid", maintain_order=True).first().select("_wid")
        num_graphs = len(kept_wids)
        node_counts = tables.node_stats.group_by("_wid", maintain_order=True).len()["len"]
        edge_counts = tables.edge_df.group_by("_wid", maintain_order=True).len()["len"]
        node_slice = _slices_from_counts(node_counts)
        edge_slice = _slices_from_counts(edge_counts)
        graph_idx = torch.arange(num_graphs + 1, dtype=torch.long)
        labels_aligned = kept_wids.join(tables.labels, on="_wid", how="left").fill_null(0)
        label_tensors = {
            n: labels_aligned.select(n).to_torch(dtype=pl.Int64).squeeze(-1)
            for n in self.label_names
        }
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, node_id=node_id, **label_tensors)
        slices = {
            "x": node_slice,
            "edge_index": edge_slice,
            "edge_attr": edge_slice,
            "node_id": node_slice,
            **{n: graph_idx for n in self.label_names},
        }
        return data, slices, num_graphs, tables.n_rows

    def run(self, df: pl.DataFrame, window_size: int, stride: int) -> tuple[Data, dict, int, int]:
        tables = self.build_tables(df, window_size, stride)
        if tables.node_stats.is_empty():
            return Data(), {}, 0, tables.n_rows
        data, slices, num_graphs, n_rows = self._to_pyg(tables)
        log.info("graphs_built", count=num_graphs)
        return data, slices, num_graphs, n_rows
