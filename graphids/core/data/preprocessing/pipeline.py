"""Thin composer over graph preprocessing primitives."""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl
from torch_geometric.data import Data

from graphids.core.data.preprocessing.edge_policy import EdgePolicy
from graphids.core.data.preprocessing.graph_ops import GraphTransform
from graphids.core.data.preprocessing.materialization import GraphTables, build_graph_tables
from graphids.core.data.preprocessing.representations import (
    GraphRepresentationCfg,
    SnapshotRepresentationCfg,
    representation_segment,
)
from graphids.core.data.preprocessing.pyg import graph_tables_to_pyg
from graphids.core.data.preprocessing.segments import (
    EntitySegmentCfg,
    MultiScaleSegmentCfg,
    SequenceSegmentCfg,
    WindowSegmentCfg,
)


@dataclass(frozen=True, slots=True)
class GraphPipeline:
    """Pure config carrier for graph preprocessing primitives."""

    node_stat_exprs: list[pl.Expr]
    edge_stat_exprs: list[pl.Expr]
    node_col_order: list[str]
    edge_col_order: tuple[str, ...]
    label_exprs: list[pl.Expr]
    edge_base_cols: list[str]
    edge_policy: EdgePolicy | None = None
    graph_transforms: list[GraphTransform] | None = None
    debug_artifacts_dir: str | None = None
    representation_cfg: GraphRepresentationCfg = field(default_factory=SnapshotRepresentationCfg)
    segment_cfg: WindowSegmentCfg | SequenceSegmentCfg | MultiScaleSegmentCfg | EntitySegmentCfg | None = None


def build_tables(pipeline: GraphPipeline, df: pl.DataFrame):
    return build_graph_tables(
        df,
        node_stat_exprs=pipeline.node_stat_exprs,
        label_exprs=pipeline.label_exprs,
        edge_policy=pipeline.edge_policy,
        edge_stat_exprs=pipeline.edge_stat_exprs,
        edge_base_cols=pipeline.edge_base_cols,
        graph_transforms=pipeline.graph_transforms,
        debug_artifacts_dir=pipeline.debug_artifacts_dir,
        segment_cfg=pipeline.segment_cfg or representation_segment(pipeline.representation_cfg),
    )


def inspect(pipeline: GraphPipeline, df: pl.DataFrame) -> dict[str, pl.DataFrame]:
    tables = build_tables(pipeline, df)
    return {
        "node_stats": tables.node_stats,
        "edge_df": tables.edge_df,
        "labels": tables.labels,
    }


def to_pyg(pipeline: GraphPipeline, tables: GraphTables) -> tuple[Data, dict, int, int]:
    return graph_tables_to_pyg(
        tables,
        node_col_order=pipeline.node_col_order,
        edge_col_order=pipeline.edge_col_order,
        label_exprs=pipeline.label_exprs,
    )


def run(pipeline: GraphPipeline, df: pl.DataFrame) -> tuple[Data, dict, int, int]:
    tables = build_tables(pipeline, df)
    if tables.node_stats.is_empty():
        return Data(), {}, 0, tables.n_rows
    return to_pyg(pipeline, tables)
