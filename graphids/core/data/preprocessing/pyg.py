"""PyG tensor packing primitives for staged graph tables."""

from __future__ import annotations

import polars as pl
import torch
from torch_geometric.data import Data

from graphids.core.data.preprocessing.materialization import GraphTables


def _slices_from_counts(counts: pl.Series) -> torch.Tensor:
    arr = counts.to_numpy().copy()
    return torch.cat(
        [
            torch.zeros(1, dtype=torch.long),
            torch.from_numpy(arr).cumsum(0).to(torch.long),
        ]
    )


def _optional_tensor(df: pl.DataFrame, col: str, *, dtype) -> torch.Tensor:
    return df.select(col).fill_null(0).fill_nan(0).to_torch(dtype=dtype).squeeze(-1)


def graph_tables_to_pyg(
    tables: GraphTables,
    *,
    node_col_order: list[str],
    edge_col_order: tuple[str, ...],
    label_exprs: list[pl.Expr],
) -> tuple[Data, dict, int, int]:
    """Compose staged tables into pre-collated PyG tensors."""
    label_names = [e.meta.output_name() for e in label_exprs]
    x = tables.node_stats.select(node_col_order).fill_null(0).fill_nan(0).to_torch(dtype=pl.Float32)
    node_id = tables.node_stats.select("node_id").to_torch(dtype=pl.Int64).squeeze(-1)
    edge_index = tables.edge_df.select("src_local", "dst_local").to_torch(dtype=pl.Int64).t().contiguous()
    edge_attr = tables.edge_df.select(list(edge_col_order)).fill_null(0).fill_nan(0).to_torch(dtype=pl.Float32)
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
        for n in label_names
    }
    extra_tensors: dict[str, torch.Tensor] = {}
    extra_slices: dict[str, torch.Tensor] = {}
    node_optional_cols = {
        "sequence_id": "node_sequence_id",
        "sequence_step": "node_sequence_step",
        "sequence_length": "node_sequence_length",
        "sequence_stride": "node_sequence_stride",
        "snapshot_wid": "node_snapshot_wid",
        "window_start_row": "node_window_start_row",
        "window_end_row": "node_window_end_row",
        "window_ordinal": "node_window_ordinal",
    }
    for col, attr in node_optional_cols.items():
        if col in tables.node_stats.columns:
            extra_tensors[attr] = _optional_tensor(tables.node_stats, col, dtype=pl.Int64)
            extra_slices[attr] = node_slice

    edge_optional_cols = {
        "sequence_id": "edge_sequence_id",
        "sequence_step": "edge_sequence_step",
        "sequence_length": "edge_sequence_length",
        "sequence_stride": "edge_sequence_stride",
        "snapshot_wid": "edge_snapshot_wid",
        "window_start_row": "edge_window_start_row",
        "window_end_row": "edge_window_end_row",
        "window_ordinal": "edge_window_ordinal",
    }
    for col, attr in edge_optional_cols.items():
        if col in tables.edge_df.columns:
            extra_tensors[attr] = _optional_tensor(tables.edge_df, col, dtype=pl.Int64)
            extra_slices[attr] = edge_slice

    extra_tensors["graph_wid"] = _optional_tensor(kept_wids, "_wid", dtype=pl.Int64)
    extra_slices["graph_wid"] = graph_idx

    graph_optional_cols = (
        "sequence_id",
        "sequence_length",
        "sequence_stride",
        "target_snapshot_wid",
        "window_start_row",
        "window_end_row",
        "window_ordinal",
    )
    for col in graph_optional_cols:
        if col in labels_aligned.columns:
            extra_tensors[col] = _optional_tensor(labels_aligned, col, dtype=pl.Int64)
            extra_slices[col] = graph_idx

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        node_id=node_id,
        **label_tensors,
        **extra_tensors,
    )
    slices = {
        "x": node_slice,
        "edge_index": edge_slice,
        "edge_attr": edge_slice,
        "node_id": node_slice,
        **{n: graph_idx for n in label_names},
        **extra_slices,
    }
    return data, slices, num_graphs, tables.n_rows
