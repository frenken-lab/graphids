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
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, node_id=node_id, **label_tensors)
    slices = {
        "x": node_slice,
        "edge_index": edge_slice,
        "edge_attr": edge_slice,
        "node_id": node_slice,
        **{n: graph_idx for n in label_names},
    }
    return data, slices, num_graphs, tables.n_rows
