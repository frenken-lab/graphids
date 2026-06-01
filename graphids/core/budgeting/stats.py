"""Dataset size extraction for budget planning."""

from __future__ import annotations

import torch

from .config import BudgetConfig


def _as_long_tensor(value) -> torch.Tensor | None:
    if value is None:
        return None
    if callable(value):
        value = value()
    return torch.as_tensor(value, dtype=torch.long).cpu().view(-1)


def _dataset_size_tensors(
    train_dataset,
    *,
    config: BudgetConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read graph sizes without materializing every graph when caches expose them."""
    cfg = config or BudgetConfig.from_env()
    if train_dataset is None:
        return torch.ones(1, dtype=torch.long), torch.full(
            (1,), int(cfg.default_edges_per_node), dtype=torch.long
        )

    sizes_t = _as_long_tensor(getattr(train_dataset, "num_nodes_per_graph", None))
    edge_sizes_t = _as_long_tensor(getattr(train_dataset, "num_edges_per_graph", None))
    if sizes_t is not None and edge_sizes_t is not None:
        if sizes_t.numel() != edge_sizes_t.numel():
            raise ValueError(
                f"num_nodes_per_graph length ({sizes_t.numel()}) != "
                f"num_edges_per_graph length ({edge_sizes_t.numel()})"
            )
        return sizes_t, edge_sizes_t

    sizes: list[int] = []
    edge_sizes: list[int] = []
    for graph in train_dataset:
        sizes.append(int(graph.num_nodes))
        edge_sizes.append(int(graph.num_edges))
    return torch.tensor(sizes, dtype=torch.long), torch.tensor(edge_sizes, dtype=torch.long)


def _dataset_size_stats(train_dataset) -> tuple[int, int, int, float]:
    sizes_t, edge_sizes_t = _dataset_size_tensors(train_dataset)
    sizes = [int(v) for v in sizes_t.tolist()]
    edge_sizes = [int(v) for v in edge_sizes_t.tolist()]
    if not sizes:
        raise RuntimeError("budget heuristic: train_dataset is empty")
    total_nodes = sum(sizes)
    total_edges = sum(edge_sizes)
    epn = total_edges / max(1, total_nodes)
    return max(sizes), max(edge_sizes), total_nodes, max(epn, 1.0)
