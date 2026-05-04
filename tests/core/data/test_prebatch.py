"""Pre-batching pipeline tests.

Tests the pre-batch path: NodeBudgetBatchSampler plans → Batch.from_data_list
→ plain list of Batches → prebatched_loader.
"""

from __future__ import annotations

import torch
from conftest import make_graph
from torch_geometric.data import Batch

from graphids.core.data.datamodule.graph import _prebatched_loader
from graphids.core.data.datamodule.sampler import NodeBudgetBatchSampler


def _prebatch(graphs, budget):
    """Helper: run sampler + collate, same logic as GraphDataModule."""
    sizes = torch.tensor([g.num_nodes for g in graphs], dtype=torch.long)
    sampler = NodeBudgetBatchSampler(
        sizes,
        max_num=budget,
        shuffle=False,
    )
    plans = list(sampler)
    return [Batch.from_data_list([graphs[i] for i in plan]) for plan in plans]


def _make_graphs(n_graphs=30, node_range=(4, 20)):
    """Synthetic dataset with deterministic variable-size graphs."""
    return [
        make_graph(
            num_nodes=node_range[0] + (i * 3) % (node_range[1] - node_range[0] + 1),
            num_edges=(node_range[0] + (i * 3) % (node_range[1] - node_range[0] + 1)) * 2,
        )
        for i in range(n_graphs)
    ]


class TestPreBatch:
    """Pre-batching correctness — graphs survive collation unchanged."""

    def test_all_graphs_covered(self):
        """INVARIANT: no graphs dropped when all fit within budget."""
        graphs = _make_graphs(n_graphs=50, node_range=(4, 15))
        batches = _prebatch(graphs, budget=60)  # max 15 < 60, all fit

        total = sum(b.num_graphs for b in batches)
        assert total == len(graphs), f"Pre-batched {total} graphs but dataset has {len(graphs)}"

    def test_batches_respect_node_budget(self):
        """INVARIANT: each batch has at most budget nodes."""
        graphs = _make_graphs(n_graphs=50, node_range=(4, 15))
        batches = _prebatch(graphs, budget=60)

        for i, batch in enumerate(batches):
            assert isinstance(batch, Batch)
            assert batch.num_nodes <= 60, f"Batch {i} has {batch.num_nodes} nodes, budget is 60"

    def test_node_features_preserved(self):
        """INVARIANT: pre-batching preserves node feature values and order."""
        graphs = _make_graphs(n_graphs=10, node_range=(5, 5))
        batches = _prebatch(graphs, budget=25)

        all_x_pre = torch.cat([b.x for b in batches], dim=0)
        all_x_orig = torch.cat([g.x for g in graphs], dim=0)
        assert all_x_pre.shape == all_x_orig.shape
        assert torch.allclose(all_x_pre, all_x_orig)

    def test_dataloader_yields_batches_directly(self):
        """INVARIANT: prebatched_loader yields Batch objects, not nested."""
        graphs = _make_graphs(n_graphs=20)
        batches = _prebatch(graphs, budget=100)
        loader = _prebatched_loader(batches, shuffle=False)

        yielded = list(loader)
        assert len(yielded) == len(batches)
        assert all(isinstance(b, Batch) for b in yielded)

    def test_shuffle_changes_batch_order(self):
        """INVARIANT: shuffle=True permutes batch order across iterations."""
        graphs = _make_graphs(n_graphs=60, node_range=(4, 10))
        batches = _prebatch(graphs, budget=30)
        loader = _prebatched_loader(batches, shuffle=True)

        order_1 = [b.num_nodes for b in loader]
        order_2 = [b.num_nodes for b in loader]

        assert len(order_1) >= 5, "Need enough batches to test shuffle"
        assert order_1 != order_2, "Two shuffled iterations produced identical order"


