"""Pre-batching pipeline tests.

Tests the pre-batch path: NodeBudgetBatchSampler plans → Batch.from_data_list
→ plain list of Batches → make_graph_loader(batch_size=None).
"""

from __future__ import annotations

import math

import torch
from conftest import make_graph
from torch_geometric.data import Batch

from graphids.core.data.sampler import NodeBudgetBatchSampler, make_graph_loader


def _prebatch(graphs, budget):
    """Helper: run sampler + collate, same logic as GraphDataModule."""
    sizes = torch.tensor([g.num_nodes for g in graphs], dtype=torch.long)
    mean_nodes = sizes.float().mean().item()
    num_steps = max(1, math.ceil(len(graphs) * mean_nodes / budget))
    sampler = NodeBudgetBatchSampler(
        sizes,
        max_num=budget,
        shuffle=False,
        skip_too_big=True,
        num_steps=num_steps,
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
        """INVARIANT: make_graph_loader(batch_size=None) yields Batch objects, not nested."""
        graphs = _make_graphs(n_graphs=20)
        batches = _prebatch(graphs, budget=100)
        loader = make_graph_loader(batches, batch_size=None, shuffle=False, num_workers=0)

        yielded = list(loader)
        assert len(yielded) == len(batches)
        assert all(isinstance(b, Batch) for b in yielded)

    def test_shuffle_changes_batch_order(self):
        """INVARIANT: shuffle=True permutes batch order across iterations."""
        graphs = _make_graphs(n_graphs=60, node_range=(4, 10))
        batches = _prebatch(graphs, budget=30)
        loader = make_graph_loader(batches, batch_size=None, shuffle=True, num_workers=0)

        order_1 = [b.num_nodes for b in loader]
        order_2 = [b.num_nodes for b in loader]

        assert len(order_1) >= 5, "Need enough batches to test shuffle"
        assert order_1 != order_2, "Two shuffled iterations produced identical order"


class TestTierPreBatch:
    """Tier-based curriculum pre-batching invariants."""

    @staticmethod
    def _make_tiered_batches(n_normal=60, n_attack=15, num_tiers=3, budget=50):
        """Pre-batch synthetic tiers, mirroring GraphDataModule._prebatch_tiers."""
        normals = [
            make_graph(num_nodes=5 + i % 10, num_edges=(5 + i % 10) * 2) for i in range(n_normal)
        ]
        attacks = [make_graph(num_nodes=8, num_edges=16) for _ in range(n_attack)]
        full_dataset = normals + attacks
        sizes = torch.tensor([g.num_nodes for g in full_dataset], dtype=torch.long)

        # Bucket normals into tiers (ascending index = ascending "difficulty")
        bucket_size = max(1, math.ceil(n_normal / num_tiers))
        normal_tiers = [
            list(range(start, min(start + bucket_size, n_normal)))
            for start in range(0, n_normal, bucket_size)
        ]
        attack_indices = list(range(n_normal, len(full_dataset)))

        # Pre-batch each tier
        tier_batches = []
        for tier_idx in normal_tiers:
            tier_sizes = sizes[tier_idx]
            num_steps = max(
                1, math.ceil(len(tier_idx) * sizes[tier_idx].float().mean().item() / budget)
            )
            sampler = NodeBudgetBatchSampler(
                tier_sizes,
                max_num=budget,
                shuffle=False,
                skip_too_big=True,
                num_steps=num_steps,
            )
            plans = list(sampler)
            tier_batches.append(
                [Batch.from_data_list([full_dataset[tier_idx[i]] for i in plan]) for plan in plans]
            )

        # Pre-batch attacks
        atk_sizes = sizes[attack_indices]
        num_steps = max(
            1, math.ceil(len(attack_indices) * atk_sizes.float().mean().item() / budget)
        )
        sampler = NodeBudgetBatchSampler(
            atk_sizes,
            max_num=budget,
            shuffle=False,
            skip_too_big=True,
            num_steps=num_steps,
        )
        plans = list(sampler)
        attack_batches = [
            Batch.from_data_list([full_dataset[attack_indices[i]] for i in plan]) for plan in plans
        ]

        return tier_batches, attack_batches, full_dataset, n_normal, n_attack

    def test_all_graphs_across_tiers_covered(self):
        """INVARIANT: every graph appears in exactly one tier's batches."""
        tier_batches, attack_batches, full_dataset, n_normal, n_attack = self._make_tiered_batches()
        normal_graphs = sum(b.num_graphs for tier in tier_batches for b in tier)
        attack_graphs = sum(b.num_graphs for b in attack_batches)
        assert normal_graphs == n_normal, (
            f"Normal tiers have {normal_graphs} graphs, expected {n_normal}"
        )
        assert attack_graphs == n_attack, (
            f"Attack tier has {attack_graphs} graphs, expected {n_attack}"
        )

    def test_tier_batches_respect_budget(self):
        """INVARIANT: each batch within each tier respects node budget."""
        budget = 50
        tier_batches, attack_batches, *_ = self._make_tiered_batches(budget=budget)
        for t_idx, tier in enumerate(tier_batches):
            for b_idx, batch in enumerate(tier):
                assert batch.num_nodes <= budget, (
                    f"Tier {t_idx} batch {b_idx}: {batch.num_nodes} nodes > budget {budget}"
                )
        for b_idx, batch in enumerate(attack_batches):
            assert batch.num_nodes <= budget, (
                f"Attack batch {b_idx}: {batch.num_nodes} nodes > budget {budget}"
            )

    def test_active_tier_concat_yields_valid_batches(self):
        """INVARIANT: concatenating active tiers produces a valid loader."""
        tier_batches, attack_batches, *_ = self._make_tiered_batches(num_tiers=4)
        # Simulate selecting first 2 tiers + attacks
        active = []
        for i in range(2):
            active.extend(tier_batches[i])
        active.extend(attack_batches)

        loader = make_graph_loader(active, batch_size=None, shuffle=False, num_workers=0)
        yielded = list(loader)
        assert len(yielded) == len(active)
        assert all(isinstance(b, Batch) for b in yielded)
