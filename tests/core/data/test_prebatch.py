"""Offline graph packer — the live primitive backing the prebatch path.

The prior live ``NodeBudgetBatchSampler`` was confirmed dead and removed in
``graph_v2`` (see ``datamodule/sampler.py`` docstring). ``pack_offline`` is
the surviving allocator. The dual node + edge budget is the load-bearing
invariant per ``.claude/rules/critical-constraints.md`` — single-axis
admission allowed edge-heavy OOMs.
"""

from __future__ import annotations

import pytest
import torch

from graphids.core.data.datamodule.sampler import pack_offline


def _sizes(seq: list[int]) -> torch.Tensor:
    return torch.tensor(seq, dtype=torch.long)


class TestPackOffline:
    def test_all_graphs_placed_when_within_budget(self):
        # INVARIANT: nothing dropped when every graph fits the node budget alone.
        sizes = _sizes([4, 7, 5, 9, 3, 6])
        bins = pack_offline(sizes, max_num=60)
        placed = sorted(i for b in bins for i in b)
        assert placed == list(range(len(sizes)))

    def test_per_bin_node_budget_respected(self):
        # INVARIANT: bin's node sum never exceeds max_num — the FFD invariant
        # the packer is responsible for.
        sizes = _sizes([4 + (i * 3) % 12 for i in range(50)])
        bins = pack_offline(sizes, max_num=60)
        for plan in bins:
            assert int(sizes[plan].sum()) <= 60

    def test_dual_budget_closes_on_either_axis(self):
        # CONTRACT: when edge_sizes + max_edges supplied, a bin closes when
        # adding a graph would exceed EITHER node OR edge budget. Regression
        # against single-axis admission (edge-heavy OOMs — see
        # critical-constraints.md).
        sizes = _sizes([5] * 10)
        edges = _sizes([200] * 10)  # tight on edges
        bins = pack_offline(sizes, max_num=1000, edge_sizes=edges, max_edges=500)
        for plan in bins:
            assert int(sizes[plan].sum()) <= 1000
            assert int(edges[plan].sum()) <= 500

    def test_oversize_graph_skipped_not_raised(self):
        # CONTRACT: a single graph that exceeds either budget is dropped with
        # a logged warning; the rest still pack. Better than aborting an
        # 8-hour fit because one graph is pathological.
        sizes = _sizes([3, 100, 4])  # idx 1 oversize on nodes
        bins = pack_offline(sizes, max_num=10)
        placed = {i for b in bins for i in b}
        assert placed == {0, 2}

    def test_invalid_budgets_rejected(self):
        with pytest.raises(ValueError, match="max_num"):
            pack_offline(_sizes([1, 2]), max_num=0)
        with pytest.raises(ValueError, match="max_edges"):
            pack_offline(_sizes([1, 2]), max_num=10, edge_sizes=_sizes([1, 2]))
        with pytest.raises(ValueError, match="length"):
            pack_offline(_sizes([1, 2, 3]), max_num=10, edge_sizes=_sizes([1, 2]), max_edges=10)

    def test_large_pack_stays_linear_in_bin_count(self):
        sizes = _sizes([10] * 10_000)
        bins = pack_offline(sizes, max_num=100)
        assert len(bins) == 1_000
