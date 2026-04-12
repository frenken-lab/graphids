"""Curriculum sampler tests — exercise the real `build_curriculum_tiers`
plus `make_scorer` factory resolution. The scoring strategy is injected,
so VGAE isn't required.
"""

from __future__ import annotations

import pytest
import torch
from conftest import make_graph

from graphids.core.data.curriculum import (
    DifficultyScorer,
    RandomScorer,
    VGAEScorer,
    active_tier_count,
    bucket_by_score,
    build_curriculum_tiers,
    make_scorer,
)


def _mk_dataset(n_normal=30, n_attack=10):
    """Train_ds-like list of graphs with y ∈ {0, 1}."""
    normals = [make_graph() for _ in range(n_normal)]
    attacks = [make_graph() for _ in range(n_attack)]
    for g in normals:
        g.y = torch.tensor([0])
    for g in attacks:
        g.y = torch.tensor([1])
    return normals + attacks


class _MonotonicScorer:
    """Deterministic scorer for invariant checks: score[i] = i."""

    def score(self, graphs):
        return torch.arange(len(graphs), dtype=torch.float)


class TestBuildCurriculumTiers:
    """Tier bucketing — decoupled from any particular scoring strategy."""

    def test_all_normals_covered(self):
        # INVARIANT: every normal graph lives in exactly one tier.
        ds = _mk_dataset(n_normal=50, n_attack=5)
        _, tiers, _, _, _ = build_curriculum_tiers(ds, RandomScorer(seed=0), num_tiers=7)
        flat = [i for tier in tiers for i in tier]
        assert sorted(flat) == list(range(50))

    def test_tiers_sorted_by_difficulty(self):
        # INVARIANT: with a monotonic scorer, tier k's mean score ≤ tier k+1.
        ds = _mk_dataset(n_normal=50, n_attack=0)
        scores, tiers, _, _, _ = build_curriculum_tiers(ds, _MonotonicScorer(), num_tiers=5)
        means = [scores[t].mean().item() for t in tiers]
        for a, b in zip(means, means[1:]):
            assert a <= b, f"{a} > {b} violates sort order"

    def test_attacks_separate(self):
        # INVARIANT: attack indices come after all normals and point to y=1.
        ds = _mk_dataset(n_normal=30, n_attack=10)
        _, tiers, attacks, full, _ = build_curriculum_tiers(
            ds, RandomScorer(seed=0), num_tiers=5,
        )
        tier_set = {i for t in tiers for i in t}
        assert tier_set.isdisjoint(attacks)
        assert len(attacks) == 10
        for i in attacks:
            assert int(full[i].y[0]) == 1

    def test_scorer_size_mismatch_raises(self):
        # CONTRACT: scorer must return len(normals) scores; mismatch must error.
        class _BrokenScorer:
            def score(self, graphs):
                return torch.zeros(len(graphs) - 1)  # off-by-one

        ds = _mk_dataset(n_normal=10, n_attack=0)
        with pytest.raises(ValueError, match="scorer returned"):
            build_curriculum_tiers(ds, _BrokenScorer(), num_tiers=3)


class TestRandomScorer:
    """RandomScorer — reference baseline that swaps in for VGAE."""

    def test_deterministic_under_seed(self):
        # CONTRACT: same seed → identical scores (reproducibility for ablations).
        ds = _mk_dataset(n_normal=20, n_attack=0)
        a = RandomScorer(seed=13).score(ds)
        b = RandomScorer(seed=13).score(ds)
        assert torch.equal(a, b)

    def test_different_seeds_diverge(self):
        ds = _mk_dataset(n_normal=20, n_attack=0)
        a = RandomScorer(seed=1).score(ds)
        b = RandomScorer(seed=2).score(ds)
        assert not torch.equal(a, b)

    def test_satisfies_protocol(self):
        # CONTRACT: RandomScorer is recognized as a DifficultyScorer.
        assert isinstance(RandomScorer(), DifficultyScorer)


class TestMakeScorer:
    """Scorer spec resolution."""

    def test_class_path_dict(self):
        spec = {
            "class_path": "graphids.core.data.curriculum.RandomScorer",
            "init_args": {"seed": 42},
        }
        s = make_scorer(spec)
        assert isinstance(s, RandomScorer)
        assert s.seed == 42

    def test_instance_passthrough(self):
        s = RandomScorer(seed=5)
        assert make_scorer(s) is s

    def test_vgae_spec(self):
        # CONTRACT: VGAEScorer resolves via class_path without loading the ckpt.
        # (Construction is cheap; scoring is what touches disk.)
        spec = {
            "class_path": "graphids.core.data.curriculum.VGAEScorer",
            "init_args": {"ckpt_path": "/nonexistent.ckpt", "canid_weight": 0.2},
        }
        s = make_scorer(spec)
        assert isinstance(s, VGAEScorer)
        assert s.canid_weight == 0.2

    def test_none_raises(self):
        # REGRESSION: curriculum with no scorer must fail loud, not silently.
        with pytest.raises(ValueError, match="scorer spec"):
            make_scorer(None)

    def test_missing_class_path_raises(self):
        with pytest.raises(ValueError, match="class_path"):
            make_scorer({"init_args": {"seed": 0}})


class TestBucketByScore:
    """Pure index-bucketing math — no dataset, no labels."""

    def test_shape(self):
        scores = torch.arange(20, dtype=torch.float)
        tiers = bucket_by_score(scores, num_tiers=5)
        assert len(tiers) == 5
        assert sum(len(t) for t in tiers) == 20
        assert sorted(i for t in tiers for i in t) == list(range(20))

    def test_sort_order(self):
        # INVARIANT: tier k's minimum score ≥ tier (k-1)'s maximum score.
        torch.manual_seed(0)
        scores = torch.rand(50)
        tiers = bucket_by_score(scores, num_tiers=5)
        for a, b in zip(tiers, tiers[1:]):
            assert scores[a].max() <= scores[b].min()

    def test_uneven_split(self):
        # 23 scores, 5 tiers → ceil(23/5) = 5 per bin, last one smaller.
        scores = torch.arange(23, dtype=torch.float)
        tiers = bucket_by_score(scores, num_tiers=5)
        assert [len(t) for t in tiers] == [5, 5, 5, 5, 3]

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="empty"):
            bucket_by_score(torch.empty(0), num_tiers=3)

    def test_rejects_zero_tiers(self):
        with pytest.raises(ValueError, match="num_tiers"):
            bucket_by_score(torch.rand(10), num_tiers=0)


class TestActiveTierCount:
    """Pure epoch-to-count gating schedule."""

    def test_start_end_bounds(self):
        # CONTRACT: epoch 0 uses start_ratio, epoch max-1 uses end_ratio.
        kw = dict(start_ratio=1.0, end_ratio=10.0, max_epochs=300)
        assert active_tier_count(0, 10, **kw) == 1
        assert active_tier_count(299, 10, **kw) == 10

    def test_clamped_past_max_epochs(self):
        # CONTRACT: running past max_epochs stays at num_tiers, doesn't overflow.
        kw = dict(start_ratio=1.0, end_ratio=10.0, max_epochs=100)
        assert active_tier_count(1_000_000, 10, **kw) == 10

    def test_monotone_non_decreasing(self):
        # INVARIANT: count never drops as epoch advances — tiers only unlock.
        kw = dict(start_ratio=1.0, end_ratio=10.0, max_epochs=300)
        counts = [active_tier_count(e, 10, **kw) for e in range(0, 300, 10)]
        for a, b in zip(counts, counts[1:]):
            assert b >= a

    def test_never_below_one(self):
        # CONTRACT: always at least one tier active, even with zero ratio.
        assert active_tier_count(0, 10, start_ratio=0.0, end_ratio=10.0, max_epochs=100) == 1
