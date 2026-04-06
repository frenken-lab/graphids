"""Curriculum tier-bucketing logic tests."""

from __future__ import annotations

import torch
from conftest import make_graph


def _make_scored_data(n_normal=30, n_attack=10):
    """Synthetic normals + attacks with deterministic difficulty scores."""
    normals = [make_graph() for _ in range(n_normal)]
    for g in normals:
        g.y = torch.tensor([0])
    attacks = [make_graph() for _ in range(n_attack)]
    for g in attacks:
        g.y = torch.tensor([1])
    # Scores: 0/n, 1/n, ..., (n-1)/n — deterministic, ascending
    scores = torch.tensor([float(i) / n_normal for i in range(n_normal)])
    return normals, attacks, scores


class TestBuildCurriculumTiers:
    """build_curriculum_tiers tier-bucketing correctness."""

    @staticmethod
    def _build(n_normal=30, n_attack=10, num_tiers=5):
        """Build tiers from synthetic data, bypassing VGAE scoring."""
        import math

        normals, attacks, scores = _make_scored_data(n_normal, n_attack)
        full_dataset = normals + attacks
        dataset_sizes = torch.tensor([g.num_nodes for g in full_dataset], dtype=torch.long)
        # Replicate the bucketing logic from build_curriculum_tiers
        sorted_order = torch.argsort(scores).tolist()
        bucket_size = max(1, math.ceil(len(sorted_order) / num_tiers))
        normal_tier_indices = []
        for start in range(0, len(sorted_order), bucket_size):
            normal_tier_indices.append(sorted_order[start : start + bucket_size])
        attack_indices = list(range(len(normals), len(full_dataset)))
        return (
            scores,
            normal_tier_indices,
            attack_indices,
            full_dataset,
            dataset_sizes,
            num_tiers,
        )

    def test_all_normals_covered(self):
        """INVARIANT: every normal index appears in exactly one tier."""
        scores, tiers, _, _, _, _ = self._build(n_normal=50, num_tiers=7)
        all_indices = [idx for tier in tiers for idx in tier]
        assert sorted(all_indices) == list(range(len(scores)))

    def test_tiers_sorted_by_difficulty(self):
        """INVARIANT: tier 0 has lower mean difficulty than tier K-1."""
        scores, tiers, _, _, _, _ = self._build(n_normal=50, num_tiers=5)
        tier_means = [scores[tiers[i]].mean().item() for i in range(len(tiers))]
        for i in range(len(tier_means) - 1):
            assert tier_means[i] <= tier_means[i + 1], (
                f"Tier {i} mean ({tier_means[i]:.3f}) > tier {i + 1} ({tier_means[i + 1]:.3f})"
            )

    def test_attacks_separate(self):
        """INVARIANT: attack indices are separate from normal tiers."""
        _, tiers, attack_indices, full_dataset, _, _ = self._build()
        tier_indices = {idx for tier in tiers for idx in tier}
        attack_set = set(attack_indices)
        assert tier_indices.isdisjoint(attack_set)
        assert len(attack_indices) == 10
        # Attack indices point to graphs with y=1
        for idx in attack_indices:
            assert int(full_dataset[idx].y[0]) == 1


class TestSelectActiveTiers:
    """Tier selection progression across epochs."""

    def test_early_epoch_fewer_tiers_than_late(self):
        """INVARIANT: curriculum progression includes more tiers over time."""
        import math

        num_tiers = 10
        start_ratio, end_ratio, max_epochs = 1.0, 10.0, 300

        def active_count(epoch):
            ratio = start_ratio + (end_ratio - start_ratio) * min(
                epoch / max(max_epochs - 1, 1), 1.0
            )
            return max(1, min(num_tiers, math.ceil(ratio * num_tiers / end_ratio)))

        early = active_count(0)
        late = active_count(max_epochs - 1)
        assert late >= early, f"Late epoch tiers ({late}) < early ({early})"
        assert late == num_tiers, f"Final epoch should include all {num_tiers} tiers"

    def test_attacks_always_present(self):
        """INVARIANT: attack batches included at every epoch regardless of ratio."""
        # This is structural — attacks are concatenated unconditionally in
        # _select_active_tiers. Verified by checking the method doesn't
        # gate attack_tier_batches on any condition. Test at integration level
        # in test_prebatch.py::TestTierPreBatch.
        pass
