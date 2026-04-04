"""CurriculumSampler resampling logic tests."""

from __future__ import annotations

import torch

from conftest import make_graph


class TestCurriculumSampler:
    """CurriculumSampler resampling logic (tested directly, not via DataModule)."""

    @staticmethod
    def _make_data_and_sampler(n_normal=20, n_attack=10):
        from graphids.core.preprocessing.sampler import CurriculumSampler

        normals = [make_graph() for _ in range(n_normal)]
        for g in normals:
            g.y = torch.tensor([0])
        attacks = [make_graph() for _ in range(n_attack)]
        for g in attacks:
            g.y = torch.tensor([1])
        scores = [float(i) / n_normal for i in range(n_normal)]
        full_dataset = normals + attacks
        normal_indices = list(range(len(normals)))
        attack_indices = list(range(len(normals), len(full_dataset)))
        dataset_sizes = torch.tensor(
            [g.num_nodes for g in full_dataset], dtype=torch.long,
        )

        sampler = CurriculumSampler(
            full_dataset, normal_indices, attack_indices, scores,
            batch_size=32, max_epochs=10,
            curriculum_start_ratio=0.3, curriculum_end_ratio=1.0,
            difficulty_percentile=75.0,
            dataset_sizes=dataset_sizes,
        )
        return sampler, full_dataset

    def test_sampler_yields_batches(self):
        sampler, _ = self._make_data_and_sampler()
        batches = list(sampler)
        assert len(batches) > 0
        # Each batch is a list of indices
        assert all(isinstance(b, list) for b in batches)

    def test_set_epoch_changes_active_indices(self):
        sampler, _ = self._make_data_and_sampler()
        sampler.set_epoch(0)
        len_at_0 = len(sampler._active_indices)
        sampler.set_epoch(5)
        len_at_5 = len(sampler._active_indices)
        # Curriculum progression should change the number of active indices
        assert len_at_5 != len_at_0, (
            f"set_epoch(5) did not change active indices "
            f"(both {len_at_0}) — curriculum may be a no-op"
        )

    def test_late_epoch_has_more_active_indices_than_early(self):
        sampler, _ = self._make_data_and_sampler()
        sampler.set_epoch(0)
        len_early = len(sampler._active_indices)
        sampler.set_epoch(9)
        len_late = len(sampler._active_indices)
        # With start_ratio=0.3 -> end_ratio=1.0, late epochs include more normals
        assert len_late >= len_early, (
            f"Active indices at epoch 9 ({len_late}) should be >= epoch 0 ({len_early})"
        )

    def test_set_node_budget_finalizes_inner_sampler(self):
        """Deferred budget path: sampler built with max_num_nodes=None has no inner
        sampler; set_node_budget() installs the budget and rebuilds it.
        CurriculumDataModule.train_dataloader() relies on this after the model
        is placed on GPU (VRAM probe requires a live CUDA model)."""
        sampler, full_dataset = self._make_data_and_sampler()
        # Default fixture uses max_num_nodes=None
        assert sampler.max_num_nodes is None
        assert sampler._inner is None
        # Batches still work via the fallback iterator (fixed batch_size)
        fallback_batches = list(sampler)
        assert len(fallback_batches) > 0

        # Simulate post-GPU placement: budget known, finalize.
        max_nodes = max(int(sampler.dataset_sizes.max().item()) * 4, 64)
        sampler.set_node_budget(max_num_nodes=max_nodes, mean_nodes=8.0)
        assert sampler.max_num_nodes == max_nodes
        assert sampler._inner is not None
        # Inner batches now respect the node budget and yield full-dataset indices
        real_batches = list(sampler)
        assert len(real_batches) > 0
        flat = [i for batch in real_batches for i in batch]
        assert max(flat) < len(full_dataset), (
            "sampler must yield full-dataset positions, not subset-local"
        )
