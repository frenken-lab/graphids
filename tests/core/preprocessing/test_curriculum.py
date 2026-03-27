"""CurriculumSampler resampling logic tests."""

from __future__ import annotations

import types

import torch

from conftest import make_graph


class TestCurriculumSampler:
    """CurriculumSampler resampling logic (tested directly, not via DataModule)."""

    @staticmethod
    def _make_data_and_sampler(n_normal=20, n_attack=10):
        from graphids.core.preprocessing.curriculum import CurriculumSampler

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

        cfg = types.SimpleNamespace(training=types.SimpleNamespace(
            batch_size=32, max_epochs=10,
            curriculum_start_ratio=1.0, curriculum_end_ratio=10.0,
            difficulty_percentile=75.0, dynamic_batching=False,
        ))
        sampler = CurriculumSampler(
            full_dataset, normal_indices, attack_indices, scores, cfg,
        )
        return sampler, full_dataset

    def test_sampler_yields_batches(self):
        sampler, _ = self._make_data_and_sampler()
        batches = list(sampler)
        assert len(batches) > 0
        # Each batch is a list of indices
        assert all(isinstance(b, list) for b in batches)

    def test_set_epoch_updates_active_indices(self):
        sampler, _ = self._make_data_and_sampler()
        initial_len = len(sampler._active_indices)
        sampler.set_epoch(5)
        # After progression, active indices may change
        assert len(sampler._active_indices) > 0

    def test_epoch_counter_via_set_epoch(self):
        sampler, _ = self._make_data_and_sampler()
        sampler.set_epoch(0)
        sampler.set_epoch(1)
        sampler.set_epoch(2)
        # No crash — sampler handles multiple epoch transitions
        assert len(list(sampler)) > 0
