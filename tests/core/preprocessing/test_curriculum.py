"""CurriculumDataModule resampling logic tests."""

from __future__ import annotations

import torch

from conftest import make_graph


class TestCurriculumDataModule:
    """CurriculumDataModule resampling logic."""

    @staticmethod
    def _make_curriculum_data(n_normal=20, n_attack=10):
        normals = [make_graph() for _ in range(n_normal)]
        for g in normals:
            g.y = torch.tensor([0])
        attacks = [make_graph() for _ in range(n_attack)]
        for g in attacks:
            g.y = torch.tensor([1])
        scores = [float(i) / n_normal for i in range(n_normal)]
        return normals, attacks, scores

    def test_train_dataloader_returns_loader(self, gat_cfg):
        from graphids.core.preprocessing.curriculum import CurriculumDataModule
        normals, attacks, scores = self._make_curriculum_data()
        val_data = [make_graph() for _ in range(5)]
        dm = CurriculumDataModule(normals, attacks, scores, val_data, gat_cfg)
        loader = dm.train_dataloader()
        assert loader is not None
        batch = next(iter(loader))
        assert hasattr(batch, "x")
        assert hasattr(batch, "y")

    def test_epoch_counter_increments(self, gat_cfg):
        from graphids.core.preprocessing.curriculum import CurriculumDataModule
        normals, attacks, scores = self._make_curriculum_data()
        dm = CurriculumDataModule(normals, attacks, scores, [make_graph()], gat_cfg)
        assert dm._current_epoch == 0
        dm.train_dataloader()
        assert dm._current_epoch == 1
        dm.train_dataloader()
        assert dm._current_epoch == 2

    def test_val_dataloader_is_fixed(self, gat_cfg):
        from graphids.core.preprocessing.curriculum import CurriculumDataModule
        normals, attacks, scores = self._make_curriculum_data()
        val_data = [make_graph() for _ in range(8)]
        dm = CurriculumDataModule(normals, attacks, scores, val_data, gat_cfg)
        vl = dm.val_dataloader()
        assert vl is not None
