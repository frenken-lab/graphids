"""Lightning module tests: training_step, checkpoint roundtrip, fast_dev_run."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.loader import DataLoader

from conftest import make_batch, make_graph


@pytest.mark.slow
class TestFastDevRun:
    """Integration: Lightning Trainer.fit with synthetic data."""

    def _loader(self, n: int = 16):
        return DataLoader([make_graph() for _ in range(n)], batch_size=4)

    def test_vgae(self, vgae_cfg):
        import pytorch_lightning as pl
        from graphids.core.models.vgae import VGAEModule
        trainer = pl.Trainer(fast_dev_run=True, accelerator="cpu", enable_progress_bar=False)
        trainer.fit(VGAEModule(vgae_cfg), self._loader(), self._loader())

    def test_gat(self, gat_cfg):
        import pytorch_lightning as pl
        from graphids.core.models.gat import GATModule
        trainer = pl.Trainer(fast_dev_run=True, accelerator="cpu", enable_progress_bar=False)
        trainer.fit(GATModule(gat_cfg), self._loader(), self._loader())

    def test_gat_loss_variants(self, gat_cfg):
        """All loss functions produce finite loss in training_step."""
        from omegaconf import open_dict
        from graphids.core.models.gat import GATModule
        for loss_fn in ("ce", "weighted_ce", "focal"):
            with open_dict(gat_cfg):
                gat_cfg.training.loss_fn = loss_fn
            module = GATModule(gat_cfg)
            module.train()
            loss = module.training_step(make_batch(4), 0)
            assert torch.isfinite(loss), f"{loss_fn} produced non-finite loss"


class TestFusionBaselineTestStep:
    """Bugs 1+5: MLPFusionModule and WeightedAvgModule must have test_step + test_metrics."""

    @staticmethod
    def _make_fusion_batch(n: int = 32, state_dim: int = 15):
        states = torch.rand(n, state_dim)
        labels = torch.randint(0, 2, (n,))
        return states, labels

    def test_mlp_has_test_metrics(self):
        from graphids.core.models.fusion_baselines import MLPFusionModule
        module = MLPFusionModule(state_dim=15)
        assert hasattr(module, "test_metrics"), "MLPFusionModule missing test_metrics"

    def test_mlp_test_step_updates_metrics(self):
        from graphids.core.models.fusion_baselines import MLPFusionModule
        module = MLPFusionModule(state_dim=15)
        module.eval()
        module.on_test_epoch_start()
        module.test_step(self._make_fusion_batch(), 0)
        result = module.test_metrics.compute()
        assert "accuracy" in result
        assert "f1" in result
        assert all(0.0 <= v.item() <= 1.0 for v in result.values())

    def test_weighted_avg_has_test_metrics(self):
        from graphids.core.models.fusion_baselines import WeightedAvgModule
        module = WeightedAvgModule()
        assert hasattr(module, "test_metrics"), "WeightedAvgModule missing test_metrics"

    def test_weighted_avg_test_step_updates_metrics(self):
        from graphids.core.models.fusion_baselines import WeightedAvgModule
        module = WeightedAvgModule()
        module.eval()
        module.on_test_epoch_start()
        module.test_step(self._make_fusion_batch(), 0)
        result = module.test_metrics.compute()
        assert "accuracy" in result
        assert "f1" in result

    @pytest.mark.slow
    def test_mlp_test_step_via_lightning_trainer(self):
        """End-to-end: trainer.test() → test_metrics.compute() without AttributeError."""
        import pytorch_lightning as pl
        from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset
        from graphids.core.models.fusion_baselines import MLPFusionModule
        module = MLPFusionModule(state_dim=15)
        states, labels = self._make_fusion_batch(64)
        loader = TorchDataLoader(TensorDataset(states, labels), batch_size=16)
        trainer = pl.Trainer(accelerator="cpu", logger=False, enable_checkpointing=False, enable_progress_bar=False)
        trainer.test(module, dataloaders=loader, verbose=False)
        result = module.test_metrics.compute()
        assert "accuracy" in result

    @pytest.mark.slow
    def test_weighted_avg_test_step_via_lightning_trainer(self):
        """End-to-end: trainer.test() → test_metrics.compute() without AttributeError."""
        import pytorch_lightning as pl
        from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset
        from graphids.core.models.fusion_baselines import WeightedAvgModule
        module = WeightedAvgModule()
        states, labels = self._make_fusion_batch(64)
        loader = TorchDataLoader(TensorDataset(states, labels), batch_size=16)
        trainer = pl.Trainer(accelerator="cpu", logger=False, enable_checkpointing=False, enable_progress_bar=False)
        trainer.test(module, dataloaders=loader, verbose=False)
        result = module.test_metrics.compute()
        assert "accuracy" in result

    def test_mlp_test_metrics_reset_between_scenarios(self):
        """Bug 5: module.test_metrics.reset() must not crash between test scenarios."""
        from graphids.core.models.fusion_baselines import MLPFusionModule
        module = MLPFusionModule(state_dim=15)
        module.eval()

        module.on_test_epoch_start()
        module.test_step(self._make_fusion_batch(32), 0)
        result_1 = module.test_metrics.compute()

        module.test_metrics.reset()

        module.test_step(self._make_fusion_batch(16), 0)
        result_2 = module.test_metrics.compute()

        assert "accuracy" in result_1
        assert "accuracy" in result_2

    def test_weighted_avg_test_metrics_reset_between_scenarios(self):
        """Bug 5: WeightedAvgModule.test_metrics.reset() works between test scenarios."""
        from graphids.core.models.fusion_baselines import WeightedAvgModule
        module = WeightedAvgModule()
        module.eval()

        module.on_test_epoch_start()
        module.test_step(self._make_fusion_batch(32), 0)
        result_1 = module.test_metrics.compute()

        module.test_metrics.reset()

        module.test_step(self._make_fusion_batch(16), 0)
        result_2 = module.test_metrics.compute()

        assert "accuracy" in result_1
        assert "accuracy" in result_2


@pytest.mark.slow
class TestRestoreBestWeights:
    """Bug 2: _restore_best_weights must be called before saving fusion checkpoints."""

    def test_restore_works_with_mlp(self, tmp_path):
        """_restore_best_weights runs without error on MLPFusionModule."""
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks import ModelCheckpoint
        from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset
        from graphids.core.models.fusion_baselines import MLPFusionModule
        from graphids.pipeline.stages.fusion import _restore_best_weights

        torch.manual_seed(0)
        module = MLPFusionModule(state_dim=15, lr=0.1)
        states = torch.rand(128, 15)
        labels = torch.randint(0, 2, (128,))
        train_dl = TorchDataLoader(TensorDataset(states, labels.float()), batch_size=32)
        val_dl = TorchDataLoader(TensorDataset(states[:32], labels[:32].float()), batch_size=32)

        ckpt_cb = ModelCheckpoint(dirpath=str(tmp_path), monitor="val_loss", mode="min", save_top_k=1)
        trainer = pl.Trainer(
            max_epochs=3, accelerator="cpu", callbacks=[ckpt_cb],
            logger=False, enable_progress_bar=False, default_root_dir=str(tmp_path),
        )
        trainer.fit(module, train_dl, val_dl)

        _restore_best_weights(trainer, module)

        # Module must be functional after restore
        logits = module(torch.rand(4, 15))
        assert logits.shape == (4,)
        assert torch.isfinite(logits).all()

    def test_restore_works_with_weighted_avg(self, tmp_path):
        """_restore_best_weights runs without error on WeightedAvgModule."""
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks import ModelCheckpoint
        from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset
        from graphids.core.models.fusion_baselines import WeightedAvgModule
        from graphids.pipeline.stages.fusion import _restore_best_weights

        torch.manual_seed(0)
        module = WeightedAvgModule(lr=0.5)
        states = torch.rand(128, 15)
        labels = torch.randint(0, 2, (128,))
        train_dl = TorchDataLoader(TensorDataset(states, labels.float()), batch_size=32)
        val_dl = TorchDataLoader(TensorDataset(states[:32], labels[:32].float()), batch_size=32)

        ckpt_cb = ModelCheckpoint(dirpath=str(tmp_path), monitor="val_loss", mode="min", save_top_k=1)
        trainer = pl.Trainer(
            max_epochs=3, accelerator="cpu", callbacks=[ckpt_cb],
            logger=False, enable_progress_bar=False, default_root_dir=str(tmp_path),
        )
        trainer.fit(module, train_dl, val_dl)

        _restore_best_weights(trainer, module)

        saved = module.state_dict_for_save()
        assert "weight" in saved
        assert "alpha" in saved


class TestCheckpointRoundtrip:
    def test_vgae(self, vgae_cfg, tmp_path):
        from graphids.core.models.vgae import VGAEModule

        m1 = VGAEModule(vgae_cfg)
        m1.eval()
        torch.save(m1.state_dict(), tmp_path / "v.ckpt")

        m2 = VGAEModule(vgae_cfg)
        m2.load_state_dict(torch.load(tmp_path / "v.ckpt", weights_only=True))
        m2.eval()

        batch = make_batch(2)
        with torch.no_grad():
            torch.manual_seed(0)
            o1 = m1(batch)
            torch.manual_seed(0)
            o2 = m2(batch)
        torch.testing.assert_close(o1[0], o2[0])

    def test_gat(self, gat_cfg, tmp_path):
        from graphids.core.models.gat import GATModule

        m1 = GATModule(gat_cfg)
        m1.eval()
        torch.save(m1.state_dict(), tmp_path / "g.ckpt")

        m2 = GATModule(gat_cfg)
        m2.load_state_dict(torch.load(tmp_path / "g.ckpt", weights_only=True))
        m2.eval()

        batch = make_batch(2)
        with torch.no_grad():
            torch.testing.assert_close(m1(batch), m2(batch))


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


class TestTemporalStage:
    """Temporal model + Lightning module + dataset — future research."""

    @staticmethod
    def _make_sequence(seq_len=4, num_nodes=8):
        return [make_graph(num_nodes=num_nodes) for _ in range(seq_len)]

    def test_temporal_classifier_forward(self):
        from graphids.core.models.temporal import TemporalGraphClassifier
        from graphids.core.models.gat import GATWithJK
        from conftest import NUM_IDS, IN_CHANNELS, EDGE_DIM

        spatial = GATWithJK(
            num_ids=NUM_IDS, in_channels=IN_CHANNELS, hidden_channels=16,
            out_channels=2, num_layers=2, heads=2, dropout=0.0,
            num_fc_layers=2, embedding_dim=4, conv_type="gatv2",
            edge_dim=EDGE_DIM, pool_aggrs=("mean",), proj_dim=0,
        )
        model = TemporalGraphClassifier(
            spatial_encoder=spatial, spatial_dim=32, num_classes=2,
            temporal_hidden=16, temporal_heads=2, temporal_layers=1,
            max_seq_len=4,
        )
        sequences = [self._make_sequence() for _ in range(3)]
        model.eval()
        with torch.no_grad():
            logits = model(sequences)
        assert logits.shape == (3, 2)

    def test_collate_produces_correct_shapes(self):
        from graphids.pipeline.stages.temporal import collate_temporal
        batch_data = [
            ([make_graph() for _ in range(4)], 0),
            ([make_graph() for _ in range(4)], 1),
        ]
        graph_sequences, labels = collate_temporal(batch_data)
        assert len(graph_sequences) == 2
        assert len(graph_sequences[0]) == 4
        assert labels.shape == (2,)

    def test_temporal_lightning_module_has_test_metrics(self):
        from graphids.pipeline.stages.temporal import TemporalLightningModule
        assert hasattr(TemporalLightningModule, "test_step")
