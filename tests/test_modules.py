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
        from graphids.pipeline.stages.modules import VGAEModule
        trainer = pl.Trainer(fast_dev_run=True, accelerator="cpu", enable_progress_bar=False)
        trainer.fit(VGAEModule(vgae_cfg), self._loader(), self._loader())

    def test_gat(self, gat_cfg):
        import pytorch_lightning as pl
        from graphids.pipeline.stages.modules import GATModule
        trainer = pl.Trainer(fast_dev_run=True, accelerator="cpu", enable_progress_bar=False)
        trainer.fit(GATModule(gat_cfg), self._loader(), self._loader())

    def test_gat_loss_variants(self, gat_cfg):
        """All loss functions produce finite loss in training_step."""
        from omegaconf import open_dict
        from graphids.pipeline.stages.modules import GATModule
        for loss_fn in ("ce", "weighted_ce", "focal"):
            with open_dict(gat_cfg):
                gat_cfg.training.loss_fn = loss_fn
            module = GATModule(gat_cfg)
            module.train()
            loss = module.training_step(make_batch(4), 0)
            assert torch.isfinite(loss), f"{loss_fn} produced non-finite loss"


class TestCheckpointRoundtrip:
    def test_vgae(self, vgae_cfg, tmp_path):
        from graphids.pipeline.stages.modules import VGAEModule

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
        from graphids.pipeline.stages.modules import GATModule

        m1 = GATModule(gat_cfg)
        m1.eval()
        torch.save(m1.state_dict(), tmp_path / "g.ckpt")

        m2 = GATModule(gat_cfg)
        m2.load_state_dict(torch.load(tmp_path / "g.ckpt", weights_only=True))
        m2.eval()

        batch = make_batch(2)
        with torch.no_grad():
            torch.testing.assert_close(m1(batch), m2(batch))
