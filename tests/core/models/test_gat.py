"""GAT: architecture, forward pass, gradient flow, loss variants, checkpoint roundtrip."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.loader import DataLoader

from conftest import EDGE_DIM, IN_CHANNELS, NUM_IDS, make_batch, make_graph, make_variable_batch


class TestGAT:
    @pytest.fixture()
    def model(self):
        from graphids.core.models.gat import GATWithJK
        return GATWithJK(
            num_ids=NUM_IDS, in_channels=IN_CHANNELS, hidden_channels=16,
            out_channels=2, num_layers=2, heads=2, dropout=0.0,
            num_fc_layers=2, embedding_dim=4, conv_type="gatv2",
            edge_dim=EDGE_DIM, pool_aggrs=("mean",), proj_dim=0,
        )

    def test_forward_shape(self, model):
        assert model(make_batch(5)).shape == (5, 2)

    def test_gradient_flow(self, model):
        model(make_batch(3)).sum().backward()
        dead = [n for n, p in model.named_parameters()
                if p.requires_grad and p.grad is None]
        assert not dead, f"No gradient: {dead}"

    def test_variable_size_graphs(self, model):
        batch = make_variable_batch([3, 20])
        assert model(batch).shape == (2, 2)


@pytest.mark.slow
class TestGATFastDevRun:
    def test_gat(self, gat_cfg):
        import pytorch_lightning as pl
        from graphids.core.models.gat import GATModule
        loader = DataLoader([make_graph() for _ in range(16)], batch_size=4)
        trainer = pl.Trainer(fast_dev_run=True, accelerator="cpu", enable_progress_bar=False)
        trainer.fit(GATModule(gat_cfg), loader, loader)

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


class TestGATCheckpointRoundtrip:
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
