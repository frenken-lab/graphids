"""GAT: architecture, forward pass, gradient flow, loss variants, checkpoint roundtrip."""

from __future__ import annotations

import pytest
import torch
from conftest import EDGE_DIM, IN_CHANNELS, NUM_IDS, make_batch, make_graph, make_variable_batch
from torch_geometric.loader import DataLoader


class TestGAT:
    @pytest.fixture()
    def model(self):
        from graphids.core.models.supervised.gat import GATWithJK

        return GATWithJK(
            num_ids=NUM_IDS,
            in_channels=IN_CHANNELS,
            hidden_channels=16,
            out_channels=2,
            num_layers=2,
            heads=2,
            dropout=0.0,
            num_fc_layers=2,
            embedding_dim=4,
            conv_type="gatv2",
            edge_dim=EDGE_DIM,
            pool_aggrs=("mean",),
            proj_dim=0,
        )

    def test_forward_shape(self, model):
        assert model(make_batch(5)).shape == (5, 2)

    def test_gradient_flow(self, model):
        model(make_batch(3)).sum().backward()
        dead = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
        assert not dead, f"No gradient: {dead}"

    def test_variable_size_graphs(self, model):
        batch = make_variable_batch([3, 20])
        assert model(batch).shape == (2, 2)


@pytest.mark.slow
class TestGATFastDevRun:
    @staticmethod
    def _make_module(cfg):
        from graphids.core.models.supervised.gat_module import GATModule

        return GATModule(
            hidden=cfg.hidden,
            layers=cfg.layers,
            heads=cfg.heads,
            fc_layers=cfg.fc_layers,
            embedding_dim=cfg.embedding_dim,
            loss_fn=cfg.loss_fn,
            focal_gamma=cfg.focal_gamma,
            loss_weight=cfg.loss_weight,
            gradient_checkpointing=False,
            compile_model=False,
            num_ids=NUM_IDS,
            in_channels=IN_CHANNELS,
        )

    def test_gat(self, gat_cfg):
        import pytorch_lightning as pl

        loader = DataLoader([make_graph() for _ in range(16)], batch_size=4)
        trainer = pl.Trainer(fast_dev_run=True, accelerator="cpu", enable_progress_bar=False)
        trainer.fit(self._make_module(gat_cfg), loader, loader)

    @pytest.mark.parametrize("loss_fn", ["ce", "weighted_ce", "focal"])
    def test_gat_loss_variant_produces_finite_loss(self, gat_cfg, loss_fn):
        import copy

        cfg = copy.deepcopy(gat_cfg)
        cfg.loss_fn = loss_fn
        module = self._make_module(cfg)
        module.train()
        loss = module.training_step(make_batch(4), 0)
        assert torch.isfinite(loss), f"{loss_fn} produced non-finite loss"


class TestGATCheckpointRoundtrip:
    def test_gat(self, gat_cfg, tmp_path):
        from graphids.core.models.supervised.gat_module import GATModule

        def _mk():
            return GATModule(
                hidden=gat_cfg.hidden,
                layers=gat_cfg.layers,
                heads=gat_cfg.heads,
                fc_layers=gat_cfg.fc_layers,
                embedding_dim=gat_cfg.embedding_dim,
                gradient_checkpointing=False,
                compile_model=False,
                num_ids=NUM_IDS,
                in_channels=IN_CHANNELS,
            )

        m1 = _mk()
        m1.eval()
        torch.save(m1.state_dict(), tmp_path / "g.ckpt")

        m2 = _mk()
        m2.load_state_dict(torch.load(tmp_path / "g.ckpt", weights_only=True))
        m2.eval()

        batch = make_batch(2)
        with torch.no_grad():
            torch.testing.assert_close(m1(batch), m2(batch))
