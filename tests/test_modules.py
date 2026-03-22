"""Lightning module tests: fast_dev_run, checkpoint roundtrip."""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf, open_dict
from torch_geometric.data import Batch, Data

NUM_IDS = 10
IN_CHANNELS = 31
EDGE_DIM = 12


def _make_graph(num_nodes: int = 8, num_edges: int = 12) -> Data:
    x = torch.rand(num_nodes, IN_CHANNELS)
    x[:, 0] = torch.randint(0, NUM_IDS, (num_nodes,)).float()
    edge_index = torch.stack([
        torch.randint(0, num_nodes, (num_edges,)),
        torch.randint(0, num_nodes, (num_edges,)),
    ])
    edge_attr = torch.rand(num_edges, EDGE_DIM)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor([1]))


def _make_batch(n: int = 4) -> Batch:
    return Batch.from_data_list([_make_graph() for _ in range(n)])


@pytest.fixture()
def tiny_cfg():
    """Minimal config for CPU testing with tiny architecture."""
    from graphids.config import resolve

    cfg = resolve("model_type=vgae", "scale=small")
    with open_dict(cfg):
        cfg.num_ids = NUM_IDS
        cfg.in_channels = IN_CHANNELS
        cfg.device = "cpu"
        cfg.training.max_epochs = 2
        cfg.training.precision = 32
        cfg.training.gradient_checkpointing = False
        cfg.training.compile_model = False
        cfg.training.dynamic_batching = False
        cfg.training.log_every_n_steps = 1
        cfg.training.patience = 100
        cfg.num_workers = 0
    return cfg


@pytest.fixture()
def gat_cfg(tiny_cfg):
    """GAT-specific tiny config."""
    with open_dict(tiny_cfg):
        tiny_cfg.model_type = "gat"
        tiny_cfg.stage = "curriculum"
    return tiny_cfg


# ---------------------------------------------------------------------------
# VGAEModule
# ---------------------------------------------------------------------------


class TestVGAEModule:
    def test_forward(self, tiny_cfg):
        """VGAEModule forward produces expected tuple."""
        from graphids.pipeline.stages.modules import VGAEModule

        module = VGAEModule(tiny_cfg)
        batch = _make_batch(3)
        module.eval()
        with torch.no_grad():
            out = module(batch)
        assert len(out) == 6  # cont, canid, nbr, z, kl, mask

    def test_training_step(self, tiny_cfg):
        """training_step returns a scalar loss."""
        from graphids.pipeline.stages.modules import VGAEModule

        module = VGAEModule(tiny_cfg)
        module.train()
        batch = _make_batch(3)
        loss = module.training_step(batch, 0)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_validation_step(self, tiny_cfg):
        """validation_step runs without error."""
        from graphids.pipeline.stages.modules import VGAEModule

        module = VGAEModule(tiny_cfg)
        module.eval()
        batch = _make_batch(3)
        module.validation_step(batch, 0)

    def test_checkpoint_roundtrip(self, tiny_cfg, tmp_path):
        """Save and load checkpoint, verify output reproducibility."""
        from graphids.pipeline.stages.modules import VGAEModule

        module = VGAEModule(tiny_cfg)
        module.eval()

        # Save
        ckpt_path = tmp_path / "vgae.ckpt"
        torch.save(module.state_dict(), ckpt_path)

        # Load into fresh module
        module2 = VGAEModule(tiny_cfg)
        module2.load_state_dict(torch.load(ckpt_path, weights_only=True))
        module2.eval()

        # Compare with fixed seed (VGAE has stochastic reparameterization)
        batch = _make_batch(2)
        with torch.no_grad():
            torch.manual_seed(0)
            out1 = module(batch)
            torch.manual_seed(0)
            out2 = module2(batch)
        torch.testing.assert_close(out1[0], out2[0])  # cont_out
        torch.testing.assert_close(out1[3], out2[3])  # z

    def test_configure_optimizers(self, tiny_cfg):
        """configure_optimizers returns optimizer (or dict with scheduler)."""
        from graphids.pipeline.stages.modules import VGAEModule

        module = VGAEModule(tiny_cfg)
        result = module.configure_optimizers()
        # Without scheduler: returns bare optimizer. With scheduler: returns dict.
        assert isinstance(result, (torch.optim.Optimizer, dict))


# ---------------------------------------------------------------------------
# GATModule
# ---------------------------------------------------------------------------


class TestGATModule:
    def test_forward(self, gat_cfg):
        """GATModule forward produces logits."""
        from graphids.pipeline.stages.modules import GATModule

        module = GATModule(gat_cfg)
        module.eval()
        batch = _make_batch(3)
        with torch.no_grad():
            logits = module(batch)
        assert logits.shape == (3, 2)

    def test_training_step(self, gat_cfg):
        """training_step returns scalar loss."""
        from graphids.pipeline.stages.modules import GATModule

        module = GATModule(gat_cfg)
        module.train()
        batch = _make_batch(3)
        loss = module.training_step(batch, 0)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_checkpoint_roundtrip(self, gat_cfg, tmp_path):
        """Save and load GAT checkpoint, verify output reproducibility."""
        from graphids.pipeline.stages.modules import GATModule

        module = GATModule(gat_cfg)
        module.eval()

        ckpt_path = tmp_path / "gat.ckpt"
        torch.save(module.state_dict(), ckpt_path)

        module2 = GATModule(gat_cfg)
        module2.load_state_dict(torch.load(ckpt_path, weights_only=True))
        module2.eval()

        batch = _make_batch(2)
        with torch.no_grad():
            logits1 = module(batch)
            logits2 = module2(batch)
        torch.testing.assert_close(logits1, logits2)

    def test_loss_fn_variants(self, gat_cfg):
        """All three loss functions (ce, weighted_ce, focal) produce finite loss."""
        from graphids.pipeline.stages.modules import GATModule

        batch = _make_batch(4)
        for loss_name in ("ce", "weighted_ce", "focal"):
            with open_dict(gat_cfg):
                gat_cfg.training.loss_fn = loss_name
            module = GATModule(gat_cfg)
            module.train()
            loss = module.training_step(batch, 0)
            assert torch.isfinite(loss), f"Non-finite loss with {loss_name}"

    def test_test_step(self, gat_cfg):
        """test_step updates metrics without error."""
        from graphids.pipeline.stages.modules import GATModule

        module = GATModule(gat_cfg)
        module.eval()
        batch = _make_batch(3)
        # test_step should not raise
        module.test_step(batch, 0)


# ---------------------------------------------------------------------------
# Fast dev run (integration)
# ---------------------------------------------------------------------------


class TestFastDevRun:
    def _make_dataset(self, n: int = 16):
        """Create a minimal list of Data objects for a DataLoader."""
        return [_make_graph() for _ in range(n)]

    def test_vgae_fast_dev_run(self, tiny_cfg):
        """VGAEModule completes fast_dev_run with synthetic data."""
        import pytorch_lightning as pl
        from torch_geometric.loader import DataLoader

        from graphids.pipeline.stages.modules import VGAEModule

        module = VGAEModule(tiny_cfg)
        ds = self._make_dataset()
        loader = DataLoader(ds, batch_size=4)

        trainer = pl.Trainer(
            fast_dev_run=True, accelerator="cpu", devices=1, enable_progress_bar=False,
        )
        trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)

    def test_gat_fast_dev_run(self, gat_cfg):
        """GATModule completes fast_dev_run with synthetic data."""
        import pytorch_lightning as pl
        from torch_geometric.loader import DataLoader

        from graphids.pipeline.stages.modules import GATModule

        module = GATModule(gat_cfg)
        ds = self._make_dataset()
        loader = DataLoader(ds, batch_size=4)

        trainer = pl.Trainer(
            fast_dev_run=True, accelerator="cpu", devices=1, enable_progress_bar=False,
        )
        trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)
