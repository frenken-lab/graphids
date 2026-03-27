"""VGAE + GPS conv: architecture, forward pass, gradient flow, checkpoint roundtrip."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.loader import DataLoader

from conftest import EDGE_DIM, IN_CHANNELS, NUM_IDS, make_batch, make_graph, make_variable_batch


class TestVGAE:
    @pytest.fixture()
    def model(self):
        from graphids.core.models.vgae import GraphAutoencoderNeighborhood
        return GraphAutoencoderNeighborhood(
            num_ids=NUM_IDS, in_channels=IN_CHANNELS, hidden_dims=[32, 16],
            latent_dim=16, encoder_heads=2, embedding_dim=4, dropout=0.0,
            conv_type="gatv2", edge_dim=EDGE_DIM, proj_dim=0,
        )

    def test_forward_shapes(self, model):
        batch = make_batch(3)
        n = batch.x.size(0)
        cont, canid, nbr, z, kl, mask = model(
            batch.x, batch.edge_index, batch.batch,
            edge_attr=batch.edge_attr, node_id=batch.node_id,
        )
        assert cont.shape == (n, IN_CHANNELS)
        assert canid.shape == (n, NUM_IDS)
        assert z.shape[0] == n
        assert kl.dim() == 0

    def test_gradient_flow(self, model):
        batch = make_batch(2)
        cont, canid, nbr, _, kl, _ = model(
            batch.x, batch.edge_index, batch.batch,
            edge_attr=batch.edge_attr, node_id=batch.node_id,
        )
        (cont.sum() + canid.sum() + nbr.sum() + kl).backward()
        dead = [n for n, p in model.named_parameters() if p.grad is None]
        assert not dead, f"No gradient: {dead}"

    def test_variable_size_graphs(self, model):
        batch = make_variable_batch([3, 15])
        out = model(
            batch.x, batch.edge_index, batch.batch,
            edge_attr=batch.edge_attr, node_id=batch.node_id,
        )
        assert out[0].shape[0] == 18


class TestGPSConv:
    """GPS conv path: _ProjectedGPS + conv_forward GPS branch."""

    @pytest.fixture()
    def model(self):
        from graphids.core.models.vgae import GraphAutoencoderNeighborhood
        return GraphAutoencoderNeighborhood(
            num_ids=NUM_IDS, in_channels=IN_CHANNELS, hidden_dims=[32, 16],
            latent_dim=16, encoder_heads=2, embedding_dim=4, dropout=0.0,
            conv_type="gps", edge_dim=EDGE_DIM, proj_dim=0,
        )

    def test_forward_shapes(self, model):
        batch = make_batch(3)
        n = batch.x.size(0)
        cont, canid, nbr, z, kl, mask = model(
            batch.x, batch.edge_index, batch.batch,
            edge_attr=batch.edge_attr, node_id=batch.node_id,
        )
        assert cont.shape == (n, IN_CHANNELS)
        assert canid.shape == (n, NUM_IDS)
        assert z.shape[0] == n
        assert kl.dim() == 0

    def test_gradient_flow(self, model):
        batch = make_batch(2)
        cont, canid, nbr, _, kl, _ = model(
            batch.x, batch.edge_index, batch.batch,
            edge_attr=batch.edge_attr, node_id=batch.node_id,
        )
        (cont.sum() + canid.sum() + nbr.sum() + kl).backward()
        dead = [n for n, p in model.named_parameters()
                if p.grad is None and "encoder_bns" not in n]
        assert not dead, f"No gradient: {dead}"


@pytest.mark.slow
class TestVGAEFastDevRun:
    def test_vgae(self, vgae_cfg):
        import pytorch_lightning as pl
        from graphids.core.models.vgae import VGAEModule
        loader = DataLoader([make_graph() for _ in range(16)], batch_size=4)
        module = VGAEModule(
            vgae=vgae_cfg.vgae, training=vgae_cfg.training,
            num_ids=NUM_IDS, in_channels=IN_CHANNELS,
        )
        trainer = pl.Trainer(fast_dev_run=True, accelerator="cpu", enable_progress_bar=False)
        trainer.fit(module, loader, loader)


class TestVGAECheckpointRoundtrip:
    def test_vgae(self, vgae_cfg, tmp_path):
        from graphids.core.models.vgae import VGAEModule

        m1 = VGAEModule(
            vgae=vgae_cfg.vgae, training=vgae_cfg.training,
            num_ids=NUM_IDS, in_channels=IN_CHANNELS,
        )
        m1.eval()
        torch.save(m1.state_dict(), tmp_path / "v.ckpt")

        m2 = VGAEModule(
            vgae=vgae_cfg.vgae, training=vgae_cfg.training,
            num_ids=NUM_IDS, in_channels=IN_CHANNELS,
        )
        m2.load_state_dict(torch.load(tmp_path / "v.ckpt", weights_only=True))
        m2.eval()

        batch = make_batch(2)
        with torch.no_grad():
            torch.manual_seed(0)
            o1 = m1(batch)
            torch.manual_seed(0)
            o2 = m2(batch)
        torch.testing.assert_close(o1[0], o2[0])
