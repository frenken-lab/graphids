"""VGAE + GPS conv: architecture, forward pass, gradient flow, checkpoint roundtrip."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.loader import DataLoader

from conftest import EDGE_DIM, IN_CHANNELS, NUM_IDS, make_batch, make_graph, make_variable_batch


class TestVGAEConvTypes:
    """Forward pass, gradient flow, and variable-size graphs for each conv type."""

    @pytest.fixture(params=["gatv2", "gps"], ids=["gatv2", "gps"])
    def model_and_conv(self, request):
        from graphids.core.models.vgae import GraphAutoencoderNeighborhood
        m = GraphAutoencoderNeighborhood(
            num_ids=NUM_IDS, in_channels=IN_CHANNELS, hidden_dims=[32, 16],
            latent_dim=16, encoder_heads=2, embedding_dim=4, dropout=0.0,
            conv_type=request.param, edge_dim=EDGE_DIM, proj_dim=0,
        )
        return m, request.param

    def test_forward_shapes(self, model_and_conv):
        model, _ = model_and_conv
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

    def test_gradient_flow(self, model_and_conv):
        model, conv_type = model_and_conv
        batch = make_batch(2)
        cont, canid, nbr, _, kl, _ = model(
            batch.x, batch.edge_index, batch.batch,
            edge_attr=batch.edge_attr, node_id=batch.node_id,
        )
        # Gradient-flow verification (not training — raw autograd check)
        torch.autograd.backward(cont.sum() + canid.sum() + nbr.sum() + kl)
        # GPS encoder_bns have no gradient by design (batch norm in GPS path)
        exclude = {"encoder_bns"} if conv_type == "gps" else set()
        dead = [n for n, p in model.named_parameters()
                if p.grad is None and not any(ex in n for ex in exclude)]
        assert not dead, f"No gradient: {dead}"

    def test_variable_size_graphs(self, model_and_conv):
        model, _ = model_and_conv
        batch = make_variable_batch([3, 15])
        out = model(
            batch.x, batch.edge_index, batch.batch,
            edge_attr=batch.edge_attr, node_id=batch.node_id,
        )
        assert out[0].shape[0] == 18


@pytest.mark.slow
@pytest.mark.slurm
class TestVGAEFastDevRun:
    def test_vgae(self, vgae_cfg):
        import pytorch_lightning as pl
        from graphids.core.models.vgae import VGAEModule
        loader = DataLoader([make_graph() for _ in range(16)], batch_size=4)
        module = VGAEModule(
            hidden_dims=vgae_cfg.hidden_dims, latent_dim=vgae_cfg.latent_dim,
            heads=vgae_cfg.heads, embedding_dim=vgae_cfg.embedding_dim,
            gradient_checkpointing=False, compile_model=False,
            num_ids=NUM_IDS, in_channels=IN_CHANNELS,
        )
        trainer = pl.Trainer(fast_dev_run=True, accelerator="cpu", enable_progress_bar=False)
        trainer.fit(module, loader, loader)


class TestVGAECheckpointRoundtrip:
    def test_save_load_produces_identical_output(self, tmp_path):
        from graphids.core.models.vgae import VGAEModule

        kwargs = dict(
            hidden_dims=[32, 16], latent_dim=16, heads=2,
            embedding_dim=4, gradient_checkpointing=False, compile_model=False,
            num_ids=NUM_IDS, in_channels=IN_CHANNELS,
        )
        m1 = VGAEModule(**kwargs)
        m1.eval()
        torch.save(m1.state_dict(), tmp_path / "v.ckpt")

        m2 = VGAEModule(**kwargs)
        m2.load_state_dict(torch.load(tmp_path / "v.ckpt", weights_only=True))
        m2.eval()

        batch = make_batch(2)
        with torch.no_grad():
            torch.manual_seed(0)
            o1 = m1(batch.x, batch.edge_index, batch.batch,
                     edge_attr=batch.edge_attr, node_id=batch.node_id)
            torch.manual_seed(0)
            o2 = m2(batch.x, batch.edge_index, batch.batch,
                     edge_attr=batch.edge_attr, node_id=batch.node_id)
        torch.testing.assert_close(o1[0], o2[0])
