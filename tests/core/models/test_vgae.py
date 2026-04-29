"""VGAE + GPS conv: architecture, forward pass, gradient flow, checkpoint roundtrip."""

from __future__ import annotations

import pytest
import torch
from conftest import (
    EDGE_DIM,
    IN_CHANNELS,
    NUM_IDS,
    make_batch,
    make_graph,
    make_variable_batch,
)
from torch_geometric.loader import DataLoader


class TestVGAEConvTypes:
    """Forward pass, gradient flow, and variable-size graphs for each conv type."""

    @pytest.fixture(params=["gatv2", "gps"], ids=["gatv2", "gps"])
    def model_and_conv(self, request):
        from graphids.core.models.autoencoder.vgae import GraphAutoencoderNeighborhood
        from graphids.core.models.id_encoding import LookupIdEncoder

        m = GraphAutoencoderNeighborhood(
            id_encoder=LookupIdEncoder(num_ids=NUM_IDS, embedding_dim=4),
            num_ids=NUM_IDS,
            in_channels=IN_CHANNELS,
            hidden_dims=[32, 16],
            latent_dim=16,
            encoder_heads=2,
            dropout=0.0,
            conv_type=request.param,
            edge_dim=EDGE_DIM,
            proj_dim=0,
        )
        return m, request.param

    def test_forward_shapes(self, model_and_conv):
        model, _ = model_and_conv
        batch = make_batch(3)
        n = batch.x.size(0)
        cont, canid_logits, nbr_logits, z, kl = model(
            batch.x,
            batch.edge_index,
            batch.batch,
            edge_attr=batch.edge_attr,
            node_id=batch.node_id,
        )
        assert cont.shape == (n, IN_CHANNELS)
        assert canid_logits.shape == (n, NUM_IDS)
        assert nbr_logits.shape == (n, NUM_IDS)
        assert z.shape[0] == n
        # kl is per-node (mean over latent dims) so per-graph KL can be
        # scatter-aggregated at test time.
        assert kl.shape == (n,)

    def test_gradient_flow(self, model_and_conv):
        model, conv_type = model_and_conv
        batch = make_batch(2)
        cont, canid_logits, nbr_logits, _z, kl = model(
            batch.x,
            batch.edge_index,
            batch.batch,
            edge_attr=batch.edge_attr,
            node_id=batch.node_id,
        )
        # Gradient-flow verification (not training — raw autograd check).
        # mask_token is a frozen Parameter (requires_grad=False) so it is
        # excluded from the dead-grad sweep below. Sum all four head
        # outputs so canid_classifier + neighborhood_decoder weights also
        # see gradient.
        torch.autograd.backward(cont.sum() + canid_logits.sum() + nbr_logits.sum() + kl.sum())
        # GPS encoder_bns have no gradient by design (batch norm in GPS path).
        # mask_token by design has requires_grad=False.
        exclude = {"encoder_bns"} if conv_type == "gps" else set()
        exclude.add("mask_token")
        dead = [
            n
            for n, p in model.named_parameters()
            if p.requires_grad and p.grad is None and not any(ex in n for ex in exclude)
        ]
        assert not dead, f"No gradient: {dead}"

    def test_variable_size_graphs(self, model_and_conv):
        model, _ = model_and_conv
        batch = make_variable_batch([3, 15])
        out = model(
            batch.x,
            batch.edge_index,
            batch.batch,
            edge_attr=batch.edge_attr,
            node_id=batch.node_id,
        )
        assert out[0].shape[0] == 18


@pytest.mark.slow
class TestVGAEFastDevRun:
    def test_vgae(self, vgae_cfg):
        """INVARIANT: VGAEModule.training_step produces a finite, backpropagable loss."""
        from graphids.core.losses.autoencoder import VGAETaskLoss
        from graphids.core.models.autoencoder.vgae_module import VGAEModule

        module = VGAEModule(
            hidden_dims=vgae_cfg.hidden_dims,
            latent_dim=vgae_cfg.latent_dim,
            heads=vgae_cfg.heads,
            embedding_dim=vgae_cfg.embedding_dim,
            loss_fn=VGAETaskLoss(),
            num_ids=NUM_IDS,
            in_channels=IN_CHANNELS,
            gradient_checkpointing=False,
            compile_model=False,
        )
        module.train()
        batch = make_batch(4)
        loss = module.training_step(batch, 0)
        assert loss is not None and torch.isfinite(loss)
        loss.backward()
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert len(grads) > 0, "No gradients after backward"


class TestVGAECheckpointRoundtrip:
    def test_save_load_produces_identical_output(self, tmp_path):
        from graphids.core.losses.autoencoder import VGAETaskLoss
        from graphids.core.models.autoencoder.vgae_module import VGAEModule

        kwargs = dict(
            hidden_dims=[32, 16],
            latent_dim=16,
            heads=2,
            embedding_dim=4,
            loss_fn=VGAETaskLoss(),
            num_ids=NUM_IDS,
            in_channels=IN_CHANNELS,
            gradient_checkpointing=False,
            compile_model=False,
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
            o1 = m1(batch)
            torch.manual_seed(0)
            o2 = m2(batch)
        torch.testing.assert_close(o1[0], o2[0])
