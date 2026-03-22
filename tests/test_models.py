"""Model tests: forward shape, gradient flow, variable-size graphs."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Batch

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
            batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr,
        )
        assert cont.shape == (n, IN_CHANNELS - 1)
        assert canid.shape == (n, NUM_IDS)
        assert z.shape[0] == n
        assert kl.dim() == 0

    def test_gradient_flow(self, model):
        batch = make_batch(2)
        cont, canid, _, _, kl, _ = model(
            batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr,
        )
        (cont.sum() + canid.sum() + kl).backward()
        dead = [n for n, p in model.named_parameters() if p.grad is None]
        assert not dead, f"No gradient: {dead}"

    def test_variable_size_graphs(self, model):
        batch = make_variable_batch([3, 15])
        out = model(batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr)
        assert out[0].shape[0] == 18


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
