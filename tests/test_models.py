"""Model tests: forward shape, gradient flow, parameter update, variable-size graphs."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Batch, Data

# ---------------------------------------------------------------------------
# Fixtures: tiny architectures for CPU tests
# ---------------------------------------------------------------------------

NUM_IDS = 10
IN_CHANNELS = 31  # CAN ID col + 30 continuous features
EDGE_DIM = 12


def _make_graph(num_nodes: int = 8, num_edges: int = 12) -> Data:
    """Synthetic single graph with valid CAN-bus-like features."""
    x = torch.rand(num_nodes, IN_CHANNELS)
    x[:, 0] = torch.randint(0, NUM_IDS, (num_nodes,)).float()
    edge_index = torch.stack([
        torch.randint(0, num_nodes, (num_edges,)),
        torch.randint(0, num_nodes, (num_edges,)),
    ])
    edge_attr = torch.rand(num_edges, EDGE_DIM)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor([1]))


def _make_batch(num_graphs: int = 4, min_nodes: int = 5, max_nodes: int = 12) -> Batch:
    """Batch of variable-size graphs."""
    graphs = []
    for _ in range(num_graphs):
        n = torch.randint(min_nodes, max_nodes + 1, (1,)).item()
        e = n * 2
        graphs.append(_make_graph(num_nodes=n, num_edges=e))
    return Batch.from_data_list(graphs)


# ---------------------------------------------------------------------------
# VGAE: GraphAutoencoderNeighborhood
# ---------------------------------------------------------------------------


class TestVGAE:
    @pytest.fixture()
    def model(self):
        from graphids.core.models.vgae import GraphAutoencoderNeighborhood

        return GraphAutoencoderNeighborhood(
            num_ids=NUM_IDS,
            in_channels=IN_CHANNELS,
            hidden_dims=[32, 16],
            latent_dim=16,
            encoder_heads=2,
            embedding_dim=4,
            dropout=0.0,
            conv_type="gatv2",
            edge_dim=EDGE_DIM,
            proj_dim=0,
        )

    def test_forward_shapes(self, model):
        """Forward returns correct output shapes."""
        batch = _make_batch(num_graphs=3)
        total_nodes = batch.x.size(0)

        cont_out, canid_logits, nbr_logits, z, kl_loss, mask = model(
            batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr,
        )
        assert cont_out.shape == (total_nodes, IN_CHANNELS - 1)
        assert canid_logits.shape == (total_nodes, NUM_IDS)
        assert nbr_logits.shape == (total_nodes, NUM_IDS)
        assert z.shape == (total_nodes, 16)
        assert kl_loss.dim() == 0  # scalar
        assert mask is None  # mask_ratio=0 by default

    def test_forward_with_masking(self, model):
        """Masking produces a bool mask tensor."""
        model.train()
        batch = _make_batch(num_graphs=2)
        total_nodes = batch.x.size(0)

        _, _, _, _, _, mask = model(
            batch.x, batch.edge_index, batch.batch,
            edge_attr=batch.edge_attr, mask_ratio=0.5,
        )
        assert mask is not None
        assert mask.shape == (total_nodes, IN_CHANNELS - 1)
        assert mask.dtype == torch.bool

    def test_gradient_flow(self, model):
        """Gradients flow to all parameters."""
        batch = _make_batch(num_graphs=2)
        cont_out, canid_logits, nbr_logits, z, kl_loss, _ = model(
            batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr,
        )
        # Use all outputs so every head gets gradients
        loss = cont_out.sum() + canid_logits.sum() + nbr_logits.sum() + kl_loss
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"

    def test_parameter_update(self, model):
        """One optimizer step changes parameters."""
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        batch = _make_batch(num_graphs=2)

        old_params = {n: p.clone() for n, p in model.named_parameters()}
        cont_out, _, _, _, kl_loss, _ = model(
            batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr,
        )
        loss = cont_out.sum() + kl_loss
        loss.backward()
        opt.step()

        changed = sum(
            1 for n, p in model.named_parameters() if not torch.equal(p, old_params[n])
        )
        assert changed > 0, "No parameters changed after optimizer step"

    def test_from_config(self):
        """from_config constructs a valid model from resolved config."""
        from graphids.config import resolve
        from graphids.core.models.vgae import GraphAutoencoderNeighborhood

        cfg = resolve("model_type=vgae", "scale=small")
        model = GraphAutoencoderNeighborhood.from_config(cfg, NUM_IDS, IN_CHANNELS)
        batch = _make_batch(num_graphs=2)
        out = model(batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr)
        assert len(out) == 6  # (cont, canid, nbr, z, kl, mask)

    def test_variable_size_graphs(self, model):
        """Model handles batches with differently-sized graphs."""
        g1 = _make_graph(num_nodes=3, num_edges=4)
        g2 = _make_graph(num_nodes=15, num_edges=30)
        batch = Batch.from_data_list([g1, g2])

        cont_out, _, _, z, _, _ = model(
            batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr,
        )
        assert cont_out.shape[0] == 18  # 3 + 15
        assert z.shape[0] == 18


# ---------------------------------------------------------------------------
# GAT: GATWithJK
# ---------------------------------------------------------------------------


class TestGAT:
    @pytest.fixture()
    def model(self):
        from graphids.core.models.gat import GATWithJK

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
        """GAT forward produces [batch_size, 2] logits."""
        batch = _make_batch(num_graphs=5)
        logits = model(batch)
        assert logits.shape == (5, 2)

    def test_return_embedding(self, model):
        """return_embedding=True gives (logits, embedding) tuple."""
        batch = _make_batch(num_graphs=3)
        logits, emb = model(batch, return_embedding=True)
        assert logits.shape == (3, 2)
        assert emb.shape[0] == 3
        assert emb.shape[1] > 0

    def test_gradient_flow(self, model):
        """Gradients flow through GAT to all parameters."""
        batch = _make_batch(num_graphs=3)
        logits = model(batch)
        loss = logits.sum()
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"

    def test_parameter_update(self, model):
        """One optimizer step changes parameters."""
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        batch = _make_batch(num_graphs=3)

        old_params = {n: p.clone() for n, p in model.named_parameters()}
        loss = model(batch).sum()
        loss.backward()
        opt.step()

        changed = sum(
            1 for n, p in model.named_parameters() if not torch.equal(p, old_params[n])
        )
        assert changed > 0

    def test_variable_size_graphs(self, model):
        """GAT handles variable-size graph batches."""
        g1 = _make_graph(num_nodes=3, num_edges=4)
        g2 = _make_graph(num_nodes=20, num_edges=40)
        g3 = _make_graph(num_nodes=7, num_edges=10)
        batch = Batch.from_data_list([g1, g2, g3])
        logits = model(batch)
        assert logits.shape == (3, 2)

    def test_from_config(self):
        """from_config constructs valid model."""
        from graphids.config import resolve
        from graphids.core.models.gat import GATWithJK

        cfg = resolve("model_type=gat", "scale=small")
        model = GATWithJK.from_config(cfg, NUM_IDS, IN_CHANNELS)
        batch = _make_batch(num_graphs=2)
        logits = model(batch)
        assert logits.shape == (2, 2)


# ---------------------------------------------------------------------------
# DQN: QNetwork
# ---------------------------------------------------------------------------


class TestQNetwork:
    def test_forward_shape(self):
        """QNetwork produces [batch, action_dim] Q-values."""
        from graphids.core.models.dqn import QNetwork

        state_dim = 15  # VGAE(8) + GAT(7) = 15
        action_dim = 21
        model = QNetwork(state_dim, action_dim, hidden_dim=32, num_layers=2)
        x = torch.randn(8, state_dim)
        q = model(x)
        assert q.shape == (8, action_dim)

    def test_gradient_flow(self):
        from graphids.core.models.dqn import QNetwork

        model = QNetwork(15, 21, hidden_dim=32, num_layers=2)
        x = torch.randn(4, 15)
        loss = model(x).sum()
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_get_known_models(self):
        """Registry returns constructors for all three model types."""
        from graphids.core.models.registry import get

        for model_type in ("vgae", "gat", "dqn"):
            fn = get(model_type)
            assert callable(fn)

    def test_get_unknown_raises(self):
        from graphids.core.models.registry import get

        with pytest.raises(KeyError, match="Unknown model_type"):
            get("nonexistent")

    def test_fusion_state_dim(self):
        """State dim is VGAE(8) + GAT(7) = 15."""
        from graphids.core.models.registry import fusion_state_dim

        assert fusion_state_dim() == 15

    def test_feature_layout_offsets(self):
        """Feature layout offsets are contiguous and match extractor dims."""
        from graphids.core.models.registry import feature_layout

        layout = feature_layout()
        assert "vgae" in layout
        assert "gat" in layout
        assert layout["vgae"].offset == 0
        assert layout["vgae"].dim == 8
        assert layout["gat"].offset == 8
        assert layout["gat"].dim == 7
