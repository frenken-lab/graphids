"""VGAE: architecture, forward pass, gradient flow, checkpoint roundtrip.

After the Phase-1 collapse, ``VGAE`` is the single class — arch + trainer
bridge in one ``nn.Module``. Tensor-form forward is exposed as
``_forward_tensors`` for focused architecture tests; ``forward(batch)`` is
the trainer-facing entry point.
"""

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


def _make_vgae(*, conv_type: str = "gatv2", **overrides):
    from graphids.core.losses.autoencoder import VGAETaskLoss
    from graphids.core.models.autoencoder.vgae import VGAE

    kwargs = dict(
        loss_fn=VGAETaskLoss(),
        hidden_dims=[32, 16],
        latent_dim=16,
        heads=2,
        embedding_dim=4,
        dropout=0.0,
        conv_type=conv_type,
        edge_dim=EDGE_DIM,
        proj_dim=0,
        num_ids=NUM_IDS,
        in_channels=IN_CHANNELS,
        gradient_checkpointing=False,
        compile_model=False,
    )
    kwargs.update(overrides)
    return VGAE(**kwargs)


class TestVGAEConvTypes:
    """Forward pass, gradient flow, and variable-size graphs for each conv type."""

    @pytest.fixture(params=["gatv2", "gps"], ids=["gatv2", "gps"])
    def model_and_conv(self, request):
        return _make_vgae(conv_type=request.param), request.param

    def test_forward_shapes(self, model_and_conv):
        model, _ = model_and_conv
        batch = make_batch(3)
        n = batch.x.size(0)
        e = batch.edge_index.size(1)
        cont, canid_logits, nbr_pred, z, kl, edge_logits = model._forward_tensors(
            batch.x,
            batch.edge_index,
            batch.batch,
            edge_attr=batch.edge_attr,
            node_id=batch.node_id,
        )
        assert cont.shape == (n, IN_CHANNELS)
        assert canid_logits.shape == (n, NUM_IDS)
        # GAD-NR neighborhood decoder predicts neighbor-mean in latent space:
        # output dim is latent_dim, NOT num_ids (the previous vocabulary-bag
        # head; see 2026-05-06-drop-neighborhood-adopt-tam.md).
        assert nbr_pred.shape == (n, model.hparams.latent_dim)
        assert z.shape[0] == n
        assert kl.shape == (n,)
        # edge_logits is None when the conv stack doesn't consume edge_attr;
        # otherwise [E, edge_dim] for the edge_decoder MLP.
        if edge_logits is not None:
            assert edge_logits.shape == (e, EDGE_DIM)

    def test_gradient_flow(self, model_and_conv):
        model, conv_type = model_and_conv
        batch = make_batch(2)
        cont, canid_logits, nbr_pred, _z, kl, edge_logits = model._forward_tensors(
            batch.x,
            batch.edge_index,
            batch.batch,
            edge_attr=batch.edge_attr,
            node_id=batch.node_id,
        )
        graph_loss = cont.sum() + canid_logits.sum() + nbr_pred.sum() + kl.sum()
        if edge_logits is not None:
            graph_loss = graph_loss + edge_logits.sum()
        torch.autograd.backward(graph_loss)
        # GPS encoder_bns have no gradient by design (batch norm in GPS path).
        # mask_token by design has requires_grad=False.
        # torchmetrics modules (roc_metric, test_metrics) have no learnable
        # params either.
        exclude = {"encoder_bns"} if conv_type == "gps" else set()
        exclude.update({"mask_token", "roc_metric", "test_metrics"})
        dead = [
            n
            for n, p in model.named_parameters()
            if p.requires_grad and p.grad is None and not any(ex in n for ex in exclude)
        ]
        assert not dead, f"No gradient: {dead}"

    def test_variable_size_graphs(self, model_and_conv):
        model, _ = model_and_conv
        batch = make_variable_batch([3, 15])
        out = model._forward_tensors(
            batch.x,
            batch.edge_index,
            batch.batch,
            edge_attr=batch.edge_attr,
            node_id=batch.node_id,
        )
        assert out[0].shape[0] == 18

    def test_tam_affinity_shape(self, model_and_conv):
        # CONTRACT: tam_affinity returns one scalar per node, finite, in [0, 2]
        # (since 1 - cosine, with cosine ∈ [-1, 1]). Isolated source nodes are
        # allowed (scatter-mean default = 0 → affinity = 1).
        from graphids.core.models._score_primitives import tam_affinity

        model, _ = model_and_conv
        batch = make_batch(3)
        with torch.no_grad():
            _, _, _, z, _, _ = model._forward_tensors(
                batch.x,
                batch.edge_index,
                batch.batch,
                edge_attr=batch.edge_attr,
                node_id=batch.node_id,
            )
            affinity = tam_affinity(z, batch.edge_index)
        assert affinity.shape == (batch.x.size(0),)
        assert torch.isfinite(affinity).all()
        assert (affinity >= 0).all() and (affinity <= 2).all()


def test_rayleigh_quotient_per_graph():
    # CONTRACT: rayleigh_quotient returns one non-negative scalar per graph,
    # finite, with shape [G] when batch index is supplied.
    from graphids.core.models._score_primitives import rayleigh_quotient

    batch = make_batch(3)
    rq = rayleigh_quotient(batch.x, batch.edge_index, batch=batch.batch)
    assert rq.shape == (3,)
    assert torch.isfinite(rq).all()
    assert (rq >= 0).all()


def test_score_output_shapes():
    # CONTRACT: _score returns 7-tuple of finite per-graph tensors shaped [G].
    model = _make_vgae()
    model.eval()
    batch = make_batch(3)
    with torch.no_grad():
        recon, recon_max, affinity, rq, mahal, kl, z = model._score(batch)
    for name, t in [
        ("recon", recon),
        ("recon_max", recon_max),
        ("affinity", affinity),
        ("rq", rq),
        ("mahal", mahal),
        ("kl", kl),
    ]:
        assert t.shape == (3,), f"{name}: expected shape (3,), got {t.shape}"
        assert torch.isfinite(t).all(), f"{name} contains non-finite values"
    assert z.shape[0] == batch.x.size(0)


def test_score_requires_fitted_norm():
    # CONTRACT: score() raises RuntimeError when score_norm_fitted is False
    # (i.e., on_test_setup has not been called).
    model = _make_vgae()
    batch = make_batch(2)
    with pytest.raises(RuntimeError, match="on_test_setup"):
        model.score(batch)


@pytest.mark.slow
class TestVGAEFastDevRun:
    def test_vgae(self, vgae_cfg):
        """INVARIANT: VGAE.training_step produces a finite, backpropagable loss."""
        module = _make_vgae(
            hidden_dims=vgae_cfg.hidden_dims,
            latent_dim=vgae_cfg.latent_dim,
            heads=vgae_cfg.heads,
            embedding_dim=vgae_cfg.embedding_dim,
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
        m1 = _make_vgae()
        m1.eval()
        torch.save(m1.state_dict(), tmp_path / "v.ckpt")

        m2 = _make_vgae()
        m2.load_state_dict(torch.load(tmp_path / "v.ckpt", weights_only=True))
        m2.eval()

        batch = make_batch(2)
        with torch.no_grad():
            torch.manual_seed(0)
            o1 = m1(batch)
            torch.manual_seed(0)
            o2 = m2(batch)
        torch.testing.assert_close(o1[0], o2[0])
