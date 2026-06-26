from __future__ import annotations

import torch
from torch_geometric.data import TemporalData


def _temporal_batch() -> TemporalData:
    return TemporalData(
        src=torch.tensor([0, 1, 1, 2], dtype=torch.long),
        dst=torch.tensor([1, 1, 2, 2], dtype=torch.long),
        t=torch.arange(4, dtype=torch.float32),
        msg=torch.tensor(
            [
                [0.0, 0.1, 0.0, 1.0],
                [1.0, 0.1, 1.0, 1.0],
                [0.0, 0.2, 0.0, 1.0],
                [1.0, 0.2, 1.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        y=torch.tensor([0, 1, 0, 1], dtype=torch.long),
        attack_type=torch.tensor([0, 2, 0, 2], dtype=torch.long),
        stream_id=torch.zeros(4, dtype=torch.long),
        reset_after=torch.tensor([False, False, False, True]),
        event_id=torch.arange(4, dtype=torch.long),
        is_scored=torch.tensor([True, False, True, True]),
    )


def test_gat_consumes_temporal_event_batches():
    from graphids.core.losses import CrossEntropyLoss
    from graphids.core.models.supervised.gat import GAT

    model = GAT(
        loss_fn=CrossEntropyLoss(),
        hidden=8,
        layers=2,
        heads=2,
        fc_layers=2,
        embedding_dim=4,
        dropout=0.0,
        num_ids=4,
        in_channels=4,
        gradient_checkpointing=False,
    )
    batch = _temporal_batch()

    logits = model(batch)
    loss = model.training_step(batch, 0)
    loss.backward()

    assert tuple(logits.shape) == (4, 2)
    assert torch.isfinite(loss)
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_vgae_consumes_temporal_event_batches():
    from graphids.core.losses import VGAETaskLoss
    from graphids.core.models.autoencoder.vgae import VGAE

    model = VGAE(
        loss_fn=VGAETaskLoss(),
        hidden_dims=[8],
        latent_dim=8,
        heads=2,
        embedding_dim=4,
        dropout=0.0,
        num_ids=4,
        in_channels=4,
        gradient_checkpointing=False,
        batch_norm=False,
    )
    batch = _temporal_batch()

    loss = model.training_step(batch, 0)
    scores = model.score(batch)
    model._fit_temporal_score_norm([batch], torch.device("cpu"))
    normalized_scores = model.score(batch)
    loss.backward()

    assert torch.isfinite(loss)
    assert tuple(scores.shape) == (4,)
    assert torch.isfinite(scores).all()
    assert tuple(normalized_scores.shape) == (4,)
    assert torch.isfinite(normalized_scores).all()
    assert bool(model.score_norm_fitted)
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)
