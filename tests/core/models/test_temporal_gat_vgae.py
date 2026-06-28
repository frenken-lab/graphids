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


def test_temporal_gat_consumes_temporal_event_batches():
    from graphids.core.losses import CrossEntropyLoss
    from graphids.core.models.temporal import TemporalGAT

    model = TemporalGAT(
        loss_fn=CrossEntropyLoss(),
        hidden=8,
        layers=1,
        heads=2,
        embedding_dim=4,
        dropout=0.0,
        num_ids=4,
        in_channels=4,
    )
    batch = _temporal_batch()

    logits = model(batch)
    loss = model.training_step(batch, 0)
    loss.backward()

    assert tuple(logits.shape) == (4, 2)
    assert torch.isfinite(loss)
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_temporal_vgae_consumes_temporal_event_batches():
    from graphids.core.models.temporal import TemporalVGAE

    model = TemporalVGAE(
        hidden=8,
        layers=1,
        embedding_dim=4,
        latent_dim=4,
        dropout=0.0,
        num_ids=4,
        in_channels=4,
    )
    batch = _temporal_batch()

    out = model(batch)
    loss = model.training_step(batch, 0)
    scores = model.score(batch)
    loss.backward()

    assert tuple(out["recon"].shape) == (4, 4)
    assert torch.isfinite(loss)
    assert tuple(scores.shape) == (4,)
    assert torch.isfinite(scores).all()
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)
