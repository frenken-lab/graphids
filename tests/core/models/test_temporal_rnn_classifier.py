from __future__ import annotations

import torch
from torch_geometric.data import TemporalData


def _temporal_batch(reset_after: list[bool] | None = None) -> TemporalData:
    labels = [0, 1, 0, 1]
    n = len(labels)
    ids = torch.arange(n, dtype=torch.long)
    return TemporalData(
        src=ids,
        dst=ids + 1,
        t=torch.arange(n, dtype=torch.float32),
        msg=torch.tensor(
            [
                [0.0, 0.1, 0.0, 1.0],
                [1.0, 0.1, 1.0, 1.0],
                [0.0, 0.2, 0.0, 1.0],
                [1.0, 0.2, 1.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        y=torch.tensor(labels, dtype=torch.long),
        attack_type=torch.tensor(labels, dtype=torch.long),
        stream_id=torch.zeros(n, dtype=torch.long),
        reset_after=torch.tensor(reset_after if reset_after is not None else [False] * n),
        event_id=ids,
        is_scored=torch.tensor([True, False, True, True]),
    )


def _model():
    from graphids.core.losses import CrossEntropyLoss
    from graphids.core.models.temporal import TemporalRNNClassifier

    torch.manual_seed(7)
    return TemporalRNNClassifier(
        loss_fn=CrossEntropyLoss(),
        hidden=8,
        layers=1,
        embedding_dim=4,
        dropout=0.0,
        num_ids=8,
        in_channels=4,
    )


def test_temporal_rnn_classifier_consumes_temporal_event_batches():
    model = _model()
    batch = _temporal_batch()

    logits = model(batch)
    loss = model.training_step(batch, 0)
    loss.backward()

    assert tuple(logits.shape) == (4, 2)
    assert torch.isfinite(loss)
    assert model._train_state is not None
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_temporal_rnn_classifier_carries_and_resets_state_across_batches():
    model = _model()
    first = _temporal_batch([False, False, False, False])
    second = _temporal_batch([False, False, False, True])

    _logits_1, state_1 = model._forward_with_state(first, None)
    assert state_1 is not None

    logits_carried, state_2 = model._forward_with_state(second, state_1.detach())
    logits_fresh, _fresh_state = model._forward_with_state(second, None)

    assert not torch.allclose(logits_carried, logits_fresh)
    assert state_2 is None


def test_temporal_rnn_classifier_resets_state_inside_batch():
    model = _model()
    batch = _temporal_batch([False, True, False, True])
    second_stream = batch[2:]

    logits, state = model._forward_with_state(batch, None)
    fresh_logits, fresh_state = model._forward_with_state(second_stream, None)

    assert torch.allclose(logits[2:], fresh_logits, atol=1e-6)
    assert state is None
    assert fresh_state is None
