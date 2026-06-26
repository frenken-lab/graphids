from __future__ import annotations


def test_temporal_event_classifier_default_loss_survives_hparam_capture():
    from graphids.core.losses import CrossEntropyLoss
    from graphids.core.models.temporal import TemporalEventClassifier

    model = TemporalEventClassifier(num_ids=3, in_channels=4)

    assert isinstance(model.loss_fn, CrossEntropyLoss)
