"""``reduction`` kwarg on classification losses.

Step 3a of the curriculum-primitives loss-masking redesign: the curriculum
wrapper needs per-example loss output. Each base classification loss
gains an opt-in ``reduction`` constructor arg (``'mean'`` default
preserves all prior call sites).
"""

from __future__ import annotations

import pytest
import torch

from graphids.core.losses import CrossEntropyLoss, FocalLoss, WeightedCrossEntropyLoss


def _xent_inputs(n=8, c=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(n, c, generator=g)
    labels = torch.randint(0, c, (n,), generator=g)
    return logits, labels


class TestReductionParity:
    """``reduction='none'`` returns per-example losses; default 'mean' unchanged."""

    def test_cross_entropy_default_unchanged(self):
        # REGRESSION: pre-existing call sites pass no reduction arg and
        # depend on scalar output.
        logits, labels = _xent_inputs()
        out = CrossEntropyLoss()(logits, labels)
        assert out.shape == ()

    def test_cross_entropy_none_returns_per_example(self):
        logits, labels = _xent_inputs(n=8)
        out = CrossEntropyLoss(reduction="none")(logits, labels)
        assert out.shape == (8,)

    def test_cross_entropy_mean_eq_none_mean(self):
        # CONTRACT: 'mean' is the per-example mean. Curriculum wrapper relies on
        # this to substitute its own weighted reduction without changing scale.
        logits, labels = _xent_inputs()
        m = CrossEntropyLoss(reduction="mean")(logits, labels)
        n = CrossEntropyLoss(reduction="none")(logits, labels)
        assert torch.allclose(m, n.mean(), atol=1e-6)

    def test_focal_default_unchanged(self):
        logits, labels = _xent_inputs()
        out = FocalLoss(gamma=2.0)(logits, labels)
        assert out.shape == ()

    def test_focal_none_returns_per_example(self):
        logits, labels = _xent_inputs(n=8)
        out = FocalLoss(gamma=2.0, reduction="none")(logits, labels)
        assert out.shape == (8,)

    def test_focal_mean_eq_none_mean(self):
        logits, labels = _xent_inputs()
        m = FocalLoss(gamma=2.0, reduction="mean")(logits, labels)
        n = FocalLoss(gamma=2.0, reduction="none")(logits, labels)
        assert torch.allclose(m, n.mean(), atol=1e-6)

    def test_weighted_ce_default_unchanged(self):
        logits, labels = _xent_inputs(c=2)
        out = WeightedCrossEntropyLoss(weights=[1.0, 5.0])(logits, labels)
        assert out.shape == ()

    def test_weighted_ce_none_returns_per_example(self):
        logits, labels = _xent_inputs(n=8, c=2)
        out = WeightedCrossEntropyLoss(weights=[1.0, 5.0], reduction="none")(logits, labels)
        assert out.shape == (8,)

    def test_weighted_ce_mean_is_weighted_mean(self):
        # CONTRACT: 'mean' for WeightedCE is weighted mean (sum / Σweights[labels]).
        # This is PyTorch's behavior (F.cross_entropy with weight + reduction='mean')
        # — we preserve it. The curriculum wrapper, when it cares about parity,
        # should reduce with reduction='none' explicitly and apply its own weights.
        logits, labels = _xent_inputs(c=2)
        weights = [1.0, 5.0]
        m = WeightedCrossEntropyLoss(weights=weights)(logits, labels)
        per_ex = WeightedCrossEntropyLoss(weights=weights, reduction="none")(logits, labels)
        denom = torch.tensor(weights)[labels].sum()
        assert torch.allclose(m, per_ex.sum() / denom, atol=1e-6)

    @pytest.mark.parametrize("cls,kw", [
        (CrossEntropyLoss, {}),
        (FocalLoss, {"gamma": 2.0}),
        (WeightedCrossEntropyLoss, {"weights": [1.0, 5.0]}),
    ])
    def test_invalid_reduction_rejected(self, cls, kw):
        with pytest.raises(ValueError, match="reduction"):
            cls(reduction="bogus", **kw)
