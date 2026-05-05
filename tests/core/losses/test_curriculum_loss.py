"""CurriculumWeightedLoss — per-example masking via (difficulty, schedule).

Step 3b of the curriculum-primitives loss-masking redesign. Wraps any
``reduction='none'`` classification loss; reduces with schedule-derived
weights at forward time. Epoch is set via :meth:`set_epoch` from a
Lightning hook.
"""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Batch, Data

from graphids.core.losses import CrossEntropyLoss, CurriculumWeightedLoss, FocalLoss
from graphids.core.losses.curriculum import LinearRampSchedule


def _mk_batch(difficulties: list[float], in_scope: list[bool], y_per_graph=None) -> Batch:
    """Build a Batch with per-graph curriculum attrs.

    Each ``Data`` has 1 node, scalar features, and y[0] is the class label.
    Curriculum reads only ``batch.difficulty`` / ``batch.in_scope``.
    """
    n = len(difficulties)
    if y_per_graph is None:
        y_per_graph = [int(s) for s in in_scope]  # in-scope → 0, out → 1
    graphs = []
    for i in range(n):
        g = Data(
            x=torch.randn(1, 4),
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            y=torch.tensor([y_per_graph[i]]),
        )
        g.difficulty = torch.tensor([float(difficulties[i])])
        g.in_scope = torch.tensor([bool(in_scope[i])])
        graphs.append(g)
    return Batch.from_data_list(graphs)


def _mk_logits(n: int, c: int = 2, seed: int = 0) -> torch.Tensor:
    return torch.randn(n, c, generator=torch.Generator().manual_seed(seed), requires_grad=True)


class TestForward:
    def test_rejects_reduction_mean_base(self):
        # CONTRACT: base must be reduction='none' so the wrapper can mask
        # before reducing. Catching this at construction beats getting a
        # cryptic shape error during training.
        with pytest.raises(ValueError, match="reduction='none'"):
            CurriculumWeightedLoss(
                base_loss=CrossEntropyLoss(reduction="mean"),
                schedule=LinearRampSchedule(max_epochs=10),
            )

    def test_rejects_missing_graph(self):
        loss = CurriculumWeightedLoss(
            base_loss=CrossEntropyLoss(reduction="none"),
            schedule=LinearRampSchedule(max_epochs=10),
        )
        with pytest.raises(ValueError, match="graph"):
            loss(torch.zeros(2, 2), torch.zeros(2, dtype=torch.long))

    def test_rejects_batch_without_curriculum_attrs(self):
        # REGRESSION: silently using a non-curriculum batch produces wrong
        # gradients (schedule sees garbage). Fail loud.
        loss = CurriculumWeightedLoss(
            base_loss=CrossEntropyLoss(reduction="none"),
            schedule=LinearRampSchedule(max_epochs=10),
        )
        batch = Batch.from_data_list([
            Data(
                x=torch.randn(1, 4),
                edge_index=torch.zeros(2, 0, dtype=torch.long),
                y=torch.tensor([0]),
            )
            for _ in range(3)
        ])
        with pytest.raises(ValueError, match="curriculum attributes"):
            loss(_mk_logits(3), batch.y, batch)

    def test_returns_scalar(self):
        loss = CurriculumWeightedLoss(
            base_loss=CrossEntropyLoss(reduction="none"),
            schedule=LinearRampSchedule(max_epochs=10),
        )
        batch = _mk_batch([1.0, 2.0, 3.0], [True, True, False])
        out = loss(_mk_logits(3), batch.y, batch)
        assert out.shape == ()


class TestMaskingSemantics:
    """The weights tensor zeros out the right gradients."""

    def test_dormant_in_scope_contributes_zero(self):
        # CONTRACT: at epoch 0 with start_ratio=1, end_ratio=10, max_epochs=10:
        # 10% of in-scope visible → only easiest in-scope unlocks; the rest
        # contribute zero loss AND zero gradient.
        sched = LinearRampSchedule(start_ratio=1.0, end_ratio=10.0, max_epochs=10)
        loss = CurriculumWeightedLoss(CrossEntropyLoss(reduction="none"), sched)
        # 10 in-scope graphs, ascending difficulty 0..9. n_active=1 → only g[0].
        batch = _mk_batch(
            difficulties=list(range(10)),
            in_scope=[True] * 10,
        )
        logits = _mk_logits(10)
        loss.set_epoch(0)
        out = loss(logits, batch.y, batch)
        out.backward()
        # Easiest (idx 0) has nonzero grad; the rest are exactly zero.
        grads = logits.grad
        assert torch.any(grads[0] != 0)
        assert torch.all(grads[1:] == 0)

    def test_out_of_scope_always_contributes(self):
        # INVARIANT: out-of-scope examples bypass the curriculum and ALWAYS
        # backprop. Three attack-like graphs all marked out-of-scope.
        sched = LinearRampSchedule(start_ratio=1.0, end_ratio=10.0, max_epochs=10)
        loss = CurriculumWeightedLoss(CrossEntropyLoss(reduction="none"), sched)
        batch = _mk_batch(
            difficulties=[100.0, 200.0, 300.0],  # arbitrary; in_scope mask overrides
            in_scope=[False, False, False],
            y_per_graph=[1, 1, 1],
        )
        logits = _mk_logits(3)
        loss.set_epoch(0)
        out = loss(logits, batch.y, batch)
        out.backward()
        assert torch.all(logits.grad.abs().sum(dim=1) > 0)

    def test_set_epoch_advances_visible_set(self):
        # INVARIANT: more examples receive nonzero gradient as epoch grows.
        sched = LinearRampSchedule(start_ratio=1.0, end_ratio=10.0, max_epochs=10)
        loss = CurriculumWeightedLoss(CrossEntropyLoss(reduction="none"), sched)
        batch = _mk_batch(
            difficulties=list(range(10)),
            in_scope=[True] * 10,
        )

        def n_active(epoch: int) -> int:
            logits = _mk_logits(10)
            loss.set_epoch(epoch)
            out = loss(logits, batch.y, batch)
            out.backward()
            nonzero = (logits.grad.abs().sum(dim=1) > 0).sum().item()
            return nonzero

        counts = [n_active(e) for e in [0, 4, 9]]
        assert counts[0] < counts[1] < counts[2]
        assert counts[2] == 10

    def test_full_unlock_matches_unweighted_mean(self):
        # DIFFERENTIAL: at epoch max-1 with start=1/end=10, every in-scope is
        # unlocked, so the schedule weights are all 1. The wrapper's reduction
        # then equals an unweighted mean of per-example losses — same number
        # CrossEntropyLoss(reduction='mean') would have produced.
        sched = LinearRampSchedule(start_ratio=1.0, end_ratio=10.0, max_epochs=10)
        wrapped = CurriculumWeightedLoss(CrossEntropyLoss(reduction="none"), sched)
        wrapped.set_epoch(9)
        batch = _mk_batch(
            difficulties=[1.0, 2.0, 3.0, 4.0],
            in_scope=[True] * 4,
            y_per_graph=[0, 1, 0, 1],
        )
        logits = _mk_logits(4, seed=42)
        wrapped_out = wrapped(logits, batch.y, batch)
        plain = CrossEntropyLoss(reduction="mean")(_mk_logits(4, seed=42), batch.y)
        assert torch.allclose(wrapped_out, plain, atol=1e-6)


class TestBaseLossSwap:
    """Wrapper composes with any per-example classification loss."""

    @pytest.mark.parametrize("base_factory", [
        lambda: CrossEntropyLoss(reduction="none"),
        lambda: FocalLoss(gamma=2.0, reduction="none"),
    ])
    def test_works_with(self, base_factory):
        sched = LinearRampSchedule(max_epochs=10)
        loss = CurriculumWeightedLoss(base_factory(), sched)
        batch = _mk_batch([1.0, 2.0, 3.0], [True, True, False])
        out = loss(_mk_logits(3), batch.y, batch)
        assert out.shape == () and torch.isfinite(out)
