"""Curriculum primitives — difficulty + schedule.

Two primitives, both pure / stateless beyond hyperparameters:
- ``score_vgae`` / ``score_random`` (free functions producing per-graph
  difficulty tensors).
- ``LinearRampSchedule`` (callable mapping epoch + difficulty + in_scope
  to per-example weights consumed by ``CurriculumWeightedLoss``).
"""

from __future__ import annotations

import pytest
import torch
from conftest import make_graph

from graphids.core.data.preprocessing.curriculum import score_random
from graphids.core.losses.curriculum import LinearRampSchedule


def _mk_graphs(n=30):
    out = [make_graph() for _ in range(n)]
    for g in out:
        g.y = torch.tensor([0])
    return out


class TestScoreRandom:
    """Random difficulty baseline — control for the curriculum mechanism.

    Not a "no-curriculum" condition. ``score_random`` paired with a
    schedule still hides examples per epoch; it isolates whether the
    *mechanism* contributes signal independent of any informed difficulty
    ordering.
    """

    def test_deterministic_under_seed(self):
        # CONTRACT: same seed → identical scores (ablation reproducibility).
        ds = _mk_graphs(20)
        a = score_random(ds, seed=13)
        b = score_random(ds, seed=13)
        assert torch.equal(a, b)

    def test_different_seeds_diverge(self):
        ds = _mk_graphs(20)
        a = score_random(ds, seed=1)
        b = score_random(ds, seed=2)
        assert not torch.equal(a, b)

    def test_output_shape_and_dtype(self):
        ds = _mk_graphs(15)
        out = score_random(ds, seed=0)
        assert out.shape == (15,)
        assert out.dtype == torch.float


class TestLinearRampSchedule:
    """Pure callable: (epoch, difficulty, in_scope) → binary weights."""

    @staticmethod
    def _make_inputs(n_in_scope=10, n_out_scope=3):
        n = n_in_scope + n_out_scope
        difficulty = torch.arange(n, dtype=torch.float)
        in_scope = torch.zeros(n, dtype=torch.bool)
        in_scope[:n_in_scope] = True
        return difficulty, in_scope

    def test_output_shape_and_dtype(self):
        # CONTRACT: weights aligned with difficulty; binary {0, 1}.
        d, s = self._make_inputs()
        sched = LinearRampSchedule(max_epochs=10)
        w = sched(0, d, s)
        assert w.shape == d.shape
        assert torch.all((w == 0) | (w == 1))

    def test_out_of_scope_always_one(self):
        # INVARIANT: out-of-scope examples bypass the curriculum at every epoch.
        d, s = self._make_inputs(n_in_scope=10, n_out_scope=4)
        sched = LinearRampSchedule(max_epochs=20)
        for epoch in [0, 5, 10, 19, 100]:
            w = sched(epoch, d, s)
            assert torch.all(w[~s] == 1.0), f"out-of-scope dropped at epoch {epoch}"

    def test_first_epoch_uses_start_fraction(self):
        # CONTRACT: epoch 0 activates ceil(start/end * n_in_scope) easiest.
        d, s = self._make_inputs(n_in_scope=10, n_out_scope=2)
        sched = LinearRampSchedule(start_ratio=1.0, end_ratio=10.0, max_epochs=10)
        w = sched(0, d, s)
        # 1/10 * 10 = 1 active in-scope. Easiest difficulty = index 0.
        assert w[0] == 1.0
        assert w[1:10].sum() == 0.0
        assert torch.all(w[10:] == 1.0)

    def test_final_epoch_unlocks_all_in_scope(self):
        # CONTRACT: epoch max-1 unlocks every in-scope example.
        d, s = self._make_inputs(n_in_scope=10, n_out_scope=2)
        sched = LinearRampSchedule(start_ratio=1.0, end_ratio=10.0, max_epochs=10)
        w = sched(9, d, s)
        assert torch.all(w == 1.0)

    def test_easiest_unlocked_first(self):
        # INVARIANT: at any epoch, active in-scope examples are the lowest-
        # difficulty ones. Sort flipped would unlock hardest first
        # (anti-curriculum) — regression guard.
        n = 20
        difficulty = torch.tensor([float(n - i) for i in range(n)])
        in_scope = torch.ones(n, dtype=torch.bool)
        sched = LinearRampSchedule(start_ratio=1.0, end_ratio=10.0, max_epochs=20)
        w = sched(5, difficulty, in_scope)
        active_idx = w.nonzero(as_tuple=True)[0].tolist()
        active_difficulties = difficulty[active_idx]
        non_active_difficulties = difficulty[w == 0]
        assert active_difficulties.max() <= non_active_difficulties.min()

    def test_monotone_non_decreasing_per_example(self):
        # INVARIANT: once unlocked, an example stays unlocked. weights[e+1] >= weights[e].
        d, s = self._make_inputs(n_in_scope=20, n_out_scope=3)
        sched = LinearRampSchedule(start_ratio=1.0, end_ratio=10.0, max_epochs=20)
        prev = sched(0, d, s)
        for e in range(1, 20):
            cur = sched(e, d, s)
            assert torch.all(cur >= prev), f"example dropped between epoch {e - 1} and {e}"
            prev = cur

    def test_clamped_past_max_epochs(self):
        # CONTRACT: running past max_epochs holds at full visibility.
        d, s = self._make_inputs(n_in_scope=10, n_out_scope=2)
        sched = LinearRampSchedule(max_epochs=10)
        w_end = sched(9, d, s)
        w_far = sched(10_000, d, s)
        assert torch.equal(w_end, w_far)

    def test_no_in_scope_returns_all_ones(self):
        # EDGE CASE: empty in-scope set. Schedule has nothing to gate;
        # everything is out-of-scope (= weight 1).
        n = 5
        d = torch.zeros(n)
        s = torch.zeros(n, dtype=torch.bool)
        sched = LinearRampSchedule(max_epochs=10)
        w = sched(0, d, s)
        assert torch.all(w == 1.0)

    def test_shape_mismatch_rejected(self):
        sched = LinearRampSchedule(max_epochs=10)
        with pytest.raises(ValueError, match="must have the same shape"):
            sched(0, torch.zeros(5), torch.zeros(7, dtype=torch.bool))

    def test_init_validates_args(self):
        with pytest.raises(ValueError, match="end_ratio"):
            LinearRampSchedule(start_ratio=1.0, end_ratio=0.0, max_epochs=10)
        with pytest.raises(ValueError, match="start_ratio"):
            LinearRampSchedule(start_ratio=-1.0, end_ratio=10.0, max_epochs=10)
        with pytest.raises(ValueError, match="start_ratio"):
            LinearRampSchedule(start_ratio=11.0, end_ratio=10.0, max_epochs=10)
        with pytest.raises(ValueError, match="max_epochs"):
            LinearRampSchedule(start_ratio=1.0, end_ratio=10.0, max_epochs=0)
