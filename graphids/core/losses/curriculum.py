"""Curriculum learning at the loss end: schedule + weighted-loss wrapper.

Two pieces, both consumed at training-step time:

- ``LinearRampSchedule``: pure callable
  ``(epoch, difficulty, in_scope) -> weights``. Zero data dependencies.
  Out-of-scope graphs always weight 1; in-scope graphs ramp from
  ``start_ratio/end_ratio`` of the easiest at epoch 0 to all visible at
  ``max_epochs - 1``.
- ``CurriculumWeightedLoss``: wraps any per-example base loss
  (``reduction='none'``); per forward, pulls the schedule's weights and
  reduces ``per_ex * weights / weights.sum()``.

Difficulty scoring (``score_vgae`` / ``score_random``) is preprocessing —
see :mod:`graphids.core.data.preprocessing.curriculum`. Graphs carry
``Data.difficulty`` + ``Data.in_scope`` from ``dm.setup``; PyG's
``Batch.from_data_list`` collates them so the wrapper reads
``batch.difficulty`` + ``batch.in_scope`` directly.

Eval paths: construct a non-wrapped loss, OR call ``set_epoch`` past
``schedule.max_epochs - 1`` so all weights default to 1.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class LinearRampSchedule:
    """Pure callable: ``(epoch, difficulty, in_scope) -> binary weights``.

    - Out-of-scope (``in_scope == False``): always weight 1.
    - In-scope, in active set: weight 1. Active set is the
      ``fraction(epoch) * n_in_scope`` easiest in-scope examples.
    - In-scope, not yet unlocked: weight 0.

    Defaults match the prior tier schedule's pacing (10% in-scope visible
    at start, 100% by ``max_epochs - 1``).
    """

    def __init__(
        self,
        start_ratio: float = 1.0,
        end_ratio: float = 10.0,
        max_epochs: int = 300,
    ):
        if end_ratio <= 0:
            raise ValueError(f"end_ratio must be positive, got {end_ratio}")
        if start_ratio <= 0 or start_ratio > end_ratio:
            raise ValueError(
                f"start_ratio must be in (0, end_ratio]; got start={start_ratio}, end={end_ratio}"
            )
        if max_epochs < 1:
            raise ValueError(f"max_epochs must be >= 1, got {max_epochs}")
        self.start_ratio = float(start_ratio)
        self.end_ratio = float(end_ratio)
        self.max_epochs = int(max_epochs)

    def fraction(self, epoch: int) -> float:
        """Visible fraction of in-scope examples at ``epoch`` ∈ [0, 1]."""
        progress = min(epoch / max(self.max_epochs - 1, 1), 1.0)
        ratio = self.start_ratio + (self.end_ratio - self.start_ratio) * progress
        return ratio / self.end_ratio

    def __call__(
        self,
        epoch: int,
        difficulty: torch.Tensor,
        in_scope: torch.Tensor,
    ) -> torch.Tensor:
        if difficulty.shape != in_scope.shape:
            raise ValueError(
                f"difficulty {tuple(difficulty.shape)} and in_scope "
                f"{tuple(in_scope.shape)} must have the same shape"
            )
        in_scope_b = in_scope.bool()
        weights = (~in_scope_b).to(torch.float)  # out-of-scope → 1, in-scope → 0
        in_idx = in_scope_b.nonzero(as_tuple=True)[0]
        n_in = int(in_idx.numel())
        if n_in == 0:
            return weights

        n_active = max(1, min(n_in, math.ceil(self.fraction(epoch) * n_in)))
        # `topk(largest=False)` returns the n_active easiest in-scope examples.
        easiest = torch.topk(difficulty[in_idx], n_active, largest=False).indices
        weights[in_idx[easiest]] = 1.0
        return weights


class CurriculumWeightedLoss(nn.Module):
    """Per-example masking via ``(difficulty, schedule)``.

    ``base_loss`` must produce per-example output (``reduction='none'``);
    this wrapper does the reduction with schedule-derived weights.
    ``schedule`` is any callable ``(epoch, difficulty, in_scope) -> weights``.

    ``epoch`` set externally via :meth:`set_epoch` from a Lightning hook
    (typically ``on_train_epoch_start``), keeping the loss forward
    signature ``(logits, labels, graph)`` stable across the codebase.
    """

    def __init__(self, base_loss: nn.Module, schedule):
        super().__init__()
        if hasattr(base_loss, "reduction") and base_loss.reduction != "none":
            raise ValueError(
                f"CurriculumWeightedLoss requires base_loss.reduction='none', "
                f"got {base_loss.reduction!r}"
            )
        self.base_loss = base_loss
        self.schedule = schedule
        self._epoch: int = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, graph=None) -> torch.Tensor:
        if graph is None:
            raise ValueError(
                "CurriculumWeightedLoss requires the `graph` (Batch) arg "
                "to read batch.difficulty and batch.in_scope"
            )
        if not hasattr(graph, "difficulty") or not hasattr(graph, "in_scope"):
            # val/test batches don't carry curriculum attributes — unweighted mean
            return self.base_loss(logits, labels).mean()
        per_ex = self.base_loss(logits, labels)
        if per_ex.dim() != 1:
            raise ValueError(
                f"base_loss must return per-example loss (1-D); got shape {tuple(per_ex.shape)}"
            )
        weights = self.schedule(self._epoch, graph.difficulty, graph.in_scope).to(
            per_ex.device, per_ex.dtype
        )
        return (per_ex * weights).sum() / weights.sum().clamp_min(1.0)
