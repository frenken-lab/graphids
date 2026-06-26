"""Classification losses with a shared module signature."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

_REDUCTIONS = ("mean", "sum", "none")


def _reduce(x: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "mean":
        return x.mean()
    if reduction == "sum":
        return x.sum()
    if reduction == "none":
        return x
    raise ValueError(f"reduction must be one of {_REDUCTIONS}, got {reduction!r}")


class CrossEntropyLoss(nn.Module):
    """Plain cross-entropy. Ignores ``graph``."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        if reduction not in _REDUCTIONS:
            raise ValueError(f"reduction must be one of {_REDUCTIONS}, got {reduction!r}")
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, graph=None) -> torch.Tensor:
        return F.cross_entropy(logits, labels, reduction=self.reduction)


class WeightedCrossEntropyLoss(nn.Module):
    """Cross-entropy with per-class weights, registered as a buffer so it
    follows the module across ``.to(device)`` without being a learned parameter.
    """

    def __init__(self, weights: list[float], reduction: str = "mean"):
        super().__init__()
        if reduction not in _REDUCTIONS:
            raise ValueError(f"reduction must be one of {_REDUCTIONS}, got {reduction!r}")
        self.register_buffer("weights", torch.as_tensor(weights, dtype=torch.float32))
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, graph=None) -> torch.Tensor:
        return F.cross_entropy(logits, labels, weight=self.weights, reduction=self.reduction)


class FocalLoss(nn.Module):
    """Focal loss (Lin et al. 2017) for class-imbalanced classification.

    ``loss = (1 - p_t)^γ * CE(logits, targets)``
    """

    def __init__(self, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        if reduction not in _REDUCTIONS:
            raise ValueError(f"reduction must be one of {_REDUCTIONS}, got {reduction!r}")
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, graph=None) -> torch.Tensor:
        ce = F.cross_entropy(logits, labels, reduction="none")
        pt = torch.exp(-ce)
        return _reduce((1 - pt) ** self.gamma * ce, self.reduction)
