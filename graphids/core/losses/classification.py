"""Classification losses as ``nn.Module`` with a uniform signature.

Every classification loss in this file implements
``forward(logits, labels, graph=None)`` so it can be composed with
:class:`graphids.core.losses.distillation.SoftLabelDistillation` — the
distillation wrapper passes ``graph`` through for the teacher forward,
while non-distillation losses ignore it.

Using ``nn.Module`` (rather than plain functions or ``nn.CrossEntropyLoss``
directly) lets the same object be a drop-in replacement regardless of
whether KD is active, which is the whole point of the Option B decoupling.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    """Plain cross-entropy. Ignores ``graph``."""

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, graph=None) -> torch.Tensor:
        return F.cross_entropy(logits, labels)


class WeightedCrossEntropyLoss(nn.Module):
    """Cross-entropy with per-class weights, registered as a buffer so it
    follows the module across ``.to(device)`` without being a learned parameter.
    """

    def __init__(self, weights: list[float]):
        super().__init__()
        self.register_buffer("weights", torch.as_tensor(weights, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, graph=None) -> torch.Tensor:
        return F.cross_entropy(logits, labels, weight=self.weights)


class FocalLoss(nn.Module):
    """Focal loss (Lin et al. 2017) for class-imbalanced classification.

    ``loss = (1 - p_t)^γ * CE(logits, targets)``
    """

    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, graph=None) -> torch.Tensor:
        ce = F.cross_entropy(logits, labels, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()
