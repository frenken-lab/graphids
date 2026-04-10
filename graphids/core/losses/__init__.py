"""Composable loss modules for GraphIDS training.

All losses in this package are ``nn.Module`` so they can be swapped in
and out via :func:`graphids.instantiate._build_loss` without touching
the training module. Knowledge distillation is expressed as a wrapper
around a base loss, not as trainer / callback / IO infrastructure.

Two signature protocols:

- Classification (``(logits, labels, graph=None) → scalar``):
  :class:`CrossEntropyLoss`, :class:`WeightedCrossEntropyLoss`,
  :class:`FocalLoss`, :class:`SoftLabelDistillation`.
- Autoencoder (``(student_outputs, batch) → scalar``):
  :class:`VGAETaskLoss`, :class:`FeatureDistillation`.
"""

from __future__ import annotations

from .autoencoder import VGAETaskLoss
from .classification import CrossEntropyLoss, FocalLoss, WeightedCrossEntropyLoss
from .distillation import FeatureDistillation, SoftLabelDistillation

__all__ = [
    "CrossEntropyLoss",
    "FeatureDistillation",
    "FocalLoss",
    "SoftLabelDistillation",
    "VGAETaskLoss",
    "WeightedCrossEntropyLoss",
]
