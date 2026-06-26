"""Loss modules for GraphIDS training."""

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
