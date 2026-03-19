"""Typed result containers for evaluation inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GATResult:
    """Results from GAT inference pass."""

    preds: np.ndarray  # [N] int
    labels: np.ndarray  # [N] int
    scores: np.ndarray  # [N] float (softmax P(anomaly))
    attack_types: np.ndarray  # [N] int
    embeddings: np.ndarray | None = None
    attention: list[dict] | None = None


@dataclass(frozen=True)
class VGAEResult:
    """Results from VGAE reconstruction-error inference."""

    errors: np.ndarray  # [N] float (reconstruction error)
    labels: np.ndarray  # [N] int
    attack_types: np.ndarray  # [N] int
    embeddings: np.ndarray | None = None
    components: dict[str, np.ndarray] | None = None


@dataclass(frozen=True)
class FusionResult:
    """Results from DQN/MLP/WeightedAvg fusion inference."""

    preds: np.ndarray  # [N] int
    labels: np.ndarray  # [N] int
    scores: np.ndarray  # [N] float (fused anomaly score)
    q_values: np.ndarray  # [N, n_actions]
