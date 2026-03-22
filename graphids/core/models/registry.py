"""Model registry: maps model_type → constructor + fusion feature extractor.

Registration order (VGAE then GAT) determines feature concatenation order
in the 15-D state vector that trained DQN checkpoints expect.
"""

from __future__ import annotations

from typing import NamedTuple

from .fusion_features import (
    FusionFeatureExtractor,
    GATFusionExtractor,
    VGAEFusionExtractor,
)
from .gat import GATWithJK
from .vgae import GraphAutoencoderNeighborhood


class FeatureLayout(NamedTuple):
    """Layout of one extractor's features within the fusion state vector."""

    offset: int
    dim: int
    confidence_idx: int


def _dqn_from_config(cfg, num_ids: int = 0, in_ch: int = 0):
    """Lazy import to break registry → dqn → registry cycle."""
    from .dqn import QNetwork

    return QNetwork.from_config(cfg)


# Order matters — VGAE then GAT matches the 15-D state layout for trained DQN checkpoints.
_MODELS: dict[str, tuple[callable, FusionFeatureExtractor | None]] = {
    "vgae": (GraphAutoencoderNeighborhood.from_config, VGAEFusionExtractor()),
    "gat": (GATWithJK.from_config, GATFusionExtractor()),
    "dqn": (_dqn_from_config, None),
}


def get(model_type: str) -> callable:
    """Return the constructor for a model type."""
    try:
        return _MODELS[model_type][0]
    except KeyError:
        raise KeyError(f"Unknown model_type '{model_type}'. Available: {list(_MODELS)}") from None


def fusion_state_dim() -> int:
    """Total dimension of the fused state vector (sum of all extractor dims)."""
    return sum(ext.feature_dim for _, ext in _MODELS.values() if ext is not None)


def feature_layout() -> dict[str, FeatureLayout]:
    """Return {name: FeatureLayout} with offsets computed from registration order."""
    layout: dict[str, FeatureLayout] = {}
    offset = 0
    for name, (_, ext) in _MODELS.items():
        if ext is not None:
            layout[name] = FeatureLayout(offset, ext.feature_dim, offset + ext.confidence_index)
            offset += ext.feature_dim
    return layout


def extractors() -> list[tuple[str, FusionFeatureExtractor]]:
    """Return (name, extractor) pairs in registration order."""
    return [(name, ext) for name, (_, ext) in _MODELS.items() if ext is not None]


def fusion_test_metrics():
    """Standard MetricCollection for fusion eval (shared by DQN, bandit, MLP, weighted_avg)."""
    from torchmetrics import MetricCollection
    from torchmetrics.classification import (
        BinaryAccuracy, BinaryAUROC, BinaryF1Score,
        BinaryPrecision, BinaryRecall, BinarySpecificity,
    )
    return MetricCollection({
        "accuracy": BinaryAccuracy(), "f1": BinaryF1Score(),
        "precision": BinaryPrecision(), "recall": BinaryRecall(),
        "specificity": BinarySpecificity(), "auc": BinaryAUROC(),
    })
