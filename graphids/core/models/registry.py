"""Model registry: centralizes model construction and fusion feature extraction.

Default registrations (at module load):
    vgae  →  VGAEFusionExtractor (8-D)
    gat   →  GATFusionExtractor  (7-D)
    dqn   →  None (consumes features, doesn't produce them)

Registration order determines feature concatenation order in
``cache_predictions``.  VGAE is registered first to preserve the existing
15-D state layout (VGAE features then GAT features).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import torch.nn as nn


from .fusion_features import (
    FusionFeatureExtractor,
    GATFusionExtractor,
    VGAEFusionExtractor,
)


class FeatureLayout(NamedTuple):
    """Layout of one extractor's features within the fusion state vector."""

    offset: int
    dim: int
    confidence_idx: int


_REGISTRY: dict[str, ModelEntry] = {}


@dataclass
class ModelEntry:
    model_type: str
    factory: Callable
    extractor: FusionFeatureExtractor | None


def register(entry: ModelEntry) -> None:
    """Register a model type."""
    _REGISTRY[entry.model_type] = entry


def get(model_type: str) -> ModelEntry:
    """Get a registered model entry."""
    if model_type not in _REGISTRY:
        raise KeyError(f"Model type '{model_type}' not registered. Available: {list(_REGISTRY)}")
    return _REGISTRY[model_type]


def fusion_state_dim() -> int:
    """Sum of all registered extractors' feature_dim values."""
    return sum(
        entry.extractor.feature_dim for entry in _REGISTRY.values() if entry.extractor is not None
    )


def feature_layout() -> dict[str, FeatureLayout]:
    """Return ``{name: FeatureLayout(offset, dim, confidence_idx)}`` for each extractor.

    Offsets are computed from registration order so the DQN agent can
    look up slices by name instead of hardcoding indices.
    """
    layout: dict[str, FeatureLayout] = {}
    offset = 0
    for name, entry in _REGISTRY.items():
        if entry.extractor is not None:
            dim = entry.extractor.feature_dim
            conf_abs = offset + entry.extractor.confidence_index
            layout[name] = FeatureLayout(offset, dim, conf_abs)
            offset += dim
    return layout


def extractors() -> list[tuple[str, FusionFeatureExtractor]]:
    """Return (name, extractor) pairs in registration order.

    Registration order is VGAE then GAT, matching the existing 15-D state
    layout used by trained DQN checkpoints.
    """
    return [
        (name, entry.extractor) for name, entry in _REGISTRY.items() if entry.extractor is not None
    ]


# ---------------------------------------------------------------------------
# Factory functions (lazy imports to avoid circular dependencies)
# ---------------------------------------------------------------------------


def _vgae_factory(cfg, num_ids: int, in_ch: int) -> nn.Module:
    from .vgae import GraphAutoencoderNeighborhood

    return GraphAutoencoderNeighborhood.from_config(cfg, num_ids, in_ch)


def _gat_factory(cfg, num_ids: int, in_ch: int) -> nn.Module:
    from .gat import GATWithJK

    return GATWithJK.from_config(cfg, num_ids, in_ch)


def _dqn_factory(cfg, num_ids: int, in_ch: int) -> nn.Module:
    from .dqn import QNetwork

    return QNetwork.from_config(cfg)


# ---------------------------------------------------------------------------
# Default registrations (order matters for feature concatenation)
# ---------------------------------------------------------------------------

register(ModelEntry("vgae", _vgae_factory, VGAEFusionExtractor()))
register(ModelEntry("gat", _gat_factory, GATFusionExtractor()))
register(ModelEntry("dqn", _dqn_factory, None))
