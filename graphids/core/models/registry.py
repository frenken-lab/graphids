"""Model registry: maps model_type → fusion feature extractor + LightningModule loader.

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


class FeatureLayout(NamedTuple):
    """Layout of one extractor's features within the fusion state vector."""

    offset: int
    dim: int
    confidence_idx: int


# Lazy module class loaders (avoid circular imports at registration time).

def _vgae_module():
    from .vgae import VGAEModule
    return VGAEModule

def _gat_module():
    from .gat import GATModule
    return GATModule

def _dgi_module():
    from .dgi import DGIModule
    return DGIModule


# Order matters — VGAE then GAT matches the 15-D state layout for trained DQN checkpoints.
# DGI has no fusion extractor (doesn't participate in fusion — contrastive, not reconstructive).
# Tuple: (fusion_extractor, module_cls_loader)
_MODELS: dict[str, tuple[FusionFeatureExtractor | None, callable | None]] = {
    "vgae": (VGAEFusionExtractor(), _vgae_module),
    "gat": (GATFusionExtractor(), _gat_module),
    "dqn": (None, None),
    "dgi": (None, _dgi_module),
}


def get_module_cls(model_type: str) -> type:
    """Return the LightningModule class for a model type."""
    entry = _MODELS.get(model_type)
    if entry is None or entry[1] is None:
        raise KeyError(f"No module class registered for '{model_type}'. Available: {[k for k, v in _MODELS.items() if v[1]]}")
    return entry[1]()


def fusion_state_dim() -> int:
    """Total dimension of the fused state vector (sum of all extractor dims)."""
    return sum(ext.feature_dim for ext, _ in _MODELS.values() if ext is not None)


def feature_layout() -> dict[str, FeatureLayout]:
    """Return {name: FeatureLayout} with offsets computed from registration order."""
    layout: dict[str, FeatureLayout] = {}
    offset = 0
    for name, (ext, _) in _MODELS.items():
        if ext is not None:
            layout[name] = FeatureLayout(offset, ext.feature_dim, offset + ext.confidence_index)
            offset += ext.feature_dim
    return layout


def extractors() -> list[tuple[str, FusionFeatureExtractor]]:
    """Return (name, extractor) pairs in registration order."""
    return [(name, ext) for name, (ext, _) in _MODELS.items() if ext is not None]


