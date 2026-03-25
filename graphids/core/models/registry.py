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


def _dgi_from_config(cfg, num_ids: int = 0, in_ch: int = 0):
    """Lazy import for DGI model."""
    from .dgi import GraphInfomaxModel

    return GraphInfomaxModel.from_config(cfg, num_ids, in_ch)


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
# Tuple: (arch_factory, fusion_extractor, module_cls_loader)
_MODELS: dict[str, tuple[callable, FusionFeatureExtractor | None, callable | None]] = {
    "vgae": (GraphAutoencoderNeighborhood.from_config, VGAEFusionExtractor(), _vgae_module),
    "gat": (GATWithJK.from_config, GATFusionExtractor(), _gat_module),
    "dqn": (_dqn_from_config, None, None),
    "dgi": (_dgi_from_config, None, _dgi_module),
}


def get(model_type: str) -> callable:
    """Return the architecture constructor for a model type."""
    try:
        return _MODELS[model_type][0]
    except KeyError:
        raise KeyError(f"Unknown model_type '{model_type}'. Available: {list(_MODELS)}") from None


def get_module_cls(model_type: str) -> type:
    """Return the LightningModule class for a model type."""
    entry = _MODELS.get(model_type)
    if entry is None or entry[2] is None:
        raise KeyError(f"No module class registered for '{model_type}'. Available: {[k for k, v in _MODELS.items() if v[2]]}")
    return entry[2]()


def fusion_state_dim() -> int:
    """Total dimension of the fused state vector (sum of all extractor dims)."""
    return sum(ext.feature_dim for _, ext, _ in _MODELS.values() if ext is not None)


def feature_layout() -> dict[str, FeatureLayout]:
    """Return {name: FeatureLayout} with offsets computed from registration order."""
    layout: dict[str, FeatureLayout] = {}
    offset = 0
    for name, (_, ext, _) in _MODELS.items():
        if ext is not None:
            layout[name] = FeatureLayout(offset, ext.feature_dim, offset + ext.confidence_index)
            offset += ext.feature_dim
    return layout


def extractors() -> list[tuple[str, FusionFeatureExtractor]]:
    """Return (name, extractor) pairs in registration order."""
    return [(name, ext) for name, (_, ext, _) in _MODELS.items() if ext is not None]


