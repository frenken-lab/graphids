"""Models package for CAN-Graph.

Family-oriented namespaces:

    from graphids.core.models.autoencoder import VGAEModule, DGIModule
    from graphids.core.models.supervised import GATModule
    from graphids.core.models.fusion import BanditFusionModule, DQNFusionModule

Public API re-exported from submodules:

    from graphids.core.models import fusion_state_dim, feature_layout, extractors
    from graphids.core.models import FusionFeatureExtractor, VGAEFusionExtractor, GATFusionExtractor
    from graphids.core.models import GraphModuleBase
"""

from . import autoencoder, fusion, supervised

from ._training import GraphModuleBase
from .fusion.fusion_features import (
    FeatureLayout,
    FusionFeatureExtractor,
    GATFusionExtractor,
    VGAEFusionExtractor,
    extractors,
    feature_layout,
    fusion_state_dim,
)
from .fusion.bandit import BanditFusionModule, NeuralLinUCBAgent
from .fusion.dqn import DQNFusionModule, EnhancedDQNFusionAgent

__all__ = [
    "autoencoder",
    "supervised",
    "fusion",
    "GraphModuleBase",
    "FeatureLayout",
    "FusionFeatureExtractor",
    "GATFusionExtractor",
    "VGAEFusionExtractor",
    "extractors",
    "feature_layout",
    "fusion_state_dim",
    "BanditFusionModule",
    "NeuralLinUCBAgent",
    "DQNFusionModule",
    "EnhancedDQNFusionAgent",
]
