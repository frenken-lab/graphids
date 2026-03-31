"""Models module for CAN-Graph project.

Public API re-exported from submodules:

    from graphids.core.models import fusion_state_dim, feature_layout, extractors
    from graphids.core.models import FusionFeatureExtractor, VGAEFusionExtractor, GATFusionExtractor
    from graphids.core.models import GraphModuleBase
"""

from ._training import GraphModuleBase
from .fusion_features import (
    FeatureLayout,
    FusionFeatureExtractor,
    GATFusionExtractor,
    VGAEFusionExtractor,
    extractors,
    feature_layout,
    fusion_state_dim,
)
from .bandit import BanditFusionModule, NeuralLinUCBAgent
from .dqn import DQNFusionModule, EnhancedDQNFusionAgent
