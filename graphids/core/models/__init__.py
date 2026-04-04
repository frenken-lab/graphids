"""Models package for CAN-Graph.

Family-oriented namespaces:

    from graphids.core.models.autoencoder import VGAEModule, DGIModule
    from graphids.core.models.supervised import GATModule
    from graphids.core.models.fusion import BanditFusionModule, DQNFusionModule

Public API re-exported from submodules:

    from graphids.core.models import STATE_DIM, LAYOUT, EXTRACTORS
    from graphids.core.models import GraphModuleBase
"""

from . import autoencoder, fusion, supervised
from ._training import GraphModuleBase
from .fusion.bandit import BanditFusionModule
from .fusion.dqn import DQNFusionModule
from .fusion.fusion_features import EXTRACTORS, LAYOUT, STATE_DIM, FeatureLayout

__all__ = [
    "autoencoder",
    "supervised",
    "fusion",
    "GraphModuleBase",
    "BanditFusionModule",
    "DQNFusionModule",
    "EXTRACTORS",
    "LAYOUT",
    "STATE_DIM",
    "FeatureLayout",
]
