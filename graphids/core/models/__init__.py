"""Models package for CAN-Graph.

    from graphids.core.models.autoencoder import VGAE, DGI
    from graphids.core.models.supervised import GATModule
    from graphids.core.models.fusion import BanditFusionModule, DQNFusionModule
"""

from . import autoencoder, fusion, supervised
from .base import GraphModuleBase
from .fusion.bandit import BanditFusionModule
from .fusion.dqn import DQNFusionModule

__all__ = [
    "autoencoder",
    "supervised",
    "fusion",
    "GraphModuleBase",
    "BanditFusionModule",
    "DQNFusionModule",
]
