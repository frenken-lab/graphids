"""Per-model Pydantic schemas — auto-generated from each ``__init__``.

Each entry is a single line that mirrors the corresponding
LightningModule's kwargs via ``schema_for``. When you change a model's
``__init__``, the schema updates on next import — no manual
maintenance, no drift.

If a specific model needs custom validators (enum constraints, range
checks, cross-field rules), subclass the auto-generated schema here.
"""

from __future__ import annotations

from graphids.core._schema_gen import schema_for
from graphids.core.models.autoencoder.dgi_module import DGIModule
from graphids.core.models.autoencoder.vgae_module import VGAEModule
from graphids.core.models.fusion.bandit import BanditFusionModule
from graphids.core.models.fusion.dqn import DQNFusionModule
from graphids.core.models.fusion.fusion_baselines import (
    MLPFusionModule,
    WeightedAvgModule,
)
from graphids.core.models.supervised.gat_module import GATModule

VGAEConfig = schema_for(VGAEModule)
DGIConfig = schema_for(DGIModule)
GATConfig = schema_for(GATModule)
BanditConfig = schema_for(BanditFusionModule)
DQNConfig = schema_for(DQNFusionModule)
MLPFusionConfig = schema_for(MLPFusionModule)
WeightedAvgConfig = schema_for(WeightedAvgModule)

__all__ = [
    "VGAEConfig",
    "DGIConfig",
    "GATConfig",
    "BanditConfig",
    "DQNConfig",
    "MLPFusionConfig",
    "WeightedAvgConfig",
]
