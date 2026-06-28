"""Root public primitive API for GraphIDS."""

from __future__ import annotations

from graphids.core.data.preprocessing.representations import (
    RepresentationCfg,
    TemporalRepresentationCfg,
    representation_kind,
)
from graphids.primitives_data import (
    CANBusCfg,
    DataCfg,
    TemporalDMCfg,
    can_bus,
    temporal_dm,
)
from graphids.primitives_losses import (
    CELossCfg,
    FocalLossCfg,
    LossFn,
    SimpleLossFn,
    WeightedCELossCfg,
    ce,
    focal,
    weighted_ce,
)
from graphids.primitives_models import (
    ModelCfg,
    TemporalEventClassifierCfg,
    TemporalGATCfg,
    TemporalVGAECfg,
    temporal_event_classifier,
    temporal_gat,
    temporal_vgae,
)

__all__ = [
    "temporal_event_classifier",
    "temporal_gat",
    "temporal_vgae",
    "focal",
    "ce",
    "weighted_ce",
    "can_bus",
    "TemporalRepresentationCfg",
    "RepresentationCfg",
    "representation_kind",
    "temporal_dm",
    "ModelCfg",
    "LossFn",
    "SimpleLossFn",
    "DataCfg",
    "TemporalEventClassifierCfg",
    "TemporalGATCfg",
    "TemporalVGAECfg",
    "FocalLossCfg",
    "CELossCfg",
    "WeightedCELossCfg",
    "CANBusCfg",
    "TemporalDMCfg",
]
