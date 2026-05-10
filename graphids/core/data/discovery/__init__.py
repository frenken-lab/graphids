"""Signal-discovery primitives for cross-vehicle ontology building."""

from .canonical import (
    CanonicalEntitySpec,
    CanonicalFeatureFrameSpec,
    CanonicalRegistry,
    build_canonical_feature_frame,
)
from .hypotheses import (
    DiscoveryStore,
    SignalHypothesisSpec,
    SignalProfileSpec,
    build_signal_profiles,
    initialize_hypotheses,
)
from .layout import (
    DataLayerLayout,
    HypothesisRecordSpec,
    MaterializedViewSpec,
    RawEventTableSpec,
)

__all__ = [
    "CanonicalEntitySpec",
    "CanonicalFeatureFrameSpec",
    "CanonicalRegistry",
    "build_canonical_feature_frame",
    "SignalProfileSpec",
    "SignalHypothesisSpec",
    "DiscoveryStore",
    "RawEventTableSpec",
    "MaterializedViewSpec",
    "HypothesisRecordSpec",
    "DataLayerLayout",
    "build_signal_profiles",
    "initialize_hypotheses",
]
