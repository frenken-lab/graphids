"""Models module for CAN-Graph project.

Public API re-exported from submodules:

    from graphids.core.models import fusion_state_dim, feature_layout, extractors
    from graphids.core.models import FusionFeatureExtractor, VGAEFusionExtractor, GATFusionExtractor
"""

from .fusion_features import (
    FusionFeatureExtractor,
    GATFusionExtractor,
    VGAEFusionExtractor,
)
from .registry import (
    FeatureLayout,
    extractors,
    feature_layout,
    fusion_state_dim,
)
