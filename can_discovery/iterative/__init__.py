"""Versioned, provisional cross-vehicle CAN discovery."""
from .rounds import (
                     fit_shared_model,
                     infer_vehicle,
                     promote_round,
                     validate_proposals,
)
from ..manifests.common import artifact_hash, write_artifact
from ..manifests.iterative import freeze_manifest
