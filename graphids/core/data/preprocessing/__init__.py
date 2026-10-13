"""CAN snapshot preprocessing configuration."""

from graphids.core.data.snapshots.preprocessing import CANSnapshotRepresentationCfg

from .representations import (
    RepresentationCfg,
    representation_digest,
    representation_kind,
    representation_payload,
)

__all__ = [
    "RepresentationCfg",
    "CANSnapshotRepresentationCfg",
    "representation_digest",
    "representation_kind",
    "representation_payload",
]
