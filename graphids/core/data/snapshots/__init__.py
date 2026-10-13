"""Causal CAN snapshots and immutable reference context."""

from .manifest import CANRecordingManifest, Recording
from .preprocessing import (
    CANSnapshotRepresentationCfg,
    SnapshotRecording,
    build_snapshots,
)

__all__ = [
    "CANRecordingManifest",
    "Recording",
    "CANSnapshotRepresentationCfg",
    "SnapshotRecording",
    "build_snapshots",
]
