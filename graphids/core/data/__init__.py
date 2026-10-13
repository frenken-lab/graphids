"""CAN recording contracts and lazy snapshot training interfaces."""

from .snapshots.manifest import CANRecordingManifest, Recording
from .snapshots.preprocessing import CANSnapshotRepresentationCfg

__all__ = ["CANRecordingManifest", "Recording", "CANSnapshotRepresentationCfg"]


def __getattr__(name):
    if name == "CANSnapshotDataModule":
        from .snapshots.datamodule import CANSnapshotDataModule

        return CANSnapshotDataModule
    raise AttributeError(name)
