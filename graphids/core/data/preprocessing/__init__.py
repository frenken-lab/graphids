"""CAN event preprocessing configuration."""

from graphids.core.data.events.representation import CANEventRepresentationCfg

from .representations import (
    RepresentationCfg,
    representation_digest,
    representation_kind,
    representation_payload,
)

__all__ = [
    "RepresentationCfg",
    "CANEventRepresentationCfg",
    "representation_digest",
    "representation_kind",
    "representation_payload",
]
