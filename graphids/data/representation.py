"""CAN frames: raw schema 1 or raw frames plus causal predecessor metadata in 2."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class CANEventRepresentationConfig:
    kind: Literal["can_event"] = "can_event"
    schema_version: Literal[1, 2] = 2

    def __post_init__(self):
        if self.kind != "can_event" or self.schema_version not in (1, 2):
            raise ValueError("requires can_event schema_version=1 or 2")
