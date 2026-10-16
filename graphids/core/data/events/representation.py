"""Raw CAN frame contract, independent of torch and the snapshot representation."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class CANEventRepresentationCfg:
    kind: Literal["can_event"] = "can_event"
    schema_version: int = 1

    def __post_init__(self):
        if self.kind != "can_event" or self.schema_version != 1:
            raise ValueError("requires can_event schema_version=1")
