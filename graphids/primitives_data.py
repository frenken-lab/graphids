"""Continuous event stream batching and pooled CAN data configuration."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from graphids.core.data.events.representation import CANEventRepresentationCfg


class _Cfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CANEventDMCfg(_Cfg):
    type: Literal["can_event_dm"] = "can_event_dm"
    manifest_path: str
    datasets: list[str]
    cache_root: str
    representation_cfg: CANEventRepresentationCfg = Field(
        default_factory=CANEventRepresentationCfg
    )
    chunk_events: int = Field(default=512, ge=1)
    warmup_events: Literal[0] = 0
    stream_batch_size: Literal[1, 2, 4, 8] = 1
    effective_batch_events: int = Field(default=8192, ge=1)
    num_workers: int = Field(default=0, ge=0)
    pin_memory: bool = False

    @model_validator(mode="after")
    def validate_microbatch(self):
        events = self.chunk_events * self.stream_batch_size
        if events > 8192 or events > self.effective_batch_events:
            raise ValueError(
                "microbatch must contain at most 8192 events and fit effective_batch_events"
            )
        return self

    def build(self):
        from graphids.core.data.events.data import CANEventDataModule

        return CANEventDataModule(
            **self.model_dump(exclude={"type", "representation_cfg"}),
            representation_cfg=self.representation_cfg,
        )


def can_event_dm(**kwargs) -> CANEventDMCfg:
    return CANEventDMCfg(**kwargs)


DataCfg = CANEventDMCfg
