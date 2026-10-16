"""Continuous event stream batching and pooled CAN data configuration."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from graphids import event_defaults as defaults
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
    chunk_events: int = Field(default=defaults.CHUNK_EVENTS, ge=1)
    stream_batch_size: int = Field(default=defaults.STREAM_BATCH_SIZE, ge=1, le=16)
    max_microbatch_events: int = Field(
        default=defaults.MAX_MICROBATCH_EVENTS, ge=1, le=262144
    )
    effective_batch_events: int = Field(default=defaults.EFFECTIVE_BATCH_EVENTS, ge=1)
    num_workers: int = Field(default=defaults.NUM_WORKERS, ge=0)
    pin_memory: bool = defaults.PIN_MEMORY
    prefetch_factor: int = Field(default=defaults.PREFETCH_FACTOR, ge=1)
    persistent_workers: bool = defaults.PERSISTENT_WORKERS
    cuda_prefetch: bool = defaults.CUDA_PREFETCH

    @model_validator(mode="after")
    def validate_microbatch(self):
        events = self.chunk_events * self.stream_batch_size
        if events > self.max_microbatch_events or events > self.effective_batch_events:
            raise ValueError(
                "microbatch must fit max_microbatch_events and effective_batch_events"
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
