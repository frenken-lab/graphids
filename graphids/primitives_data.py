"""Data primitive configs and dataset registry helpers."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from graphids.core.data.preprocessing.representations import (
    RepresentationCfg,
    TemporalRepresentationCfg,
    representation_kind,
)
from graphids.paths import load_catalog

_DEFAULT_REPRESENTATION_CFG = TemporalRepresentationCfg()


class _Cfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CANBusCfg(_Cfg):
    type: Literal["can_bus"] = "can_bus"
    name: str
    seed: int
    val_fraction: float = 0.2
    train_source_mode: Literal["mixed", "attack_free"] = "mixed"
    val_split: Literal["chronological", "recordings"] = "chronological"
    split_seed: int = 42
    representation_cfg: RepresentationCfg = Field(default_factory=TemporalRepresentationCfg)


class TemporalDMCfg(_Cfg):
    type: Literal["temporal_dm"] = "temporal_dm"
    source: CANBusCfg
    batch_size: int = 256
    batch_mode: Literal["events", "stream_lanes"] = "events"
    stream_lanes: int | None = None
    chunk_size: int | None = None
    num_workers: int = 0
    prebuild_batches: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    drop_partial_lane_batches: bool = False
    coverage_policy: Literal["full", "diagnostic"] = "full"
    stream_execution: Literal["continuous", "segmented"] = "continuous"
    segment_events: int = Field(default=2048, ge=1)
    segment_warmup_events: int = Field(default=256, ge=0)
    eval_batch_mode: Literal["events", "stream_lanes"] = "events"
    val_prefix_events: int | None = Field(default=None, ge=1)
    val_warmup_events: int = 0
    test_warmup_events: int = 0

    @model_validator(mode="after")
    def _validate_batch_mode(self) -> Self:
        if self.batch_mode == "stream_lanes":
            if self.stream_lanes is None or self.stream_lanes < 2:
                raise ValueError("stream_lanes must be >= 2 for batch_mode='stream_lanes'")
            if self.chunk_size is None or self.chunk_size < 1:
                raise ValueError("chunk_size must be positive for batch_mode='stream_lanes'")
            if self.coverage_policy == "full" and self.drop_partial_lane_batches:
                raise ValueError("full coverage requires drop_partial_lane_batches: false")
        if self.stream_execution == "segmented" and self.batch_mode != "stream_lanes":
            raise ValueError("segmented execution requires stream_lanes")
        if self.eval_batch_mode == "stream_lanes" and (self.stream_lanes is None or self.chunk_size is None):
            raise ValueError("lane evaluation requires stream_lanes and chunk_size")
        if self.source.val_split == "recordings" and self.val_prefix_events is not None:
            raise ValueError("recording validation requires complete streams; remove val_prefix_events")
        return self

    def build(self) -> Any:
        from graphids.core.data.datamodule.temporal import TemporalDataModule
        from graphids.core.data.datasets.can_bus import CANBusTemporalSource

        if representation_kind(self.source.representation_cfg) != "temporal":
            raise ValueError("temporal_dm requires representation_cfg.kind='temporal'")
        source = CANBusTemporalSource(
            name=self.source.name,
            val_fraction=self.source.val_fraction,
            representation_cfg=self.source.representation_cfg,
            train_source_mode=self.source.train_source_mode,
            val_split=self.source.val_split,
            split_seed=self.source.split_seed,
            val_warmup_events=self.val_warmup_events,
            test_warmup_events=self.test_warmup_events,
        )
        return TemporalDataModule(
            dataset=source,
            batch_size=self.batch_size,
            batch_mode=self.batch_mode,
            stream_lanes=self.stream_lanes,
            chunk_size=self.chunk_size,
            num_workers=self.num_workers,
            prebuild_batches=self.prebuild_batches,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_partial_lane_batches=self.drop_partial_lane_batches,
            coverage_policy=self.coverage_policy,
            stream_execution=self.stream_execution,
            segment_events=self.segment_events,
            segment_warmup_events=self.segment_warmup_events,
            eval_batch_mode=self.eval_batch_mode,
            val_prefix_events=self.val_prefix_events,

        )


DataCfg = Annotated[TemporalDMCfg, Field(discriminator="type")]


def can_bus(
    *,
    dataset: str,
    seed: int,
    val_fraction: float = 0.2,
    train_source_mode: Literal["mixed", "attack_free"] = "mixed",
    val_split: Literal["chronological", "recordings"] = "chronological",
    split_seed: int = 42,
    representation_cfg: RepresentationCfg = _DEFAULT_REPRESENTATION_CFG,
) -> CANBusCfg:
    registry = load_catalog()
    if dataset not in registry:
        raise ValueError(f"unknown dataset: {dataset} (registry: {', '.join(sorted(registry))})")
    return CANBusCfg(
        name=dataset,
        seed=seed,
        val_fraction=val_fraction,
        train_source_mode=train_source_mode,
        val_split=val_split,
        split_seed=split_seed,
        representation_cfg=representation_cfg,
    )


def temporal_dm(
    *,
    source: CANBusCfg,
    **overrides: Any,
) -> TemporalDMCfg:
    return TemporalDMCfg(source=source, **overrides)
