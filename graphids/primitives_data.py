"""Data primitive configs and dataset registry helpers."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from graphids.config_validation import validate_lane_config
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


    def build(self, *, val_warmup_events: int = 0, test_warmup_events: int = 0) -> Any:
        from graphids.core.data.datasets.can_bus import CANBusTemporalSource

        return CANBusTemporalSource(
            **self.model_dump(exclude={"type", "seed", "representation_cfg"}),
            representation_cfg=self.representation_cfg,
            val_warmup_events=val_warmup_events,
            test_warmup_events=test_warmup_events,
        )


class CrossVehicleCfg(_Cfg):
    type: Literal["cross_vehicle"] = "cross_vehicle"
    held_out_vehicle: Literal["chevrolet_silverado", "subaru_forester", "chevrolet_traverse"]
    source_vehicle: Literal["chevrolet_impala", "chevrolet_silverado", "subaru_forester", "chevrolet_traverse"] | None = None
    val_fraction: float = 0.1
    split_seed: int = 42
    val_split: Literal["recordings"] = "recordings"
    representation_cfg: RepresentationCfg = Field(default_factory=TemporalRepresentationCfg)

    def build(self, *, val_warmup_events: int = 64, test_warmup_events: int = 64) -> Any:
        from graphids.core.data.datasets.cross_vehicle import CrossVehicleTemporalSource

        return CrossVehicleTemporalSource(
            held_out_vehicle=self.held_out_vehicle,
            source_vehicle=self.source_vehicle,
            val_fraction=self.val_fraction,
            split_seed=self.split_seed,
            representation_cfg=self.representation_cfg,
            val_warmup_events=val_warmup_events,
            test_warmup_events=test_warmup_events,
        )


class TemporalDMCfg(_Cfg):
    type: Literal["temporal_dm"] = "temporal_dm"
    source: CANBusCfg | CrossVehicleCfg = Field(discriminator="type")
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
    eval_stream_lanes: int | None = None
    eval_chunk_size: int | None = None
    val_prefix_events: int | None = Field(default=None, ge=1)
    val_warmup_events: int = 0
    test_warmup_events: int = 0

    @model_validator(mode="after")
    def _validate_batch_mode(self) -> Self:
        validate_lane_config(self, val_split=self.source.val_split)
        return self

    def build_source(self) -> Any:
        return self.source.build(
            val_warmup_events=self.val_warmup_events,
            test_warmup_events=self.test_warmup_events,
        )

    def build(self, *, dataset: Any = None) -> Any:
        from graphids.core.data.datamodule.temporal import TemporalDataModule

        if representation_kind(self.source.representation_cfg) != "temporal":
            raise ValueError("temporal_dm requires representation_cfg.kind='temporal'")
        options = self.model_dump(exclude={"type", "source", "val_warmup_events", "test_warmup_events"})
        return TemporalDataModule(dataset=self.build_source() if dataset is None else dataset, **options)


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


def cross_vehicle(
    *, held_out_vehicle: str, val_fraction: float = 0.1, split_seed: int = 42,
    source_vehicle: str | None = None,
    representation_cfg: RepresentationCfg = _DEFAULT_REPRESENTATION_CFG,
) -> CrossVehicleCfg:
    return CrossVehicleCfg(
        held_out_vehicle=held_out_vehicle, source_vehicle=source_vehicle,
        val_fraction=val_fraction,
        split_seed=split_seed, representation_cfg=representation_cfg,
    )


def temporal_dm(
    *,
    source: CANBusCfg,
    **overrides: Any,
) -> TemporalDMCfg:
    return TemporalDMCfg(source=source, **overrides)
