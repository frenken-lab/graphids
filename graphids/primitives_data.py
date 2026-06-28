"""Data primitive configs and dataset registry helpers."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

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
    representation_cfg: RepresentationCfg = Field(default_factory=TemporalRepresentationCfg)


class TemporalDMCfg(_Cfg):
    type: Literal["temporal_dm"] = "temporal_dm"
    source: CANBusCfg
    batch_size: int = 256
    val_warmup_events: int = 0
    test_warmup_events: int = 0

    def build(self) -> Any:
        from graphids.core.data.datamodule.temporal import TemporalDataModule
        from graphids.core.data.datasets.can_bus import CANBusTemporalSource

        if representation_kind(self.source.representation_cfg) != "temporal":
            raise ValueError("temporal_dm requires representation_cfg.kind='temporal'")
        source = CANBusTemporalSource(
            name=self.source.name,
            val_fraction=self.source.val_fraction,
            representation_cfg=self.source.representation_cfg,
            val_warmup_events=self.val_warmup_events,
            test_warmup_events=self.test_warmup_events,
        )
        return TemporalDataModule(dataset=source, batch_size=self.batch_size)


DataCfg = Annotated[TemporalDMCfg, Field(discriminator="type")]


def can_bus(
    *,
    dataset: str,
    seed: int,
    val_fraction: float = 0.2,
    representation_cfg: RepresentationCfg = _DEFAULT_REPRESENTATION_CFG,
) -> CANBusCfg:
    registry = load_catalog()
    if dataset not in registry:
        raise ValueError(f"unknown dataset: {dataset} (registry: {', '.join(sorted(registry))})")
    return CANBusCfg(
        name=dataset,
        seed=seed,
        val_fraction=val_fraction,
        representation_cfg=representation_cfg,
    )


def temporal_dm(
    *,
    source: CANBusCfg,
    **overrides: Any,
) -> TemporalDMCfg:
    return TemporalDMCfg(source=source, **overrides)
