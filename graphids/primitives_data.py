"""Data primitive configs, dataset registry helpers, and reward dicts."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from graphids.core.data.preprocessing.representations import (
    GraphRepresentationCfg,
    TemporalRepresentationCfg,
    representation_kind,
    representation_window_defaults,
)
from graphids.core.data.preprocessing.scaler import (
    RobustBenignScalerCfg,
    ScalerCfg,
    ZBenignScalerCfg,
)
from graphids.paths import load_catalog, trial_dir

_DEFAULT_SCALER_CFG = ZBenignScalerCfg()
_DEFAULT_REPRESENTATION_CFG = TemporalRepresentationCfg()


class _Cfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CANBusCfg(_Cfg):
    type: Literal["can_bus"] = "can_bus"
    name: str
    seed: int
    val_fraction: float = 0.2
    scaler_cfg: ScalerCfg = Field(default_factory=ZBenignScalerCfg)
    representation_cfg: GraphRepresentationCfg = Field(default_factory=TemporalRepresentationCfg)

    def resolved_window_size_stride(self) -> tuple[int, int]:
        return representation_window_defaults(self.representation_cfg)

    @property
    def window_size(self) -> int:
        return self.resolved_window_size_stride()[0]

    @property
    def stride(self) -> int:
        return self.resolved_window_size_stride()[1]


class GraphDMCfg(_Cfg):
    type: Literal["graph_dm"] = "graph_dm"
    source: CANBusCfg
    batch_size: int = 32
    num_workers: int | None = None
    prefetch_factor: int = 2
    dynamic_batching: bool = True
    label_filter: str | None = None
    min_steps_per_epoch: int = 1
    require_cache: bool = False

    def build(self) -> Any:
        if representation_kind(self.source.representation_cfg) == "temporal":
            raise ValueError("graph_dm requires a snapshot representation; use temporal_dm for temporal data")
        from graphids.core.data.datamodule.graph import GraphDataModule
        from graphids.core.data.datasets.can_bus import CANBusSource
        source = CANBusSource(
            name=self.source.name,
            val_fraction=self.source.val_fraction,
            scaler_cfg=self.source.scaler_cfg,
            representation_cfg=self.source.representation_cfg,
        )
        return GraphDataModule(
            dataset=source,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            dynamic_batching=self.dynamic_batching,
            label_filter=self.label_filter,
            min_steps_per_epoch=self.min_steps_per_epoch,
            require_cache=self.require_cache,
        )


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


class FusionDMCfg(_Cfg):
    type: Literal["fusion_dm"] = "fusion_dm"
    cached_states_dir: Path
    method: str
    batch_size: int = 128
    episode_sample_size: int = 20_000

    def build(self) -> Any:
        from graphids.core.data.datamodule.fusion import FusionDataModule

        return FusionDataModule(
            cached_states_dir=self.cached_states_dir,
            method=self.method,
            batch_size=self.batch_size,
            episode_sample_size=self.episode_sample_size,
        )


DataCfg = Annotated[
    TemporalDMCfg | GraphDMCfg | FusionDMCfg,
    Field(discriminator="type"),
]


REWARD: dict[str, Any] = {
    "vgae_weights": [0.4, 0.3, 0.3],
    "correct": 3.0,
    "incorrect": -3.0,
    "confidence_weight": 0.5,
    "combined_conf_weight": 0.3,
    "disagreement_penalty": -1.0,
    "overconf_penalty": -1.5,
    "balance_weight": 0.3,
}


def can_bus(
    *,
    dataset: str,
    seed: int,
    val_fraction: float = 0.2,
    scaler_cfg: ScalerCfg = _DEFAULT_SCALER_CFG,
    representation_cfg: GraphRepresentationCfg = _DEFAULT_REPRESENTATION_CFG,
) -> CANBusCfg:
    registry = load_catalog()
    if dataset not in registry:
        raise ValueError(f"unknown dataset: {dataset} (registry: {', '.join(sorted(registry))})")
    return CANBusCfg(
        name=dataset,
        seed=seed,
        val_fraction=val_fraction,
        scaler_cfg=scaler_cfg,
        representation_cfg=representation_cfg,
    )


def z_benign_scaler() -> ZBenignScalerCfg:
    return ZBenignScalerCfg()


def robust_benign_scaler() -> RobustBenignScalerCfg:
    return RobustBenignScalerCfg()


def graph_dm(
    *,
    source: CANBusCfg,
    label_filter: str | None = None,
    require_cache: bool = False,
    **overrides: Any,
) -> GraphDMCfg:
    return GraphDMCfg(
        source=source,
        label_filter=label_filter,
        require_cache=require_cache,
        **overrides,
    )


def temporal_dm(
    *,
    source: CANBusCfg,
    **overrides: Any,
) -> TemporalDMCfg:
    return TemporalDMCfg(source=source, **overrides)


def fusion_dm(
    *,
    dataset: str,
    seed: int,
    method: str,
    batch_size: int = 128,
    episode_sample_size: int = 20_000,
    states_variant: str = "default",
) -> FusionDMCfg:
    return FusionDMCfg(
        cached_states_dir=trial_dir() / "cached_states" / dataset / states_variant / f"seed_{int(seed)}",
        method=method,
        batch_size=batch_size,
        episode_sample_size=episode_sample_size,
    )
