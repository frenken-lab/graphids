"""Data primitive configs, dataset registry helpers, and reward dicts."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from graphids.core.data.preprocessing.representations import (
    GraphRepresentationCfg,
    SnapshotRepresentationCfg,
    representation_window_defaults,
)
from graphids.core.data.preprocessing.scaler import (
    RobustBenignScalerCfg,
    ScalerCfg,
    ZBenignScalerCfg,
)
from graphids.paths import load_catalog, trial_dir

_DEFAULT_SCALER_CFG = ZBenignScalerCfg()
_DEFAULT_REPRESENTATION_CFG = SnapshotRepresentationCfg()


class _Cfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ScoreRandomCfg(_Cfg):
    type: Literal["score_random"] = "score_random"
    seed: int = 0

    def build(self) -> Any:
        from graphids.core.data.preprocessing.curriculum import score_random

        return partial(score_random, seed=self.seed)


class ScoreVGAECfg(_Cfg):
    type: Literal["score_vgae"] = "score_vgae"
    ckpt_path: str

    def build(self) -> Any:
        from graphids.core.data.preprocessing.curriculum import score_vgae

        return partial(score_vgae, ckpt_path=self.ckpt_path)


DifficultyCfg = Annotated[
    ScoreRandomCfg | ScoreVGAECfg,
    Field(discriminator="type"),
]


class CANBusCfg(_Cfg):
    type: Literal["can_bus"] = "can_bus"
    name: str
    seed: int
    val_fraction: float = 0.2
    scaler_cfg: ScalerCfg = Field(default_factory=ZBenignScalerCfg)
    representation_cfg: GraphRepresentationCfg = Field(default_factory=SnapshotRepresentationCfg)

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
    difficulty: DifficultyCfg | None = None
    scope_label: int = 0
    min_steps_per_epoch: int = 1
    require_cache: bool = False

    def build(self) -> Any:
        from graphids.core.data.datamodule.graph import GraphDataModule
        from graphids.core.data.datasets.can_bus import CANBusSource
        source = CANBusSource(
            name=self.source.name,
            seed=self.source.seed,
            val_fraction=self.source.val_fraction,
            scaler_cfg=self.source.scaler_cfg,
            representation_cfg=self.source.representation_cfg,
        )
        difficulty = self.difficulty.build() if self.difficulty is not None else None
        return GraphDataModule(
            dataset=source,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            dynamic_batching=self.dynamic_batching,
            label_filter=self.label_filter,
            difficulty=difficulty,
            scope_label=self.scope_label,
            min_steps_per_epoch=self.min_steps_per_epoch,
            require_cache=self.require_cache,
        )


class CanonicalEntityCfg(_Cfg):
    type: Literal["canonical_entity"] = "canonical_entity"
    canonical_id: str
    name: str
    aliases: tuple[str, ...] = ()
    vehicle_aliases: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    kind: Literal["signal", "message", "state", "entity"] = "signal"
    description: str | None = None


class CanonicalRegistryCfg(_Cfg):
    type: Literal["canonical_registry"] = "canonical_registry"
    entities: tuple[CanonicalEntityCfg, ...]

    def build(self) -> Any:
        from graphids.core.data.discovery.canonical import (
            CanonicalEntitySpec,
            CanonicalRegistry,
        )

        return CanonicalRegistry(
            entities=tuple(
                CanonicalEntitySpec(
                    canonical_id=e.canonical_id,
                    name=e.name,
                    aliases=e.aliases,
                    vehicle_aliases=e.vehicle_aliases,
                    kind=e.kind,
                    description=e.description,
                )
                for e in self.entities
            )
        )


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
    GraphDMCfg | FusionDMCfg,
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


def score_random(seed: int = 0) -> ScoreRandomCfg:
    return ScoreRandomCfg(seed=seed)


def score_vgae(ckpt_path: str) -> ScoreVGAECfg:
    return ScoreVGAECfg(ckpt_path=ckpt_path)


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
    difficulty: DifficultyCfg | None = None,
    scope_label: int = 0,
    require_cache: bool = False,
    **overrides: Any,
) -> GraphDMCfg:
    return GraphDMCfg(
        source=source,
        label_filter=label_filter,
        difficulty=difficulty,
        scope_label=scope_label,
        require_cache=require_cache,
        **overrides,
    )


def canonical_entity(
    *,
    canonical_id: str,
    name: str,
    aliases: tuple[str, ...] = (),
    vehicle_aliases: dict[str, tuple[str, ...]] | None = None,
    kind: Literal["signal", "message", "state", "entity"] = "signal",
    description: str | None = None,
) -> CanonicalEntityCfg:
    return CanonicalEntityCfg(
        canonical_id=canonical_id,
        name=name,
        aliases=aliases,
        vehicle_aliases=vehicle_aliases or {},
        kind=kind,
        description=description,
    )


def canonical_registry(*, entities: tuple[CanonicalEntityCfg, ...]) -> CanonicalRegistryCfg:
    return CanonicalRegistryCfg(entities=entities)


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
