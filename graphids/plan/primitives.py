"""Typed configs and factory functions for plans.

Config classes (bottom) are frozen Pydantic discriminated-union members.
Factory functions (top) return those configs — the only public API plan
authors need.

Each config class carries a ``.build()`` method that instantiates the
corresponding runtime object. Imports inside ``.build()`` are lazy so
importing this module on the login node (render time) never pulls in torch.

Adding a new variant: add a class + update the union alias + implement
``.build()`` — no changes needed in ``orchestrate.py``.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from graphids.paths import load_catalog
from graphids.paths import states_dir as _states_dir

# ── Base ─────────────────────────────────────────────────────────────────────


class _Cfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# ── Model configs ─────────────────────────────────────────────────────────────


class GATCfg(_Cfg):
    type: Literal["gat"] = "gat"
    scale: Literal["small", "large"] = "small"
    id_encoder_class_path: str | None = None
    id_encoder_kwargs: dict[str, Any] = Field(default_factory=dict)

    def build(self, *, loss_fn: Any = None) -> Any:
        from graphids.core.models.supervised.gat import GAT

        kwargs: dict[str, Any] = {"loss_fn": loss_fn, "scale": self.scale}
        if self.id_encoder_class_path is not None:
            kwargs["id_encoder_class_path"] = self.id_encoder_class_path
        if self.id_encoder_kwargs:
            kwargs["id_encoder_kwargs"] = self.id_encoder_kwargs
        return GAT(**kwargs)


class VGAECfg(_Cfg):
    type: Literal["vgae"] = "vgae"
    scale: Literal["small", "large"] = "small"

    def build(self, *, loss_fn: Any = None) -> Any:
        from graphids.core.models.autoencoder.vgae import VGAE

        return VGAE(loss_fn=loss_fn, scale=self.scale)


class DGICfg(_Cfg):
    type: Literal["dgi"] = "dgi"
    scale: Literal["small", "large"] = "small"

    def build(self, *, loss_fn: Any = None) -> Any:
        from graphids.core.models.autoencoder.dgi import DGI

        return DGI(scale=self.scale)


class BanditCfg(_Cfg):
    type: Literal["bandit"] = "bandit"
    state_dim: int
    reward_kwargs: dict[str, Any] = Field(default_factory=dict)

    def build(self, *, loss_fn: Any = None) -> Any:
        from graphids.core.models.fusion.bandit import BanditFusionModule

        return BanditFusionModule(
            state_dim=self.state_dim,
            reward_kwargs=self.reward_kwargs or None,
        )


class DQNCfg(_Cfg):
    type: Literal["dqn"] = "dqn"
    state_dim: int
    reward_kwargs: dict[str, Any] = Field(default_factory=dict)

    def build(self, *, loss_fn: Any = None) -> Any:
        from graphids.core.models.fusion.dqn import DQNFusionModule

        return DQNFusionModule(
            state_dim=self.state_dim,
            reward_kwargs=self.reward_kwargs or None,
        )


class MLPFusionCfg(_Cfg):
    type: Literal["mlp_fusion"] = "mlp_fusion"
    state_dim: int

    def build(self, *, loss_fn: Any = None) -> Any:
        from graphids.core.models.fusion.mlp import MLPFusionModule

        return MLPFusionModule(state_dim=self.state_dim)


class MoECfg(_Cfg):
    type: Literal["moe"] = "moe"
    state_dim: int
    aux_weight: float = 0.01

    def build(self, *, loss_fn: Any = None) -> Any:
        from graphids.core.models.fusion.moe import MoEFusionModule

        return MoEFusionModule(state_dim=self.state_dim, aux_weight=self.aux_weight)


class WeightedAvgCfg(_Cfg):
    type: Literal["weighted_avg"] = "weighted_avg"
    state_dim: int

    def build(self, *, loss_fn: Any = None) -> Any:
        from graphids.core.models.fusion.weighted_avg import WeightedAvgModule

        return WeightedAvgModule(state_dim=self.state_dim)


ModelCfg = Annotated[
    GATCfg | VGAECfg | DGICfg | BanditCfg | DQNCfg | MLPFusionCfg | MoECfg | WeightedAvgCfg,
    Field(discriminator="type"),
]


# ── Loss configs ──────────────────────────────────────────────────────────────


class FocalLossCfg(_Cfg):
    type: Literal["focal"] = "focal"
    gamma: float = 2.0
    reduction: str = "mean"

    def build(self) -> Any:
        from graphids.core.losses import FocalLoss

        return FocalLoss(gamma=self.gamma, reduction=self.reduction)


class CELossCfg(_Cfg):
    type: Literal["ce"] = "ce"
    reduction: str = "mean"

    def build(self) -> Any:
        from graphids.core.losses import CrossEntropyLoss

        return CrossEntropyLoss(reduction=self.reduction)


class WeightedCELossCfg(_Cfg):
    type: Literal["weighted_ce"] = "weighted_ce"
    weights: tuple[float, ...] = (1.0, 5.0)
    reduction: str = "mean"

    def build(self) -> Any:
        from graphids.core.losses import WeightedCrossEntropyLoss

        return WeightedCrossEntropyLoss(weights=list(self.weights), reduction=self.reduction)


class VGAETaskLossCfg(_Cfg):
    type: Literal["vgae_task"] = "vgae_task"
    kl_weight: float = 0.01
    canid_weight: float = 0.1
    nbr_weight: float = 0.1
    edge_weight: float = 0.1
    k_neg: int = 32

    def build(self) -> Any:
        from graphids.core.losses import VGAETaskLoss

        return VGAETaskLoss(
            kl_weight=self.kl_weight,
            canid_weight=self.canid_weight,
            nbr_weight=self.nbr_weight,
            edge_weight=self.edge_weight,
            k_neg=self.k_neg,
        )


SimpleLossFn = Annotated[
    FocalLossCfg | CELossCfg | WeightedCELossCfg | VGAETaskLossCfg,
    Field(discriminator="type"),
]


class LinearRampCfg(_Cfg):
    type: Literal["linear_ramp"] = "linear_ramp"
    start_ratio: float = 1.0
    end_ratio: float = 10.0
    max_epochs: int = 300


class CurriculumLossCfg(_Cfg):
    type: Literal["curriculum"] = "curriculum"
    base_loss: SimpleLossFn
    schedule: LinearRampCfg = Field(default_factory=LinearRampCfg)

    def build(self) -> Any:
        from graphids.core.losses import CurriculumWeightedLoss
        from graphids.core.losses.curriculum import LinearRampSchedule

        base = self.base_loss.model_copy(update={"reduction": "none"}).build()
        schedule = LinearRampSchedule(
            start_ratio=self.schedule.start_ratio,
            end_ratio=self.schedule.end_ratio,
            max_epochs=self.schedule.max_epochs,
        )
        return CurriculumWeightedLoss(base_loss=base, schedule=schedule)


class SoftLabelDistillationCfg(_Cfg):
    """Soft-label KD for supervised (GAT) student."""

    type: Literal["soft_label_distillation"] = "soft_label_distillation"
    base_loss: SimpleLossFn
    teacher_model: GATCfg
    teacher_ckpt_path: str
    temperature: float = 4.0
    alpha: float = 0.7

    def build(self) -> Any:
        from graphids.core.losses.distillation import SoftLabelDistillation

        return SoftLabelDistillation(
            base_loss=self.base_loss.build(),
            teacher_model=self.teacher_model.build(),
            teacher_ckpt_path=self.teacher_ckpt_path,
            temperature=self.temperature,
            alpha=self.alpha,
        )


class FeatureDistillationCfg(_Cfg):
    """Feature-matching KD for VGAE student."""

    type: Literal["feature_distillation"] = "feature_distillation"
    base_loss: SimpleLossFn
    teacher_model: VGAECfg
    teacher_ckpt_path: str
    latent_weight: float = 1.0
    recon_weight: float = 1.0
    alpha: float = 0.7
    projection_in_features: int | None = None
    projection_out_features: int | None = None

    def build(self) -> Any:
        import torch.nn

        from graphids.core.losses.distillation import FeatureDistillation

        projection = None
        if self.projection_in_features is not None and self.projection_out_features is not None:
            projection = torch.nn.Linear(self.projection_in_features, self.projection_out_features)
        return FeatureDistillation(
            base_loss=self.base_loss.build(),
            teacher_model=self.teacher_model.build(),
            teacher_ckpt_path=self.teacher_ckpt_path,
            latent_weight=self.latent_weight,
            recon_weight=self.recon_weight,
            alpha=self.alpha,
            projection=projection,
        )


LossFn = Annotated[
    FocalLossCfg
    | CELossCfg
    | WeightedCELossCfg
    | VGAETaskLossCfg
    | CurriculumLossCfg
    | SoftLabelDistillationCfg
    | FeatureDistillationCfg,
    Field(discriminator="type"),
]


# ── Difficulty scorer configs ─────────────────────────────────────────────────


class ScoreRandomCfg(_Cfg):
    type: Literal["score_random"] = "score_random"
    seed: int = 0

    def build(self) -> Any:
        from graphids.core.data.preprocessing.curriculum import ScoreRandom

        return ScoreRandom(seed=self.seed)


class ScoreVGAECfg(_Cfg):
    type: Literal["score_vgae"] = "score_vgae"
    ckpt_path: str

    def build(self) -> Any:
        from graphids.core.data.preprocessing.curriculum import ScoreVGAE

        return ScoreVGAE(ckpt_path=self.ckpt_path)


DifficultyCfg = Annotated[
    ScoreRandomCfg | ScoreVGAECfg,
    Field(discriminator="type"),
]


# ── Data configs ──────────────────────────────────────────────────────────────


class CANBusCfg(_Cfg):
    type: Literal["can_bus"] = "can_bus"
    name: str
    seed: int
    window_size: int = 100
    stride: int = 100
    val_fraction: float = 0.2


class GraphDMCfg(_Cfg):
    type: Literal["graph_dm"] = "graph_dm"
    source: CANBusCfg
    label_filter: str | None = None
    difficulty: DifficultyCfg | None = None
    scope_label: int = 0
    min_steps_per_epoch: int = 1

    def build(self) -> Any:
        from graphids.core.data.datamodule.graph import GraphDataModule
        from graphids.core.data.datasets.can_bus import CANBusSource

        source = CANBusSource(
            name=self.source.name,
            seed=self.source.seed,
            window_size=self.source.window_size,
            stride=self.source.stride,
            val_fraction=self.source.val_fraction,
        )
        difficulty = self.difficulty.build() if self.difficulty is not None else None
        return GraphDataModule(
            dataset=source,
            label_filter=self.label_filter,
            difficulty=difficulty,
            scope_label=self.scope_label,
            min_steps_per_epoch=self.min_steps_per_epoch,
        )


class FusionDMCfg(_Cfg):
    type: Literal["fusion_dm"] = "fusion_dm"
    cached_states_dir: str
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


# ── Reward shaping dicts (RL fusion only) ────────────────────────────────────

# Legacy 2-way base reward. Pre-2026-05-07 default. Diagnosed as cause of
# all-benign equilibrium — the agreement bonus rewards the majority-class rule.
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

# PBRS-compliant 4-way (TP/TN/FP/FN) base reward + attack-gated confidence bonus.
# FN cost = 4×FP encodes IDS F2-optimization (Davis & Goadrich 2006).
REWARD_MINIMAL: dict[str, Any] = {
    "mode": "minimal",
    "vgae_weights": [0.4, 0.3, 0.3],
    "tp_reward": 3.0,
    "tn_reward": 1.5,
    "fp_cost": -1.5,
    "fn_cost": -6.0,
    "confidence_weight": 0.3,
}


# ── Model factories ───────────────────────────────────────────────────────────


def gat(
    scale: str = "small",
    *,
    id_encoder_class_path: str | None = None,
    id_encoder_kwargs: dict[str, Any] | None = None,
) -> GATCfg:
    return GATCfg(
        scale=scale,
        id_encoder_class_path=id_encoder_class_path,
        id_encoder_kwargs=id_encoder_kwargs or {},
    )


def vgae(scale: str = "small") -> VGAECfg:
    return VGAECfg(scale=scale)


def dgi(scale: str = "small") -> DGICfg:
    return DGICfg(scale=scale)


def bandit(state_dim: int, *, reward_kwargs: dict[str, Any] | None = None) -> BanditCfg:
    return BanditCfg(state_dim=state_dim, reward_kwargs=reward_kwargs or {})


def dqn(state_dim: int, *, reward_kwargs: dict[str, Any] | None = None) -> DQNCfg:
    return DQNCfg(state_dim=state_dim, reward_kwargs=reward_kwargs or {})


def mlp_fusion(state_dim: int) -> MLPFusionCfg:
    return MLPFusionCfg(state_dim=state_dim)


def moe(state_dim: int, *, aux_weight: float = 0.01) -> MoECfg:
    return MoECfg(state_dim=state_dim, aux_weight=aux_weight)


def weighted_avg(state_dim: int) -> WeightedAvgCfg:
    return WeightedAvgCfg(state_dim=state_dim)


# ── Loss factories ────────────────────────────────────────────────────────────


def focal(gamma: float = 2.0, reduction: str = "mean") -> FocalLossCfg:
    return FocalLossCfg(gamma=gamma, reduction=reduction)


def ce(reduction: str = "mean") -> CELossCfg:
    return CELossCfg(reduction=reduction)


def weighted_ce(
    weights: list[float] | tuple[float, ...] = (1.0, 5.0), reduction: str = "mean"
) -> WeightedCELossCfg:
    return WeightedCELossCfg(weights=tuple(weights), reduction=reduction)


def vgae_task(
    kl_weight: float = 0.01,
    canid_weight: float = 0.1,
    nbr_weight: float = 0.1,
    edge_weight: float = 0.1,
    k_neg: int = 32,
) -> VGAETaskLossCfg:
    return VGAETaskLossCfg(
        kl_weight=kl_weight,
        canid_weight=canid_weight,
        nbr_weight=nbr_weight,
        edge_weight=edge_weight,
        k_neg=k_neg,
    )


def curriculum(
    base: SimpleLossFn,
    schedule: LinearRampCfg | None = None,
) -> CurriculumLossCfg:
    return CurriculumLossCfg(
        base_loss=base,
        schedule=schedule if schedule is not None else LinearRampCfg(),
    )


def soft_label_distillation(
    base_loss: SimpleLossFn,
    teacher_model: GATCfg,
    teacher_ckpt_path: str,
    *,
    temperature: float = 4.0,
    alpha: float = 0.7,
) -> SoftLabelDistillationCfg:
    return SoftLabelDistillationCfg(
        base_loss=base_loss,
        teacher_model=teacher_model,
        teacher_ckpt_path=teacher_ckpt_path,
        temperature=temperature,
        alpha=alpha,
    )


def feature_distillation(
    base_loss: SimpleLossFn,
    teacher_model: VGAECfg,
    teacher_ckpt_path: str,
    *,
    latent_weight: float = 1.0,
    recon_weight: float = 1.0,
    alpha: float = 0.7,
    projection_in_features: int | None = None,
    projection_out_features: int | None = None,
) -> FeatureDistillationCfg:
    return FeatureDistillationCfg(
        base_loss=base_loss,
        teacher_model=teacher_model,
        teacher_ckpt_path=teacher_ckpt_path,
        latent_weight=latent_weight,
        recon_weight=recon_weight,
        alpha=alpha,
        projection_in_features=projection_in_features,
        projection_out_features=projection_out_features,
    )


# ── Difficulty scorer factories ───────────────────────────────────────────────


def score_random(seed: int = 0) -> ScoreRandomCfg:
    return ScoreRandomCfg(seed=seed)


def score_vgae(ckpt_path: str) -> ScoreVGAECfg:
    return ScoreVGAECfg(ckpt_path=ckpt_path)


# ── Data factories ────────────────────────────────────────────────────────────


def can_bus(*, dataset: str, seed: int, **overrides: Any) -> CANBusCfg:
    """``CANBusSource`` config with registry validation at render time."""
    registry = load_catalog()
    if dataset not in registry:
        raise ValueError(f"unknown dataset: {dataset} (registry: {', '.join(sorted(registry))})")
    return CANBusCfg(name=dataset, seed=seed, **overrides)


def graph_dm(
    *,
    source: CANBusCfg,
    label_filter: str | None = None,
    difficulty: DifficultyCfg | None = None,
    scope_label: int = 0,
    **overrides: Any,
) -> GraphDMCfg:
    return GraphDMCfg(
        source=source,
        label_filter=label_filter,
        difficulty=difficulty,
        scope_label=scope_label,
        **overrides,
    )


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
        cached_states_dir=_states_dir(dataset, seed, states_variant),
        method=method,
        batch_size=batch_size,
        episode_sample_size=episode_sample_size,
    )
