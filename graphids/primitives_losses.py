"""Loss primitive configs and factories."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _Cfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


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


class SoftLabelDistillationCfg(_Cfg):
    type: Literal["soft_label_distillation"] = "soft_label_distillation"
    base_loss: SimpleLossFn
    teacher_model: Any
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
    type: Literal["feature_distillation"] = "feature_distillation"
    base_loss: SimpleLossFn
    teacher_model: Any
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
    | SoftLabelDistillationCfg
    | FeatureDistillationCfg,
    Field(discriminator="type"),
]


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


def soft_label_distillation(
    base_loss: SimpleLossFn,
    teacher_model: Any,
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
    teacher_model: Any,
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
