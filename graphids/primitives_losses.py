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


LossFn = Annotated[FocalLossCfg | CELossCfg | WeightedCELossCfg, Field(discriminator="type")]
SimpleLossFn = LossFn


def focal(gamma: float = 2.0, reduction: str = "mean") -> FocalLossCfg:
    return FocalLossCfg(gamma=gamma, reduction=reduction)


def ce(reduction: str = "mean") -> CELossCfg:
    return CELossCfg(reduction=reduction)


def weighted_ce(
    weights: list[float] | tuple[float, ...] = (1.0, 5.0), reduction: str = "mean"
) -> WeightedCELossCfg:
    return WeightedCELossCfg(weights=tuple(weights), reduction=reduction)
