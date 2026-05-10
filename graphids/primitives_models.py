"""Model primitive configs and factories.

This is the model half of the public primitive API. It stays root-level so
plan authors and launch code don't need the old ``graphids.plan`` namespace.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from graphids.core.models.id_encoding import IdEncodingCfg


class _Cfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class GATCfg(_Cfg):
    type: Literal["gat"] = "gat"
    scale: Literal["small", "large"] = "small"
    id_encoder_cfg: IdEncodingCfg | None = None
    id_encoder_class_path: str | None = None
    id_encoder_kwargs: dict[str, Any] = Field(default_factory=dict)

    def build(self, *, loss_fn: Any = None) -> Any:
        from graphids.core.models.supervised.gat import GAT

        kwargs: dict[str, Any] = {"loss_fn": loss_fn, "scale": self.scale}
        if self.id_encoder_cfg is not None:
            kwargs["id_encoder_cfg"] = self.id_encoder_cfg
        if self.id_encoder_class_path is not None:
            kwargs["id_encoder_class_path"] = self.id_encoder_class_path
        if self.id_encoder_kwargs:
            kwargs["id_encoder_kwargs"] = self.id_encoder_kwargs
        return GAT(**kwargs)


class VGAECfg(_Cfg):
    type: Literal["vgae"] = "vgae"
    scale: Literal["small", "large"] = "small"
    id_encoder_cfg: IdEncodingCfg | None = None
    id_encoder_class_path: str | None = None
    id_encoder_kwargs: dict[str, Any] = Field(default_factory=dict)

    def build(self, *, loss_fn: Any = None) -> Any:
        from graphids.core.models.autoencoder.vgae import VGAE

        kwargs: dict[str, Any] = {"loss_fn": loss_fn, "scale": self.scale}
        if self.id_encoder_cfg is not None:
            kwargs["id_encoder_cfg"] = self.id_encoder_cfg
        if self.id_encoder_class_path is not None:
            kwargs["id_encoder_class_path"] = self.id_encoder_class_path
        if self.id_encoder_kwargs:
            kwargs["id_encoder_kwargs"] = self.id_encoder_kwargs
        return VGAE(**kwargs)


class DGICfg(_Cfg):
    type: Literal["dgi"] = "dgi"
    scale: Literal["small", "large"] = "small"
    id_encoder_cfg: IdEncodingCfg | None = None
    id_encoder_class_path: str | None = None
    id_encoder_kwargs: dict[str, Any] = Field(default_factory=dict)

    def build(self, *, loss_fn: Any = None) -> Any:
        from graphids.core.models.autoencoder.dgi import DGI

        kwargs: dict[str, Any] = {"scale": self.scale}
        if self.id_encoder_cfg is not None:
            kwargs["id_encoder_cfg"] = self.id_encoder_cfg
        if self.id_encoder_class_path is not None:
            kwargs["id_encoder_class_path"] = self.id_encoder_class_path
        if self.id_encoder_kwargs:
            kwargs["id_encoder_kwargs"] = self.id_encoder_kwargs
        return DGI(**kwargs)


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


def gat(
    scale: str = "small",
    *,
    id_encoder_cfg: IdEncodingCfg | None = None,
    id_encoder_class_path: str | None = None,
    id_encoder_kwargs: dict[str, Any] | None = None,
) -> GATCfg:
    return GATCfg(
        scale=scale,
        id_encoder_cfg=id_encoder_cfg,
        id_encoder_class_path=id_encoder_class_path,
        id_encoder_kwargs=id_encoder_kwargs or {},
    )


def vgae(
    scale: str = "small",
    *,
    id_encoder_cfg: IdEncodingCfg | None = None,
    id_encoder_class_path: str | None = None,
    id_encoder_kwargs: dict[str, Any] | None = None,
) -> VGAECfg:
    return VGAECfg(
        scale=scale,
        id_encoder_cfg=id_encoder_cfg,
        id_encoder_class_path=id_encoder_class_path,
        id_encoder_kwargs=id_encoder_kwargs or {},
    )


def dgi(
    scale: str = "small",
    *,
    id_encoder_cfg: IdEncodingCfg | None = None,
    id_encoder_class_path: str | None = None,
    id_encoder_kwargs: dict[str, Any] | None = None,
) -> DGICfg:
    return DGICfg(
        scale=scale,
        id_encoder_cfg=id_encoder_cfg,
        id_encoder_class_path=id_encoder_class_path,
        id_encoder_kwargs=id_encoder_kwargs or {},
    )


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
