"""Model primitive configs and factories.

This is the model half of the public primitive API. It stays root-level so
plan authors and launch code don't need the old ``graphids.plan`` namespace.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _Cfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TemporalEventClassifierCfg(_Cfg):
    type: Literal["temporal_event_classifier"] = "temporal_event_classifier"
    scale: Literal["small", "large"] = "small"
    hidden: int | None = None
    layers: int | None = None
    embedding_dim: int | None = None
    dropout: float = 0.2

    def build(self, *, loss_fn: Any = None) -> Any:
        from graphids.core.models.temporal import TemporalEventClassifier

        return TemporalEventClassifier(
            loss_fn=loss_fn,
            scale=self.scale,
            hidden=self.hidden,
            layers=self.layers,
            embedding_dim=self.embedding_dim,
            dropout=self.dropout,
        )


class TemporalGATCfg(_Cfg):
    type: Literal["temporal_gat"] = "temporal_gat"
    scale: Literal["small", "large"] = "small"
    hidden: int | None = None
    layers: int | None = None
    heads: int | None = None
    embedding_dim: int | None = None
    dropout: float = 0.2

    def build(self, *, loss_fn: Any = None) -> Any:
        from graphids.core.models.temporal import TemporalGAT

        return TemporalGAT(
            loss_fn=loss_fn,
            scale=self.scale,
            hidden=self.hidden,
            layers=self.layers,
            heads=self.heads,
            embedding_dim=self.embedding_dim,
            dropout=self.dropout,
        )


class TemporalRNNClassifierCfg(_Cfg):
    type: Literal["temporal_rnn_classifier"] = "temporal_rnn_classifier"
    scale: Literal["small", "large"] = "small"
    hidden: int | None = None
    layers: int | None = None
    embedding_dim: int | None = None
    dropout: float = 0.2

    def build(self, *, loss_fn: Any = None) -> Any:
        from graphids.core.models.temporal import TemporalRNNClassifier

        return TemporalRNNClassifier(
            loss_fn=loss_fn,
            scale=self.scale,
            hidden=self.hidden,
            layers=self.layers,
            embedding_dim=self.embedding_dim,
            dropout=self.dropout,
        )


class TemporalVGAECfg(_Cfg):
    type: Literal["temporal_vgae"] = "temporal_vgae"
    scale: Literal["small", "large"] = "small"
    hidden: int | None = None
    layers: int | None = None
    embedding_dim: int | None = None
    latent_dim: int | None = None
    dropout: float = 0.1
    kl_weight: float = 0.01

    def build(self, *, loss_fn: Any = None) -> Any:
        from graphids.core.models.temporal import TemporalVGAE

        del loss_fn
        return TemporalVGAE(
            scale=self.scale,
            hidden=self.hidden,
            layers=self.layers,
            embedding_dim=self.embedding_dim,
            latent_dim=self.latent_dim,
            dropout=self.dropout,
            kl_weight=self.kl_weight,
        )


class TemporalHybridInputCfg(_Cfg):
    embedding_dim: int | None = None
    hidden: int | None = None
    dropout: float = 0.1


class TemporalHybridMemoryCfg(_Cfg):
    type: Literal["tgn"] = "tgn"
    enabled: bool = True
    reset_on_stream_end: bool = True


class TemporalHybridBackboneCfg(_Cfg):
    type: Literal["none", "gru", "ssm_lite", "mamba"] = "gru"
    layers: int | None = None
    dropout: float = 0.1


class TemporalHybridHeadsCfg(_Cfg):
    classification: bool | None = None
    next_id: bool | None = None
    iat: bool | None = None
    payload_delta: bool | None = None


class TemporalHybridCfg(_Cfg):
    type: Literal["temporal_hybrid"] = "temporal_hybrid"
    scale: Literal["small", "large"] = "small"
    objective: Literal["supervised", "anomaly", "joint"] = "supervised"
    input: TemporalHybridInputCfg = Field(default_factory=TemporalHybridInputCfg)
    memory: TemporalHybridMemoryCfg = Field(default_factory=TemporalHybridMemoryCfg)
    backbone: TemporalHybridBackboneCfg = Field(default_factory=TemporalHybridBackboneCfg)
    heads: TemporalHybridHeadsCfg = Field(default_factory=TemporalHybridHeadsCfg)
    loss_weights: dict[str, float] = Field(default_factory=dict)
    anomaly_score_weights: dict[str, float] = Field(default_factory=dict)

    def build(self, *, loss_fn: Any = None) -> Any:
        from graphids.core.models.temporal import TemporalHybridModel

        return TemporalHybridModel(
            loss_fn=loss_fn,
            scale=self.scale,
            objective=self.objective,
            input=self.input.model_dump(exclude_none=True),
            memory=self.memory.model_dump(),
            backbone=self.backbone.model_dump(exclude_none=True),
            heads=self.heads.model_dump(exclude_none=True),
            loss_weights=dict(self.loss_weights),
            anomaly_score_weights=dict(self.anomaly_score_weights),
        )


ModelCfg = Annotated[
    TemporalEventClassifierCfg
    | TemporalGATCfg
    | TemporalRNNClassifierCfg
    | TemporalVGAECfg
    | TemporalHybridCfg,
    Field(discriminator="type"),
]


def temporal_event_classifier(
    scale: str = "small",
    *,
    hidden: int | None = None,
    layers: int | None = None,
    embedding_dim: int | None = None,
    dropout: float = 0.2,
) -> TemporalEventClassifierCfg:
    return TemporalEventClassifierCfg(
        scale=scale,
        hidden=hidden,
        layers=layers,
        embedding_dim=embedding_dim,
        dropout=dropout,
    )


def temporal_gat(
    scale: str = "small",
    *,
    hidden: int | None = None,
    layers: int | None = None,
    heads: int | None = None,
    embedding_dim: int | None = None,
    dropout: float = 0.2,
) -> TemporalGATCfg:
    return TemporalGATCfg(
        scale=scale,
        hidden=hidden,
        layers=layers,
        heads=heads,
        embedding_dim=embedding_dim,
        dropout=dropout,
    )


def temporal_rnn_classifier(
    scale: str = "small",
    *,
    hidden: int | None = None,
    layers: int | None = None,
    embedding_dim: int | None = None,
    dropout: float = 0.2,
) -> TemporalRNNClassifierCfg:
    return TemporalRNNClassifierCfg(
        scale=scale,
        hidden=hidden,
        layers=layers,
        embedding_dim=embedding_dim,
        dropout=dropout,
    )


def temporal_vgae(
    scale: str = "small",
    *,
    hidden: int | None = None,
    layers: int | None = None,
    embedding_dim: int | None = None,
    latent_dim: int | None = None,
    dropout: float = 0.1,
    kl_weight: float = 0.01,
) -> TemporalVGAECfg:
    return TemporalVGAECfg(
        scale=scale,
        hidden=hidden,
        layers=layers,
        embedding_dim=embedding_dim,
        latent_dim=latent_dim,
        dropout=dropout,
        kl_weight=kl_weight,
    )


def temporal_hybrid(
    scale: str = "small",
    *,
    objective: str = "supervised",
    input: dict[str, Any] | TemporalHybridInputCfg | None = None,
    memory: dict[str, Any] | TemporalHybridMemoryCfg | None = None,
    backbone: dict[str, Any] | TemporalHybridBackboneCfg | None = None,
    heads: dict[str, Any] | TemporalHybridHeadsCfg | None = None,
    loss_weights: dict[str, float] | None = None,
    anomaly_score_weights: dict[str, float] | None = None,
) -> TemporalHybridCfg:
    return TemporalHybridCfg(
        scale=scale,
        objective=objective,
        input=TemporalHybridInputCfg.model_validate(input or {}),
        memory=TemporalHybridMemoryCfg.model_validate(memory or {}),
        backbone=TemporalHybridBackboneCfg.model_validate(backbone or {}),
        heads=TemporalHybridHeadsCfg.model_validate(heads or {}),
        loss_weights=dict(loss_weights or {}),
        anomaly_score_weights=dict(anomaly_score_weights or {}),
    )
