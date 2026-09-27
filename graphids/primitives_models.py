"""Model primitive configs and factories.

This is the model half of the public primitive API. It stays root-level so
plan authors and launch code don't need the old ``graphids.plan`` namespace.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
    use_source: bool = True
    use_destination: bool = True
    time_encoding_dim: int = 0


class TemporalHybridBackboneCfg(_Cfg):
    type: Literal["none", "gru", "ssm_lite", "mamba"] = "gru"
    layers: int | None = None
    dropout: float = 0.1


class TemporalHybridHeadsCfg(_Cfg):
    classification: bool | None = None
    next_id: bool | None = None
    iat: bool | None = None
    payload_delta: bool | None = None


class TemporalHybridAnomalyCfg(_Cfg):
    mode: Literal["regression", "nll"] = "regression"
    min_log_scale: float = -7.0
    max_log_scale: float = 5.0


class TemporalHybridRhythmCfg(_Cfg):
    enabled: bool = False


class TemporalHybridMotifCfg(_Cfg):
    enabled: bool = False
    length: int = 3
    embedding_dim: int | None = None
    time_dim: int | None = None


class TemporalHybridCfg(_Cfg):
    type: Literal["temporal_hybrid"] = "temporal_hybrid"
    scale: Literal["small", "large"] = "small"
    objective: Literal["supervised", "anomaly", "joint"] = "supervised"
    input: TemporalHybridInputCfg = Field(default_factory=TemporalHybridInputCfg)
    memory: TemporalHybridMemoryCfg = Field(default_factory=TemporalHybridMemoryCfg)
    backbone: TemporalHybridBackboneCfg = Field(default_factory=TemporalHybridBackboneCfg)
    heads: TemporalHybridHeadsCfg = Field(default_factory=TemporalHybridHeadsCfg)
    anomaly: TemporalHybridAnomalyCfg = Field(default_factory=TemporalHybridAnomalyCfg)
    rhythm: TemporalHybridRhythmCfg = Field(default_factory=TemporalHybridRhythmCfg)
    motif: TemporalHybridMotifCfg = Field(default_factory=TemporalHybridMotifCfg)
    loss_weights: dict[str, float] = Field(default_factory=dict)
    anomaly_score_weights: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_objective_heads(self) -> Self:
        default_ssl = self.objective in {"anomaly", "joint"}
        classification = (
            self.objective in {"supervised", "joint"}
            if self.heads.classification is None
            else self.heads.classification
        )
        anomaly_heads = {
            "next_id": default_ssl if self.heads.next_id is None else self.heads.next_id,
            "iat": default_ssl if self.heads.iat is None else self.heads.iat,
            "payload_delta": (
                default_ssl if self.heads.payload_delta is None else self.heads.payload_delta
            ),
        }
        has_anomaly_head = any(anomaly_heads.values())

        if self.objective == "supervised" and not classification:
            raise ValueError("objective='supervised' requires heads.classification=true")
        if self.objective == "anomaly":
            if classification:
                raise ValueError("objective='anomaly' requires heads.classification=false")
            if not has_anomaly_head:
                raise ValueError("objective='anomaly' requires at least one anomaly head")
        if self.objective == "joint":
            if not classification:
                raise ValueError("objective='joint' requires heads.classification=true")
            if not has_anomaly_head:
                raise ValueError("objective='joint' requires at least one anomaly head")
        if self.memory.time_encoding_dim < 0:
            raise ValueError("memory.time_encoding_dim must be non-negative")
        if self.anomaly.min_log_scale > self.anomaly.max_log_scale:
            raise ValueError("anomaly.min_log_scale must be <= anomaly.max_log_scale")
        if self.motif.length < 1:
            raise ValueError("motif.length must be positive")
        if self.motif.embedding_dim is not None and self.motif.embedding_dim < 1:
            raise ValueError("motif.embedding_dim must be positive")
        if self.motif.time_dim is not None and self.motif.time_dim < 1:
            raise ValueError("motif.time_dim must be positive")
        return self

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
            anomaly=self.anomaly.model_dump(),
            rhythm=self.rhythm.model_dump(),
            motif=self.motif.model_dump(exclude_none=True),
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
    anomaly: dict[str, Any] | TemporalHybridAnomalyCfg | None = None,
    rhythm: dict[str, Any] | TemporalHybridRhythmCfg | None = None,
    motif: dict[str, Any] | TemporalHybridMotifCfg | None = None,
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
        anomaly=TemporalHybridAnomalyCfg.model_validate(anomaly or {}),
        rhythm=TemporalHybridRhythmCfg.model_validate(rhythm or {}),
        motif=TemporalHybridMotifCfg.model_validate(motif or {}),
        loss_weights=dict(loss_weights or {}),
        anomaly_score_weights=dict(anomaly_score_weights or {}),
    )
