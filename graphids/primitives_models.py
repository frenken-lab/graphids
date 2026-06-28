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


ModelCfg = Annotated[
    TemporalEventClassifierCfg
    | TemporalGATCfg
    | TemporalVGAECfg,
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
