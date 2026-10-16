"""Typed configuration for the shared graph-Mamba detector."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Cfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EventGraphMambaCfg(_Cfg):
    type: Literal["event_graph_mamba"] = "event_graph_mamba"
    width: int = Field(default=128, ge=1)
    layers: int = Field(default=2, ge=1)
    d_state: int = Field(default=16, ge=1)
    d_conv: Literal[2, 3, 4] = 4
    expand: int = Field(default=2, ge=1)
    dropout: float = Field(default=0.1, ge=0, lt=1)
    backend: Literal["official", "reference"] = "official"
    learning_rate: float = Field(default=0.001, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    payload_weight: float = Field(default=1.0, ge=0, allow_inf_nan=False)
    node_capacity: int = Field(default=512, ge=1)
    max_graph_tokens: int = Field(default=1048576, ge=1)
    schema_version: Literal[1] = 1

    def build(self, *, loss_fn=None):
        if loss_fn is not None:
            raise ValueError(
                "event_graph_mamba defines frame BCE and normal-only payload likelihood"
            )
        from graphids.core.models.event_graph_mamba import EventGraphMamba

        return EventGraphMamba(**self.model_dump(exclude={"type"}))


def event_graph_mamba(**kwargs) -> EventGraphMambaCfg:
    return EventGraphMambaCfg(**kwargs)


ModelCfg = EventGraphMambaCfg
