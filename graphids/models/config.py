"""Typed configuration for the shared graph-Mamba detector."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from graphids import event_defaults as defaults


class EventGraphMambaConfig(BaseModel):
    """Validated settings shared by YAML construction and checkpoint loading."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    type: Literal["event_graph_mamba"] = "event_graph_mamba"
    width: int = Field(default=128, ge=1)
    layers: int = Field(default=2, ge=1)
    d_state: int = Field(default=16, ge=1)
    d_conv: Literal[2, 3, 4] = 4
    expand: int = Field(default=2, ge=1)
    dropout: float = Field(default=0.1, ge=0, lt=1)
    backend: Literal["official", "reference"] = "official"
    learning_rate: float = Field(default=0.001, ge=0)
    weight_decay: float = Field(default=0.01, ge=0)
    payload_weight: float = Field(default=1.0, ge=0, allow_inf_nan=False)
    positive_weight: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    node_capacity: int = Field(default=512, ge=1)
    max_graph_tokens: int = Field(default=defaults.MAX_GRAPH_TOKENS, ge=1)
    use_elapsed_time: bool = True
    timing_mode: Literal["raw", "normalized"] = "raw"
    use_id_gap: bool = False
    payload_context: Literal["none", "delta", "xor", "both"] = "none"
    graph_age: bool = False
    graph_age_scale: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    timing_min_support: int = Field(default=32, ge=1)
    schema_version: Literal[1, 2] = 1

    @model_validator(mode="after")
    def validate_feature_schema(self):
        changed = (
            not self.use_elapsed_time
            or self.timing_mode != "raw"
            or self.use_id_gap
            or self.payload_context != "none"
            or self.graph_age
        )
        if changed and self.schema_version != 2:
            raise ValueError(
                "local-context and timing ablations require model schema_version=2"
            )
        return self

    def build(self):
        from graphids.models.event_graph_mamba import EventGraphMamba

        return EventGraphMamba(**self.model_dump(exclude={"type"}))
