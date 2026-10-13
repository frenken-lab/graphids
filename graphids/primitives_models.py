"""Typed configuration for the shared graph-Mamba detector."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Cfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class VehicleGraphMambaJointCfg(_Cfg):
    type: Literal["vehicle_graph_mamba_joint"] = "vehicle_graph_mamba_joint"
    width: int = Field(default=128, ge=1)
    layers: int = Field(default=2, ge=1)
    d_state: int = Field(default=16, ge=1)
    d_conv: Literal[2, 3, 4] = 4
    expand: int = Field(default=2, ge=1)
    dropout: float = Field(default=0.1, ge=0, lt=1)
    backend: Literal["official", "reference"] = "official"
    graph_enabled: bool = True
    classification: bool = True
    forecasting: bool = True
    learning_rate: float = Field(default=0.001, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    loss_weights: dict[str, float] = Field(default_factory=dict)
    alert_quantile: float = Field(default=0.999, gt=0, lt=1)
    schema_version: Literal[1] = 1

    @model_validator(mode="after")
    def validate_objectives(self):
        import math

        if not (self.classification or self.forecasting):
            raise ValueError("at least one detection objective must be enabled")
        for name, value in self.loss_weights.items():
            if (
                name not in {"classification", "count", "change", "new_id"}
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(
                    "loss_weights must name classification/count/change/new_id and be finite and nonnegative"
                )
        return self

    def build(self, *, loss_fn=None):
        if loss_fn is not None:
            raise ValueError(
                "graph-Mamba defines separate classification/forecast losses; use loss_weights"
            )
        from graphids.core.models.graph_mamba import VehicleGraphMambaJoint

        return VehicleGraphMambaJoint(**self.model_dump(exclude={"type"}))


def vehicle_graph_mamba_joint(**kwargs) -> VehicleGraphMambaJointCfg:
    return VehicleGraphMambaJointCfg(**kwargs)


ModelCfg = Annotated[VehicleGraphMambaJointCfg, Field(discriminator="type")]
