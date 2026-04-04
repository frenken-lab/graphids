"""Recipe envelope expansion for compact config recipes."""

from __future__ import annotations

import itertools
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .contracts import KDEntry


class _SweepSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_family: str
    stage: str
    scale: str | list[str] = "small"
    fusion_method: str | list[str] | None = None
    model_overrides: dict[str, Any] = Field(default_factory=dict)
    kd: KDEntry | None = None


class _SelectionSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    datasets: list[str] = Field(default_factory=list)
    model_families: list[str] = Field(default_factory=list)
    scales: list[str] = Field(default_factory=list)
    stages: dict[str, list[str]] = Field(default_factory=dict)
    fusion_methods: list[str] = Field(default_factory=list)


class _RecipeEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recipe: dict[str, Any] = Field(default_factory=dict)
    seeds: list[int] = Field(default_factory=list)
    overrides: dict[str, Any] = Field(default_factory=dict)
    selection: _SelectionSpec | None = None
    sweeps: list[_SweepSpec] = Field(default_factory=list)
    trainer_overrides: dict[str, Any] = Field(default_factory=dict)
    stage_overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)
    resource_overrides: dict[str, Any] = Field(default_factory=dict)

    @field_validator("stage_overrides")
    @classmethod
    def _valid_stage_names(cls, v: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        from .topology import STAGES

        bad = [s for s in v if s not in STAGES]
        if bad:
            raise ValueError(f"Unknown stages in stage_overrides: {bad}. Valid: {sorted(STAGES)}")
        return v


def _flatten_dict(d: dict[str, Any], prefix: str = "") -> dict[str, str]:
    """Flatten nested dict to dotted-key CLI strings.

    >>> _flatten_dict({"max_epochs": 2}, "trainer")
    {"trainer.max_epochs": "2"}
    """
    out: dict[str, str] = {}
    for k, v in d.items():
        full = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten_dict(v, full))
        elif isinstance(v, (str, int, float, bool, type(None))):
            out[full] = str(v).lower() if isinstance(v, bool) else str(v)
        else:
            raise TypeError(
                f"Non-scalar value for override key {full!r}: {type(v).__name__}={v!r}. "
                "Only scalars (str, int, float, bool) are supported in trainer_overrides."
            )
    return out


def _stage_chain(stage: str) -> tuple[str, ...]:
    if stage == "fusion":
        return ("autoencoder", "curriculum", "normal", "fusion")
    if stage == "curriculum":
        return ("autoencoder", "curriculum")
    return (stage,)


def _expand_sweep(sweep: _SweepSpec) -> list[tuple[str, dict[str, Any]]]:
    scales = sweep.scale if isinstance(sweep.scale, list) else [sweep.scale]
    methods = (sweep.fusion_method if isinstance(sweep.fusion_method, list) else [sweep.fusion_method]) if sweep.fusion_method is not None else [None]

    init_args = (sweep.model_overrides or {}).get("init_args", {})
    axis_keys: list[str] = []
    axis_vals: list[list[Any]] = []
    for key, value in init_args.items():
        if isinstance(value, list):
            axis_keys.append(key)
            axis_vals.append(value)
        else:
            axis_keys.append(key)
            axis_vals.append([value])

    expanded: list[tuple[str, dict[str, Any]]] = []
    for scale, method, init_combo in itertools.product(
        scales,
        methods,
        itertools.product(*axis_vals) if axis_vals else [()],
    ):
        over: dict[str, Any] = {"scale": scale, "stages": list(_stage_chain(sweep.stage))}
        if sweep.model_family != "fusion":
            over["model_type"] = sweep.model_family
        if method:
            over["fusion_method"] = method
        if sweep.kd is not None:
            kd_payload = sweep.kd.model_dump(exclude_none=True)
            over["auxiliaries"] = [kd_payload]
        for key, value in zip(axis_keys, init_combo):
            if key in {"conv_type", "loss_fn", "variational"}:
                over[key] = value

        suffix = "_".join(f"{k}-{over[k]}" for k in sorted(over) if k != "stages")
        name = f"{sweep.model_family}_{sweep.stage}_{suffix}".replace("/", "_")
        expanded.append((name, over))

    return expanded


def expand_recipe_configs(
    raw_recipe: dict[str, Any],
    *,
    valid_scales: set[str] | frozenset[str],
    valid_fusion_methods: set[str] | frozenset[str],
) -> dict[str, Any]:
    """Normalize new recipe formats to orchestrator format."""
    envelope = _RecipeEnvelope(**raw_recipe)

    defaults: dict[str, Any] = dict(envelope.overrides)
    configs: dict[str, dict[str, Any]] = {}

    if envelope.sweeps:
        for sweep in envelope.sweeps:
            for name, over in _expand_sweep(sweep):
                configs[name] = over

    if envelope.selection is not None:
        sel = envelope.selection
        scales = sel.scales or list(valid_scales)
        methods = sel.fusion_methods or list(valid_fusion_methods)
        for family in sel.model_families:
            for stage in sel.stages.get(family, []):
                for scale in scales:
                    if family == "fusion":
                        for method in methods:
                            name = f"{family}_{stage}_{scale}_{method}"
                            configs[name] = {
                                "stages": list(_stage_chain(stage)),
                                "scale": scale,
                                "fusion_method": method,
                            }
                    else:
                        name = f"{family}_{stage}_{scale}"
                        configs[name] = {
                            "stages": list(_stage_chain(stage)),
                            "scale": scale,
                            "model_type": family,
                        }

    if not configs:
        raise ValueError(
            "Recipe contains no runnable configs after expansion. "
            "Provide at least one sweep or selection block."
        )

    seed_list = envelope.seeds if envelope.seeds else [42]
    return {
        "defaults": defaults,
        "configs": configs,
        "sweep": {"seeds": seed_list},
        "trainer_overrides": _flatten_dict(envelope.trainer_overrides),
        "stage_overrides": {
            stage: _flatten_dict(overrides)
            for stage, overrides in envelope.stage_overrides.items()
        },
        "resource_overrides": dict(envelope.resource_overrides),
    }
