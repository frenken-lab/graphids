"""Recipe envelope expansion for compact config recipes."""

from __future__ import annotations

import itertools
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _KDSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str = "kd"
    alpha: float = 0.7
    teacher_scale: str = "large"
    temperature: float | None = None
    model_path: str | None = None
    vgae_latent_weight: float | None = None
    vgae_recon_weight: float | None = None


class _SweepSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_family: str
    stage: str
    scale: str | list[str] = "small"
    fusion_method: str | list[str] | None = None
    model_overrides: dict[str, Any] = Field(default_factory=dict)
    kd: _KDSpec | None = None


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


def _normalize_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return [value]


def _stage_chain(stage: str) -> tuple[str, ...]:
    if stage == "fusion":
        return ("autoencoder", "curriculum", "normal", "fusion")
    if stage == "curriculum":
        return ("autoencoder", "curriculum")
    return (stage,)


def _expand_sweep(sweep: _SweepSpec) -> list[tuple[str, dict[str, Any]]]:
    scales = _normalize_list(sweep.scale)
    methods = _normalize_list(sweep.fusion_method) if sweep.fusion_method is not None else [None]

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

    seed_list = envelope.seeds or raw_recipe.get("sweep", {}).get("seeds", [42])
    return {"defaults": defaults, "configs": configs, "sweep": {"seeds": seed_list}}
