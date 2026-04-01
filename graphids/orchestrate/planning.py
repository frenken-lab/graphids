"""Pure planning logic for orchestrated stage assets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from graphids.config import STAGE_MODEL_MAP, TrainingRunConfig, compute_identity_hash, expand_recipe_configs
from graphids.core.contracts import TrainingContract

# Recipe key -> identity key (where names differ)
_RECIPE_TO_IDENTITY = {"fusion_method": "method"}


def _identity_value(key: str, merged: TrainingRunConfig | dict, stages: list[str]) -> Any:
    if key == "gat_stage":
        return "curriculum" if "curriculum" in stages else "normal"
    _get = merged.get if isinstance(merged, dict) else lambda k, d=None: getattr(merged, k, d)
    for rk, ik in _RECIPE_TO_IDENTITY.items():
        if ik == key:
            return _get(rk)
    return _get(key)


@dataclass(frozen=True)
class StageConfig:
    """Training config for one asset. Pure data, no dagster dependency."""

    asset_name: str
    stage: str
    model_type: str
    scale: str
    config_files: tuple[str, ...] = ()
    model_init_overrides: dict[str, Any] = field(default_factory=dict)
    identity: str = ""
    kd_tag: str = ""
    resource_model: str = ""  # model key for resource lookup (fusion method for fusion stages)
    upstream_asset_names: tuple[str, ...] = ()
    upstream_model_families: dict[str, str] = field(default_factory=dict)


def enumerate_assets(pipeline: dict, recipe: dict) -> list[StageConfig]:
    """Two-pass enumeration: compute canonical keys, then build configs with resolved deps."""
    recipe = expand_recipe_configs(recipe)
    default_cfg = TrainingRunConfig(**recipe.get("defaults", {}))
    stages_def = pipeline["stages"]

    # Pass 1: canonical key map
    canonical: dict[tuple[str, str], str] = {}
    config_stages: dict[str, dict[str, str]] = {}

    for config_name, config_overrides in recipe["configs"].items():
        merged = default_cfg.merge(config_overrides or {})
        stages = list(merged.stages)
        has_kd = bool(merged.auxiliaries)
        config_stages[config_name] = {}

        for stage in stages:
            if stage not in stages_def:
                continue
            stage_def = stages_def[stage]
            id_keys = stage_def.get("identity_keys", [])
            id_cfg = {k: _identity_value(k, merged, stages) for k in id_keys}
            identity = compute_identity_hash(stage, id_cfg)
            kd_tag = "_kd" if has_kd else ""
            asset_name = f"{stage}{identity}{kd_tag}"

            dedup_key = (stage, f"{identity}{kd_tag}")
            if dedup_key not in canonical:
                canonical[dedup_key] = asset_name
            config_stages[config_name][stage] = canonical[dedup_key]

    # Pass 2: build StageConfigs
    built: dict[str, StageConfig] = {}

    for config_name, config_overrides in recipe["configs"].items():
        merged = default_cfg.merge(config_overrides or {})
        stages = list(merged.stages)
        has_kd = bool(merged.auxiliaries)
        stage_map = config_stages[config_name]

        for stage in stages:
            if stage not in stages_def:
                continue
            asset_name = stage_map.get(stage)
            if not asset_name or asset_name in built:
                continue

            stage_def = stages_def[stage]
            id_keys = stage_def.get("identity_keys", [])
            id_cfg = {k: _identity_value(k, merged, stages) for k in id_keys}
            identity = compute_identity_hash(stage, id_cfg)
            model_family = STAGE_MODEL_MAP[stage]
            if merged.model_type is not None and stage_def.get("learning_type") == "unsupervised":
                model_family = merged.model_type
            config_files = TrainingContract.resolve_config_files(
                stage,
                merged.scale,
                model_family=model_family,
                fusion_method=merged.fusion_method,
                include_kd_overlay=bool(merged.auxiliaries),
            )

            model_keys = set(stage_def.get("model_keys", id_keys))
            overrides: dict[str, str] = {}
            for key in id_keys:
                if key not in model_keys:
                    continue
                if key == "scale":
                    continue
                if key == "method" and stage == "fusion":
                    continue
                val = id_cfg.get(key)
                if val is not None:
                    overrides[key] = str(val).lower() if isinstance(val, bool) else str(val)

            upstream_names: list[str] = []
            upstream_models: dict[str, str] = {}
            seen_models: set[str] = set()
            for dep in stage_def.get("depends_on", []):
                dep_model = dep["model"]
                dep_asset = stage_map.get(dep["stage"])
                if not dep_asset or dep_model in seen_models:
                    continue
                seen_models.add(dep_model)
                upstream_names.append(dep_asset)
                upstream_models[dep_asset] = dep_model

            if has_kd:
                teacher_scale = merged.auxiliaries[0].teacher_scale if merged.auxiliaries else None
                if teacher_scale:
                    for tc_name, tc_overrides in recipe["configs"].items():
                        tc_merged = default_cfg.merge(tc_overrides or {})
                        if tc_merged.scale == teacher_scale and not tc_merged.auxiliaries:
                            tc_map = config_stages.get(tc_name, {})
                            for s in ("autoencoder", "curriculum", "normal"):
                                if s == stage and s in tc_map:
                                    teacher_asset = tc_map[s]
                                    upstream_names.append(teacher_asset)
                                    upstream_models[teacher_asset] = STAGE_MODEL_MAP[s]
                            break

            model_dir = model_family
            res_model = merged.fusion_method if stage == "fusion" else model_family

            built[asset_name] = StageConfig(
                asset_name=asset_name,
                stage=stage,
                model_type=model_dir,
                scale=merged.scale,
                config_files=tuple(config_files),
                model_init_overrides=overrides,
                identity=identity,
                kd_tag="_kd" if has_kd else "",
                resource_model=res_model,
                upstream_asset_names=tuple(sorted(set(upstream_names))),
                upstream_model_families=upstream_models,
            )

    return list(built.values())
