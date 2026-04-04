"""Pure planning logic for orchestrated stage assets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from graphids.config import STAGE_MODEL_MAP, TrainingRunConfig, compute_identity_hash
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
    val = _get(key)
    # model_type=None means "use stage default" — resolve for stable identity hashes
    if key == "model_type" and val is None:
        return "vgae"
    return val


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
    kd_overrides: dict[str, Any] = field(default_factory=dict)  # raw KDEntry payload
    trainer_overrides: dict[str, str] = field(default_factory=dict)
    stage_overrides: dict[str, str] = field(default_factory=dict)
    resource_overrides: dict[str, str | int] = field(default_factory=dict)
    upstream_asset_names: tuple[str, ...] = ()
    upstream_model_families: dict[str, str] = field(default_factory=dict)


def _resolve_kd_teachers(
    *,
    student_config: str,
    stage: str,
    auxiliaries: tuple,
    recipe: dict,
    default_cfg: TrainingRunConfig,
    config_stages: dict[str, dict[str, str]],
    upstream_names: list[str],
    upstream_models: dict[str, str],
) -> None:
    """Wire KD teacher assets as upstream dependencies by explicit recipe config name.

    Each KD auxiliary must name its teacher via ``teacher_config`` (a key in
    ``recipe["configs"]``). We validate that the named config exists, trains
    without its own KD auxiliaries, and produces an asset for the student's
    current stage. All mismatches raise with the student name + stage + the
    set of valid alternatives so the failure is actionable.

    This replaces the legacy scale-based inference (iterate configs, first
    match wins) which silently rewired the student to different teachers
    when recipe key order changed. See ``docs/reference/orchestration-risks.md``
    item #2.
    """
    for idx, aux in enumerate(auxiliaries):
        teacher_config = getattr(aux, "teacher_config", None)
        if teacher_config is None:
            raise ValueError(
                f"KD student '{student_config}' (stage '{stage}', auxiliary "
                f"index {idx}): missing teacher_config. Explicitly name the "
                f"recipe config to use as teacher (e.g., "
                f"teacher_config: baseline_large). Silent scale-based "
                f"inference was removed — it depended on recipe key order."
            )
        if teacher_config not in recipe["configs"]:
            raise ValueError(
                f"KD student '{student_config}' (stage '{stage}'): "
                f"teacher_config='{teacher_config}' does not name a config "
                f"in this recipe. Available configs: "
                f"{sorted(recipe['configs'].keys())}"
            )
        tc_overrides = recipe["configs"][teacher_config]
        tc_merged = default_cfg.merge(tc_overrides or {})
        if tc_merged.auxiliaries:
            raise ValueError(
                f"KD student '{student_config}' (stage '{stage}'): "
                f"teacher_config='{teacher_config}' has its own auxiliaries "
                f"— teachers must train without KD."
            )
        tc_map = config_stages.get(teacher_config, {})
        if stage not in tc_map:
            raise ValueError(
                f"KD student '{student_config}' (stage '{stage}'): "
                f"teacher_config='{teacher_config}' does not produce a "
                f"'{stage}' asset. Teacher stages: {sorted(tc_map.keys())}"
            )
        teacher_asset = tc_map[stage]
        if teacher_asset not in upstream_names:
            upstream_names.append(teacher_asset)
            upstream_models[teacher_asset] = STAGE_MODEL_MAP[stage]


def enumerate_assets(pipeline: dict, recipe: dict) -> list[StageConfig]:
    """Two-pass enumeration: compute canonical keys, then build configs with resolved deps.

    Expects an already-expanded recipe (output of ``expand_recipe_configs``).
    """
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
            # Only override unsupervised model_family when model_type IS an
            # unsupervised model (vgae, dgi). A GAT curriculum sweep sets
            # model_type="gat" but the upstream autoencoder must stay VGAE.
            _UNSUPERVISED_MODELS = {"vgae", "dgi"}
            if (merged.model_type is not None
                    and stage_def.get("learning_type") == "unsupervised"
                    and merged.model_type in _UNSUPERVISED_MODELS):
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
                # variational is VGAE-only; DGI doesn't accept it
                if key == "variational" and model_family == "dgi":
                    continue
                val = id_cfg.get(key)
                if val is not None:
                    overrides[key] = str(val).lower() if isinstance(val, bool) else str(val)

            kd_payload: dict[str, Any] = {}
            if has_kd and merged.auxiliaries:
                kd_payload = merged.auxiliaries[0].model_dump(exclude_none=True)

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
                _resolve_kd_teachers(
                    student_config=config_name,
                    stage=stage,
                    auxiliaries=merged.auxiliaries,
                    recipe=recipe,
                    default_cfg=default_cfg,
                    config_stages=config_stages,
                    upstream_names=upstream_names,
                    upstream_models=upstream_models,
                )

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
                kd_overrides=kd_payload,
                trainer_overrides=recipe.get("trainer_overrides", {}),
                stage_overrides=recipe.get("stage_overrides", {}).get(stage, {}),
                resource_overrides=recipe.get("resource_overrides", {}),
                upstream_asset_names=tuple(sorted(set(upstream_names))),
                upstream_model_families=upstream_models,
            )

    return list(built.values())
