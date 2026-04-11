"""Recipe expansion + asset enumeration — Layer 1 of the orchestrate stack.

Pure planning: ``PipelineConfig`` or recipe dict → ``list[StageConfig]``.
No torch imports, no jsonnet subprocess at module scope, no side effects.

Two entry points:

- ``build_pipeline_stages(cfg)`` — single-config path for ``pipeline-run``.
  Skips the recipe/sweep machinery entirely.
- ``enumerate_assets(recipe)`` — multi-config path for recipe sweeps.
  Handles KD teacher cross-config resolution.
"""

from __future__ import annotations

from typing import Any

from graphids.config.constants import CONFIG_DIR, PROJECT_ROOT, VALID_FUSION_METHODS, VALID_SCALES
from graphids.config.topology import TOPOLOGY
from graphids.orchestrate.config import PipelineConfig, StageConfig, TrainingRunConfig

_STAGES_DIR = PROJECT_ROOT / "configs" / "stages"
_STAGE_JSONNET: dict[str, str] = {s: f"{s}.jsonnet" for s in TOPOLOGY.stages}


def resolve_jsonnet_path(stage: str) -> str:
    """Return the absolute path to the jsonnet file for a stage."""
    filename = _STAGE_JSONNET.get(stage)
    if filename is None:
        raise ValueError(
            f"No jsonnet stage file for stage={stage!r}. Known: {sorted(_STAGE_JSONNET)}"
        )
    return str(_STAGES_DIR / filename)


def _resolve_upstream(
    stage: str,
    stage_to_asset: dict[str, str],
) -> tuple[list[str], dict[str, str]]:
    """Resolve upstream ``(asset_names, asset→family)`` for ``stage``.

    Walks ``stage_def.depends_on`` and returns one entry per distinct
    family (i.e. multiple deps on the same family collapse to the first).
    Shared by the pipeline-run path and the recipe-sweep path.
    """
    upstream_names: list[str] = []
    upstream_models: dict[str, str] = {}
    seen_families: set[str] = set()
    for dep in TOPOLOGY.stages[stage].depends_on:
        dep_asset = stage_to_asset.get(dep["stage"])
        if dep_asset and dep["family"] not in seen_families:
            seen_families.add(dep["family"])
            upstream_names.append(dep_asset)
            upstream_models[dep_asset] = dep["family"]
    return upstream_names, upstream_models


def build_pipeline_stages(config: PipelineConfig) -> list[StageConfig]:
    """``PipelineConfig → list[StageConfig]`` — single-config fast path.

    Skips the recipe dict + enumerate_assets round-trip. For recipe
    sweeps (not yet a CLI entry point), use ``enumerate_assets``.
    """
    training_run = config.to_training_run()
    trainer_overrides = dict(config.tla_overrides)
    stage_to_asset: dict[str, str] = {}
    stages: list[StageConfig] = []

    for stage in training_run.stages:
        if stage not in TOPOLOGY.stages:
            continue
        upstream_names, upstream_models = _resolve_upstream(stage, stage_to_asset)
        cfg = StageConfig.for_stage(
            stage,
            training_run,
            upstream_names=upstream_names,
            upstream_models=upstream_models,
            trainer_overrides=trainer_overrides,
        )
        stage_to_asset[stage] = cfg.asset_name
        stages.append(cfg)

    stage_order = {s: i for i, s in enumerate(config.stages)}
    stages.sort(key=lambda c: stage_order.get(c.stage, 99))
    return stages


def enumerate_assets(recipe: dict) -> list[StageConfig]:
    """Enumerate unique training assets from an expanded recipe.

    Multi-config planner for recipe sweeps. Single pass builds
    ``StageConfig``s with topology deps; KD teacher deps are wired in a
    post-pass that requires all asset names to be known first.
    """
    default_cfg = TrainingRunConfig(**recipe.get("defaults", {}))
    trainer_overrides = recipe.get("trainer_overrides", {})
    stage_overrides_map = recipe.get("stage_overrides", {})
    resource_overrides = recipe.get("resource_overrides", {})

    config_stages: dict[str, dict[str, str]] = {}  # config_name → stage → asset_name
    built: dict[str, StageConfig] = {}
    kd_deferred: list[tuple[str, str, tuple]] = []

    for config_name, overrides in recipe["configs"].items():
        merged = default_cfg.merge(overrides or {})
        stage_to_asset: dict[str, str] = {}
        config_stages[config_name] = stage_to_asset

        for stage in merged.stages:
            if stage not in TOPOLOGY.stages:
                continue
            upstream_names, upstream_models = _resolve_upstream(stage, stage_to_asset)
            cfg = StageConfig.for_stage(
                stage,
                merged,
                upstream_names=upstream_names,
                upstream_models=upstream_models,
                trainer_overrides=trainer_overrides,
                stage_overrides=stage_overrides_map.get(stage, {}),
                resource_overrides=resource_overrides,
            )
            stage_to_asset[stage] = cfg.asset_name
            if cfg.asset_name in built:
                continue
            built[cfg.asset_name] = cfg

            if merged.auxiliaries:
                kd_deferred.append((cfg.asset_name, stage, merged.auxiliaries))

    # KD teacher post-pass: requires all asset names known first.
    for asset_name, stage, auxiliaries in kd_deferred:
        cfg = built[asset_name]
        upstream = list(cfg.upstream_asset_names)
        models = dict(cfg.upstream_model_families)
        for aux in auxiliaries:
            teacher_asset = config_stages.get(aux.teacher_config or "", {}).get(stage)
            if teacher_asset is None:
                raise ValueError(
                    f"KD '{asset_name}': teacher_config='{aux.teacher_config}' "
                    f"has no '{stage}' asset. Check recipe configs."
                )
            if teacher_asset not in upstream:
                upstream.append(teacher_asset)
                models[teacher_asset] = TOPOLOGY.stage_family_map[stage]
        built[asset_name] = cfg.model_copy(
            update={
                "upstream_asset_names": tuple(sorted(set(upstream))),
                "upstream_model_families": models,
            }
        )

    return list(built.values())


def expand_recipe_configs(raw_recipe: dict[str, Any]) -> dict[str, Any]:
    """Expand a raw recipe dict to an orchestrator-ready config list.

    Jsonnet expansion (``configs/recipes/_expand.jsonnet``) handles
    sweep/selection cartesian products, override flattening, and defaults.
    """
    from graphids.config.jsonnet import render

    return render(
        CONFIG_DIR / "recipes" / "_expand.jsonnet",
        tla={
            "recipe": raw_recipe,
            "valid_scales": sorted(VALID_SCALES),
            "valid_fusion_methods": sorted(VALID_FUSION_METHODS),
        },
    )
