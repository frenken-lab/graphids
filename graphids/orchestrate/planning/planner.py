"""Pure planning logic for orchestrated stage assets."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from graphids.config.topology import TOPOLOGY, compute_identity_hash
from graphids.orchestrate.contracts import resolve_jsonnet_path
from graphids.orchestrate.planning.recipes import TrainingRunConfig

_UNSUPERVISED_MODELS = frozenset({"vgae", "dgi"})


class StageConfig(BaseModel):
    """Training config for one asset. Pure data, no torch/Lightning imports."""

    model_config = ConfigDict(frozen=True)

    asset_name: str
    stage: str
    model_type: str
    scale: str
    jsonnet_path: str = ""
    model_init_overrides: dict[str, Any] = Field(default_factory=dict)
    identity: str = ""
    kd_tag: str = ""
    resource_model: str = ""  # model key for resource lookup (fusion method for fusion stages)
    kd_overrides: dict[str, Any] = Field(default_factory=dict)
    trainer_overrides: dict[str, Any] = Field(default_factory=dict)
    stage_overrides: dict[str, Any] = Field(default_factory=dict)
    resource_overrides: dict[str, str | int] = Field(default_factory=dict)
    upstream_asset_names: tuple[str, ...] = ()
    upstream_model_families: dict[str, str] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict (Monarch endpoint args must be serializable)."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StageConfig:
        """Reconstruct from a dict (inverse of ``to_dict``)."""
        return cls.model_validate(d)

    @classmethod
    def from_recipe(
        cls,
        *,
        asset_name: str,
        stage: str,
        merged: TrainingRunConfig,
        recipe: dict[str, Any],
        upstream_names: list[str],
        upstream_models: dict[str, str],
    ) -> StageConfig:
        """Build a StageConfig from merged recipe config + topology."""
        stage_def = TOPOLOGY.stages[stage]

        model_type = stage_def.family
        if (
            merged.model_type is not None
            and stage_def.learning_type == "unsupervised"
            and merged.model_type in _UNSUPERVISED_MODELS
        ):
            model_type = merged.model_type

        id_cfg = merged.identity_for(stage)
        accepted = set(stage_def.stage_tlas)

        return cls(
            asset_name=asset_name,
            stage=stage,
            model_type=model_type,
            scale=merged.scale,
            jsonnet_path=resolve_jsonnet_path(stage),
            model_init_overrides={
                k: str(v).lower() if isinstance(v, bool) else str(v)
                for k, v in id_cfg.items()
                if v is not None and k in accepted
            },
            identity=compute_identity_hash(stage, id_cfg),
            kd_tag="_kd" if merged.auxiliaries else "",
            resource_model=merged.fusion_method if stage == "fusion" else model_type,
            kd_overrides=(
                merged.auxiliaries[0].model_dump(exclude_none=True) if merged.auxiliaries else {}
            ),
            trainer_overrides=recipe.get("trainer_overrides", {}),
            stage_overrides=recipe.get("stage_overrides", {}).get(stage, {}),
            resource_overrides=recipe.get("resource_overrides", {}),
            upstream_asset_names=tuple(sorted(upstream_names)),
            upstream_model_families=upstream_models,
        )


def enumerate_assets(recipe: dict) -> list[StageConfig]:
    """Enumerate unique training assets from an expanded recipe.

    Single pass builds StageConfigs with topology deps. KD teacher
    deps are wired in a post-pass (requires all asset names known).
    """
    default_cfg = TrainingRunConfig(**recipe.get("defaults", {}))

    config_stages: dict[str, dict[str, str]] = {}  # config_name → stage → asset_name
    built: dict[str, StageConfig] = {}
    kd_deferred: list[tuple[str, str, tuple]] = []

    for config_name, overrides in recipe["configs"].items():
        merged = default_cfg.merge(overrides or {})
        config_stages[config_name] = {}

        for stage in merged.stages:
            if stage not in TOPOLOGY.stages:
                continue

            asset_name = merged.asset_key(stage)
            config_stages[config_name][stage] = asset_name
            if asset_name in built:
                continue

            # Topology deps (stages are in topo order, so earlier stages are resolved)
            stage_def = TOPOLOGY.stages[stage]
            upstream_names: list[str] = []
            upstream_models: dict[str, str] = {}
            seen_families: set[str] = set()
            for dep in stage_def.depends_on:
                dep_asset = config_stages[config_name].get(dep["stage"])
                if dep_asset and dep["family"] not in seen_families:
                    seen_families.add(dep["family"])
                    upstream_names.append(dep_asset)
                    upstream_models[dep_asset] = dep["family"]

            built[asset_name] = StageConfig.from_recipe(
                asset_name=asset_name,
                stage=stage,
                merged=merged,
                recipe=recipe,
                upstream_names=upstream_names,
                upstream_models=upstream_models,
            )

            if merged.auxiliaries:
                kd_deferred.append((asset_name, stage, merged.auxiliaries))

    # Post-pass: wire KD teacher checkpoints as upstream deps
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
