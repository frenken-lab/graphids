"""Dagster Component: SLURM-based ML training pipeline.

Reads pipeline topology from pipeline.yaml, sweep recipe from ablation.yaml.
Generates one dagster asset per unique (stage, identity_hash) pair.
Each asset submits a SLURM job and polls for completion.

NO torch/Lightning imports — only YAML + path computation at definition time.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import dagster as dg
import yaml

from graphids.config import (
    CATALOG_PATH,
    CONFIG_DIR,
    STAGE_DEPENDENCIES,
    STAGE_MODEL_MAP,
    _CKPT_MODEL,
    compute_identity_hash,
    run_dir,
)
from graphids.orchestrate.resources import get_failure_reactions, get_resources, scale_resources
from graphids.orchestrate.slurm import generate_script, poll, submit

STAGES_DIR = CONFIG_DIR / "stages"
OVERLAYS_DIR = CONFIG_DIR / "overlays"
RECIPE_PATH = CONFIG_DIR / "ablation.yaml"

# Upstream stage → CLI flag for passing its checkpoint to downstream
_CKPT_CLI_FLAGS = {
    "autoencoder": "--data.init_args.vgae_ckpt_path",
    "curriculum": "--data.init_args.gat_ckpt_path",
    "normal": "--data.init_args.gat_ckpt_path",
}

# Recipe key → identity key name (where names differ)
_RECIPE_TO_IDENTITY = {"fusion_method": "method"}


# ---------------------------------------------------------------------------
# Convention-based helpers (replace _stage_args if/elif chain)
# ---------------------------------------------------------------------------


def _cli_val(v: Any) -> str:
    """Convert Python value to CLI-safe string (booleans lowercase for jsonargparse)."""
    return str(v).lower() if isinstance(v, bool) else str(v)


def _overlay_model(stage_def: dict, merged: dict) -> str:
    """Which model name to use for overlay file lookup.

    Recipe model_type (e.g. "dgi") only overrides the stage model for
    unsupervised stages — it selects the unsupervised method variant.
    Other stages (supervised, fusion) always use their pipeline.yaml model.
    """
    scale = merged.get("scale", "small")
    if "model_type" in merged and stage_def.get("learning_type") == "unsupervised":
        if (OVERLAYS_DIR / f"{scale}_{merged['model_type']}.yaml").exists():
            return merged["model_type"]
    return stage_def["model"]


def _resolve_config_files(stage: str, pipeline: dict, merged: dict) -> tuple[list[str], bool]:
    """Convention-based config file resolution.

    Returns (config_files, has_scale_overlay).

    Convention:
    1. stages/{stage}.yaml (always)
    2. overlays/{scale}_{model}.yaml (if file exists — fusion has none)
    3. overlays/kd_{model}.yaml (if auxiliaries in recipe and file exists)
    """
    stage_def = pipeline["stages"][stage]
    scale = merged.get("scale", "small")
    model = _overlay_model(stage_def, merged)

    configs: list[str] = [str(STAGES_DIR / f"{stage}.yaml")]

    overlay = OVERLAYS_DIR / f"{scale}_{model}.yaml"
    has_overlay = overlay.exists()
    if has_overlay:
        configs.append(str(overlay))

    if "auxiliaries" in merged:
        kd_overlay = OVERLAYS_DIR / f"kd_{model}.yaml"
        if kd_overlay.exists():
            configs.append(str(kd_overlay))

    return configs, has_overlay


def _identity_value(key: str, merged: dict, stages: list[str]) -> Any:
    """Resolve an identity key's value from merged recipe config."""
    if key == "gat_stage":
        return "curriculum" if "curriculum" in stages else "normal"
    for recipe_key, id_key in _RECIPE_TO_IDENTITY.items():
        if id_key == key and recipe_key in merged:
            return merged[recipe_key]
    return merged.get(key)


def _build_identity_cfg(
    stage: str, pipeline: dict, merged: dict, stages: list[str],
) -> dict:
    """Build identity config dict from pipeline.yaml identity_keys."""
    identity_keys = pipeline["stages"][stage].get("identity_keys", [])
    return {k: _identity_value(k, merged, stages) for k in identity_keys}


def _build_model_overrides(
    stage: str, pipeline: dict, id_cfg: dict, has_overlay: bool,
) -> dict[str, str]:
    """Build CLI --model.init_args overrides from identity config.

    All identity keys become overrides EXCEPT 'scale' when an overlay
    handles it. No per-stage if/elif needed.
    """
    identity_keys = pipeline["stages"][stage].get("identity_keys", [])
    overrides: dict[str, str] = {}
    for key in identity_keys:
        if key == "scale" and has_overlay:
            continue
        value = id_cfg.get(key)
        if value is not None:
            overrides[key] = _cli_val(value)
    return overrides


# ---------------------------------------------------------------------------
# Asset topology enumeration
# ---------------------------------------------------------------------------


def enumerate_assets(
    pipeline: dict, recipe: dict,
) -> dict[str, dict]:
    """Enumerate unique assets from pipeline topology × recipe configs.

    Returns dict: asset_name → {stage, model_type, scale, config_files,
    model_overrides, identity, kd_tag, deps}.
    """
    defaults = recipe.get("defaults", {})
    default_stages = defaults.get("stages", ["autoencoder", "curriculum", "fusion"])

    assets: dict[str, dict] = {}
    config_chains: dict[str, dict[str, str]] = {}

    for config_name, config_overrides in recipe["configs"].items():
        merged = {**defaults, **(config_overrides or {})}
        stages = merged.get("stages", default_stages)
        has_kd = "auxiliaries" in merged

        for stage in stages:
            if stage not in pipeline["stages"]:
                continue

            config_files, has_overlay = _resolve_config_files(stage, pipeline, merged)
            id_cfg = _build_identity_cfg(stage, pipeline, merged, stages)
            identity = compute_identity_hash(stage, id_cfg)
            kd_tag = "_kd" if has_kd else ""
            asset_name = f"{stage}{identity}{kd_tag}"

            if asset_name not in assets:
                model_overrides = _build_model_overrides(
                    stage, pipeline, id_cfg, has_overlay,
                )
                model_dir = _CKPT_MODEL.get(
                    STAGE_MODEL_MAP[stage], STAGE_MODEL_MAP[stage],
                )
                assets[asset_name] = {
                    "stage": stage,
                    "model_type": model_dir,
                    "scale": merged.get("scale", "small"),
                    "config_files": config_files,
                    "model_overrides": model_overrides,
                    "identity": identity,
                    "kd_tag": kd_tag,
                }

            config_chains.setdefault(config_name, {})[stage] = asset_name

    # --- Same-pipeline deps from STAGE_DEPENDENCIES ---
    deps_map: dict[str, set[str]] = defaultdict(set)
    for _config_name, stage_map in config_chains.items():
        for stage, asset_name in stage_map.items():
            for _dep_model, dep_stage in STAGE_DEPENDENCIES.get(stage, []):
                if dep_stage in stage_map:
                    deps_map[asset_name].add(stage_map[dep_stage])

    # --- KD cross-pipeline deps ---
    _kd_stages = {"autoencoder", "curriculum", "normal"}
    for config_name, stage_map in config_chains.items():
        merged = {**defaults, **(recipe["configs"][config_name] or {})}
        if "auxiliaries" not in merged:
            continue
        teacher_scale = merged["auxiliaries"][0].get("teacher_scale")
        if not teacher_scale:
            continue
        for tc_name, tc_stages in config_chains.items():
            tc_merged = {**defaults, **(recipe["configs"][tc_name] or {})}
            if tc_merged.get("scale") == teacher_scale and "auxiliaries" not in tc_merged:
                for stage, kd_asset in stage_map.items():
                    if stage in _kd_stages and "_kd" in kd_asset and stage in tc_stages:
                        deps_map[kd_asset].add(tc_stages[stage])
                break

    for name in assets:
        assets[name]["deps"] = sorted(deps_map.get(name, set()))

    return assets


# ---------------------------------------------------------------------------
# Component
# ---------------------------------------------------------------------------


class SlurmTrainingComponent(dg.Component, dg.Model, dg.Resolvable):
    """SLURM-based ML training pipeline.

    Reads pipeline topology from pipeline.yaml, sweep recipe from ablation.yaml.
    Generates one dagster asset per unique (stage, identity_hash) pair.
    Each asset submits a SLURM job via sbatch and polls for completion.
    """

    lake_root: str = "experimentruns"
    user: str = "unknown"
    dry_run: bool = False

    def build_defs(self, context: dg.ComponentLoadContext) -> dg.Definitions:
        pipeline = yaml.safe_load((CONFIG_DIR / "pipeline.yaml").read_text())
        recipe = yaml.safe_load(RECIPE_PATH.read_text())

        # Partitions: all datasets × seeds from recipe
        datasets = [
            k for k in yaml.safe_load(CATALOG_PATH.read_text())
            if not k.startswith("_")
        ]
        seeds = [str(s) for s in recipe.get("sweep", {}).get("seeds", [42])]
        partitions = dg.MultiPartitionsDefinition({
            "dataset": dg.StaticPartitionsDefinition(datasets),
            "seed": dg.StaticPartitionsDefinition(seeds),
        })

        assets_info = enumerate_assets(pipeline, recipe)
        failure_reactions = get_failure_reactions()

        result: list[dg.AssetsDefinition] = []
        for asset_name, info in assets_info.items():
            dep_infos = self._build_dep_infos(info, assets_info)
            result.append(self._make_asset(
                asset_name, info, dep_infos, partitions, failure_reactions,
            ))

        return dg.Definitions(
            assets=result,
            executor=dg.multiprocess_executor.configured({"max_concurrent": 8}),
        )

    @staticmethod
    def _build_dep_infos(info: dict, assets_info: dict) -> list[dict]:
        """Pre-compute upstream dependency info for checkpoint resolution."""
        dep_infos = []
        for dep_name in info["deps"]:
            dep = assets_info[dep_name]
            dep_infos.append({
                "stage": dep["stage"],
                "model_type": dep["model_type"],
                "scale": dep["scale"],
                "identity": dep["identity"],
                "kd_tag": dep["kd_tag"],
                "is_kd_teacher": dep["stage"] == info["stage"],
            })
        return dep_infos

    def _make_asset(
        self,
        name: str,
        info: dict,
        dep_infos: list[dict],
        partitions_def: dg.MultiPartitionsDefinition,
        failure_reactions: dict,
    ) -> dg.AssetsDefinition:
        """Create one @dg.asset for a unique pipeline stage."""
        # Capture in closure via local vars
        stage = info["stage"]
        model_type = info["model_type"]
        scale = info["scale"]
        config_files = info["config_files"]
        model_overrides = info["model_overrides"]
        identity = info["identity"]
        kd_tag = info["kd_tag"]
        lake_root = self.lake_root
        user = self.user
        dry_run = self.dry_run

        @dg.asset(
            name=name,
            deps=[dg.AssetKey(d) for d in info["deps"]],
            partitions_def=partitions_def,
            group_name=stage,
        )
        def _asset(context) -> dg.MaterializeResult:
            keys = context.partition_key
            if not isinstance(keys, dg.MultiPartitionKey):
                raise RuntimeError(
                    f"Expected MultiPartitionKey, got {type(keys).__name__}")
            dataset = keys.keys_by_dimension["dataset"]
            seed = int(keys.keys_by_dimension["seed"])

            rd = run_dir(lake_root, user, dataset, model_type, scale,
                         stage, identity, kd_tag, seed)
            rd_path = Path(rd)

            # Skip if already complete
            if (rd_path / "checkpoints" / "best_model.ckpt").exists():
                context.log.info(f"Skipping {name} — already complete")
                metrics_file = rd_path / "metrics.json"
                metrics = (json.loads(metrics_file.read_text())
                           if metrics_file.exists() else {})
                return dg.MaterializeResult(metadata=metrics)

            # Upstream checkpoint CLI overrides
            ckpt_overrides: list[str] = []
            for dep in dep_infos:
                dep_rd = run_dir(
                    lake_root, user, dataset, dep["model_type"],
                    dep["scale"], dep["stage"], dep["identity"],
                    dep["kd_tag"], seed,
                )
                ckpt = f"{dep_rd}/checkpoints/best_model.ckpt"
                if dep["is_kd_teacher"]:
                    ckpt_overrides.append(
                        f"--model.init_args.auxiliaries="
                        f"[{{model_path: {ckpt}}}]")
                elif dep["stage"] in _CKPT_CLI_FLAGS:
                    ckpt_overrides.append(
                        f"{_CKPT_CLI_FLAGS[dep['stage']]}={ckpt}")

            # Build full CLI args
            cli_overrides = [
                f"--data.init_args.dataset={dataset}",
                f"--seed_everything={seed}",
                f"--trainer.default_root_dir={rd}",
            ]
            for k, v in model_overrides.items():
                cli_overrides.append(f"--model.init_args.{k}={v}")
            cli_overrides.extend(ckpt_overrides)

            # Resource lookup + adaptive retry
            resources = get_resources(model_type, scale, stage)
            if context.retry_number > 0:
                for reason in ("OUT_OF_MEMORY", "TIMEOUT"):
                    resources = scale_resources(resources, reason)

            # Resume from last.ckpt
            ckpt_path = rd_path / "checkpoints" / "last.ckpt"
            script = generate_script(
                config_files, resources,
                ckpt_path=ckpt_path if ckpt_path.exists() else None,
                cli_overrides=cli_overrides,
            )
            job_id = submit(
                script, resources,
                job_name=f"{name}_{dataset}_s{seed}", dry_run=dry_run,
            )

            if dry_run:
                return dg.MaterializeResult(
                    metadata={"dry_run": True, "job_name": name})

            # Poll + handle failure
            state = poll(job_id)
            if state != "COMPLETED":
                reaction = failure_reactions.get(state, {})
                if reaction.get("max_retries", 0) > 0:
                    raise dg.RetryRequested(
                        max_retries=reaction["max_retries"],
                        seconds_to_wait=30,
                    )
                raise RuntimeError(f"SLURM job {job_id} failed: {state}")

            metrics_file = rd_path / "metrics.json"
            metrics = (json.loads(metrics_file.read_text())
                       if metrics_file.exists() else {})
            return dg.MaterializeResult(metadata=metrics)

        return _asset
