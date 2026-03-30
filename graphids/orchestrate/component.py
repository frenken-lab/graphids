"""Dagster Component: SLURM-based ML training pipeline.

Assets represent trained model checkpoints. AssetSpecs describe identity
(key, deps, tags, kinds). Multi-asset functions define materialization
behavior (submit to SLURM, return checkpoint path). IOManager handles
checkpoint path handoff between dependent stages.

NO torch/Lightning imports at definition time.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import dagster as dg
import yaml

from graphids.config import (
    CATALOG_PATH,
    CONFIG_DIR,
    LAKE_ROOT,
    PIPELINE_YAML,
    STAGE_DEPENDENCIES,
    STAGE_MODEL_MAP,
    compute_identity_hash,
    run_dir,
)
from graphids.orchestrate.resources import get_failure_reactions, get_resources, scale_resources
from graphids.orchestrate.slurm import generate_script, poll, sacct_query, submit

STAGES_DIR = CONFIG_DIR / "stages"
OVERLAYS_DIR = CONFIG_DIR / "overlays"
RECIPES_DIR = CONFIG_DIR / "recipes"
RECIPE_PATH = Path(os.environ.get("KD_GAT_RECIPE", RECIPES_DIR / "ablation.yaml"))

# Upstream model → CLI flag for passing its checkpoint downstream
_CKPT_FLAG = {
    "vgae": "--data.init_args.vgae_ckpt_path",
    "dgi": "--data.init_args.vgae_ckpt_path",
    "gat": "--data.init_args.gat_ckpt_path",
}

# Recipe key → identity key (where names differ)
_RECIPE_TO_IDENTITY = {"fusion_method": "method"}


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------


class CheckpointPathIOManager(dg.ConfigurableIOManager):
    """Persists checkpoint path strings between assets via JSON sidecars.

    Assets return a checkpoint path string. handle_output writes it to a
    JSON file keyed by asset_key + partition. load_input reads it back so
    downstream assets receive the path as a function parameter.
    """

    base_dir: str

    def _sidecar(self, context: dg.OutputContext | dg.InputContext) -> Path:
        key = "/".join(context.asset_key.path)
        partition = str(getattr(context, "asset_partition_key", "default"))
        return Path(self.base_dir) / key / f"{partition}.json"

    def handle_output(self, context: dg.OutputContext, ckpt_path: str) -> None:
        p = self._sidecar(context)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"checkpoint_path": ckpt_path}))
        context.add_output_metadata({
            "checkpoint_path": dg.MetadataValue.path(ckpt_path),
        })

    def load_input(self, context: dg.InputContext) -> str:
        p = self._sidecar(context)
        if not p.exists():
            raise FileNotFoundError(f"No checkpoint path sidecar at {p}")
        return json.loads(p.read_text())["checkpoint_path"]


class SlurmTrainingResource(dg.ConfigurableResource):
    """Submits training jobs to SLURM and polls for completion."""

    dry_run: bool = False
    poll_interval: int = 60

    def submit_and_wait(
        self,
        config_files: list[str],
        resources: Any,
        job_name: str,
        cli_overrides: list[str] | None = None,
        ckpt_path: Path | None = None,
        on_state=None,
    ) -> tuple[str, int]:
        """Submit SLURM job and poll. Returns (state, job_id)."""
        script = generate_script(
            config_files, resources,
            ckpt_path=ckpt_path, cli_overrides=cli_overrides,
        )
        job_id = submit(script, resources, job_name=job_name, dry_run=self.dry_run)
        if self.dry_run:
            return "DRY_RUN", 0
        state = poll(job_id, interval=self.poll_interval, on_state=on_state)
        return state, job_id


# ---------------------------------------------------------------------------
# Convention-based config resolution
# ---------------------------------------------------------------------------


def _cli_val(v: Any) -> str:
    return str(v).lower() if isinstance(v, bool) else str(v)


def _overlay_model(stage_def: dict, merged: dict) -> str:
    """Model name for overlay lookup. Recipe model_type only applies to unsupervised stages."""
    scale = merged.get("scale", "small")
    if "model_type" in merged and stage_def.get("learning_type") == "unsupervised":
        if (OVERLAYS_DIR / f"{scale}_{merged['model_type']}.yaml").exists():
            return merged["model_type"]
    return stage_def["model"]


def _resolve_config_files(stage: str, stage_def: dict, merged: dict) -> tuple[list[str], bool]:
    scale = merged.get("scale", "small")
    model = _overlay_model(stage_def, merged)
    # Method-specific stage YAML (e.g. fusion_dqn.yaml) layers on top of base stage YAML
    method = merged.get("fusion_method")
    variant = STAGES_DIR / f"{stage}_{method}.yaml" if method else None
    used_variant = variant is not None and variant.exists()
    configs = [str(STAGES_DIR / f"{stage}.yaml")]
    if used_variant:
        configs.append(str(variant))
    overlay = OVERLAYS_DIR / f"{scale}_{model}.yaml"
    has_overlay = overlay.exists()
    if has_overlay:
        configs.append(str(overlay))
    if "auxiliaries" in merged:
        kd = OVERLAYS_DIR / f"kd_{model}.yaml"
        if kd.exists():
            configs.append(str(kd))
    return configs, has_overlay, used_variant


def _identity_value(key: str, merged: dict, stages: list[str]) -> Any:
    if key == "gat_stage":
        return "curriculum" if "curriculum" in stages else "normal"
    for rk, ik in _RECIPE_TO_IDENTITY.items():
        if ik == key and rk in merged:
            return merged[rk]
    return merged.get(key)


# ---------------------------------------------------------------------------
# Specs (identity) — WHAT exists
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageConfig:
    """Training config for one asset. Pure data, no dagster dependency."""

    asset_name: str
    stage: str
    model_type: str
    scale: str
    config_files: tuple[str, ...] = ()
    model_overrides: dict[str, str] = field(default_factory=dict)
    identity: str = ""
    kd_tag: str = ""
    upstream_asset_names: tuple[str, ...] = ()
    upstream_ckpt_flags: dict[str, str] = field(default_factory=dict)


def enumerate_assets(pipeline: dict, recipe: dict) -> list[StageConfig]:
    """Two-pass enumeration: compute canonical keys, then build configs with resolved deps."""
    defaults = recipe.get("defaults", {})
    default_stages = defaults.get("stages", ["autoencoder", "curriculum", "fusion"])
    stages_def = pipeline["stages"]

    # Pass 1: canonical key map
    canonical: dict[tuple[str, str], str] = {}
    config_stages: dict[str, dict[str, str]] = {}

    for config_name, config_overrides in recipe["configs"].items():
        merged = {**defaults, **(config_overrides or {})}
        stages = merged.get("stages", default_stages)
        has_kd = "auxiliaries" in merged
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
        merged = {**defaults, **(config_overrides or {})}
        stages = merged.get("stages", default_stages)
        has_kd = "auxiliaries" in merged
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
            config_files, has_overlay, used_variant = _resolve_config_files(stage, stage_def, merged)

            # model_keys: only these identity keys become --model.init_args.* overrides
            # If unset, all identity keys are passed (backward compat for non-fusion stages)
            model_keys = set(stage_def.get("model_keys", id_keys))
            overrides: dict[str, str] = {}
            for key in id_keys:
                if key not in model_keys:
                    continue
                if key == "scale" and has_overlay:
                    continue
                if key == "method" and used_variant:
                    continue
                val = id_cfg.get(key)
                if val is not None:
                    overrides[key] = _cli_val(val)

            upstream_names: list[str] = []
            upstream_flags: dict[str, str] = {}
            for dep in stage_def.get("depends_on", []):
                dep_stage = dep["stage"]
                dep_asset = stage_map.get(dep_stage)
                if dep_asset:
                    upstream_names.append(dep_asset)
                    flag = _CKPT_FLAG.get(dep["model"], "")
                    if flag:
                        upstream_flags[dep_asset] = flag

            if has_kd:
                teacher_scale = merged.get("auxiliaries", [{}])[0].get("teacher_scale")
                if teacher_scale:
                    for tc_name, tc_overrides in recipe["configs"].items():
                        tc_merged = {**defaults, **(tc_overrides or {})}
                        if tc_merged.get("scale") == teacher_scale and "auxiliaries" not in tc_merged:
                            tc_map = config_stages.get(tc_name, {})
                            for s in ("autoencoder", "curriculum", "normal"):
                                if s == stage and s in tc_map:
                                    upstream_names.append(tc_map[s])
                            break

            model_dir = STAGE_MODEL_MAP[stage]
            built[asset_name] = StageConfig(
                asset_name=asset_name,
                stage=stage,
                model_type=model_dir,
                scale=merged.get("scale", "small"),
                config_files=tuple(config_files),
                model_overrides=overrides,
                identity=identity,
                kd_tag="_kd" if has_kd else "",
                upstream_asset_names=tuple(sorted(set(upstream_names))),
                upstream_ckpt_flags=upstream_flags,
            )

    return list(built.values())


# ---------------------------------------------------------------------------
# Assets (behavior) — HOW to materialize
# Each @asset gets its identity from StageConfig (tags, kinds, group, deps)
# and receives upstream checkpoint paths via IOManager (ins= parameter deps).
# ---------------------------------------------------------------------------


def build_cli_args(
    cfg: StageConfig,
    dataset: str,
    seed: int,
    rd: str,
    upstream_ckpts: dict[str, str] | None = None,
) -> list[str]:
    """Build CLI override args for a training job. Pure function — no side effects."""
    args = [
        f"--data.init_args.dataset={dataset}",
        f"--seed_everything={seed}",
        f"--trainer.default_root_dir={rd}",
    ]
    for k, v in cfg.model_overrides.items():
        args.append(f"--model.init_args.{k}={v}")
    for up_name, ckpt_path in (upstream_ckpts or {}).items():
        flag = cfg.upstream_ckpt_flags.get(up_name)
        if flag:
            args.append(f"{flag}={ckpt_path}")
    return args


def _make_asset(
    cfg: StageConfig,
    partitions_def: dg.MultiPartitionsDefinition,
    lake_root: str,
    user: str,
) -> dg.AssetsDefinition:
    """One @asset per StageConfig. IOManager handles upstream checkpoint paths."""
    # ins= maps parameter names to upstream asset keys — IOManager loads them
    ins = {name: dg.AssetIn(key=dg.AssetKey(name)) for name in cfg.upstream_asset_names}
    is_eval = cfg.stage == "evaluation"

    @dg.asset(
        name=cfg.asset_name,
        ins=ins,
        partitions_def=partitions_def,
        group_name=cfg.stage,
        kinds={"metrics"} if is_eval else {"checkpoint"},
        tags={"stage": cfg.stage, "model_type": cfg.model_type, "scale": cfg.scale},
        description=f"{cfg.stage} ({cfg.model_type}, {cfg.scale})",
    )
    def _train(context, slurm: SlurmTrainingResource, **upstream_ckpts: str) -> str:
        dataset = context.partition_key.keys_by_dimension["dataset"]
        seed = int(context.partition_key.keys_by_dimension["seed"])

        rd = run_dir(
            lake_root, user, dataset, cfg.model_type, cfg.scale,
            cfg.stage, cfg.identity, cfg.kd_tag, seed,
        )
        rd_path = Path(rd)
        ckpt_file = rd_path / "checkpoints" / "best_model.ckpt"

        # Skip if already complete (checkpoint + marker from a successful run)
        complete_marker = rd_path / ".complete"
        if ckpt_file.exists() and complete_marker.exists():
            context.log.info(f"Already complete: {ckpt_file}")
            return str(ckpt_file)
        if ckpt_file.exists() and not complete_marker.exists():
            context.log.warning(
                f"Stale checkpoint (no .complete marker), retraining: {ckpt_file}"
            )

        cli_args = build_cli_args(cfg, dataset, seed, rd, upstream_ckpts)

        # SLURM resources + adaptive retry
        resources = get_resources(cfg.model_type, cfg.scale, cfg.stage)
        if context.retry_number > 0:
            for reason in ("OUT_OF_MEMORY", "TIMEOUT"):
                resources = scale_resources(resources, reason)

        # Observation callback: emit SLURM state transitions to dagster event log
        def _observe(slurm_state, jid):
            context.log_event(dg.AssetObservation(
                asset_key=context.asset_key,
                metadata={"slurm_state": slurm_state, "job_id": jid},
            ))

        resume = rd_path / "checkpoints" / "last.ckpt"
        state, job_id = slurm.submit_and_wait(
            config_files=list(cfg.config_files),
            resources=resources,
            job_name=f"{cfg.asset_name}_{dataset}_s{seed}",
            cli_overrides=cli_args,
            ckpt_path=resume if resume.exists() else None,
            on_state=_observe,
        )

        if state == "DRY_RUN":
            return str(ckpt_file)

        if state != "COMPLETED":
            reactions = get_failure_reactions()
            reaction = reactions.get(state, {})
            if reaction.get("max_retries", 0) > 0:
                raise dg.RetryRequested(
                    max_retries=reaction["max_retries"], seconds_to_wait=30)
            raise RuntimeError(f"SLURM job failed: {state}")

        # Attach SLURM accounting metadata to the asset materialization
        # Parent row has Elapsed; .batch row has MaxRSS.
        if job_id:
            out = sacct_query([job_id], "JobID,Elapsed,MaxRSS", units="G")
            wall, rss = "", ""
            if out:
                for line in out.strip().split("\n"):
                    fields = line.split("|")
                    if len(fields) < 3:
                        continue
                    jid_field = fields[0].strip()
                    if "." not in jid_field:
                        wall = fields[1].strip()
                    elif jid_field.endswith(".batch"):
                        rss = fields[2].strip()
            context.add_output_metadata({
                "job_id": job_id,
                "wall_time": wall,
                "peak_rss": rss,
            })

        complete_marker.touch()
        return str(ckpt_file)  # IOManager.handle_output stores this path

    return _train


# ---------------------------------------------------------------------------
# Asset checks
# ---------------------------------------------------------------------------


def _make_checkpoint_checks(
    cfg_lookup: dict[str, StageConfig],
    partitions_def: dg.MultiPartitionsDefinition,
    lake_root: str,
    user: str,
) -> list[dg.AssetChecksDefinition]:
    """One blocking checkpoint_exists check per asset."""
    checks = []
    for asset_name, cfg in cfg_lookup.items():

        def _make_check(name: str, c: StageConfig):
            @dg.asset_check(
                asset=dg.AssetKey(name),
                name=f"checkpoint_exists_{name}",
                blocking=True,
                description=f"Verify checkpoint for {name}",
                partitions_def=partitions_def,
            )
            def _check(context) -> dg.AssetCheckResult:
                dataset = context.partition_key.keys_by_dimension["dataset"]
                seed = int(context.partition_key.keys_by_dimension["seed"])
                rd = run_dir(
                    lake_root, user, dataset, c.model_type, c.scale,
                    c.stage, c.identity, c.kd_tag, seed,
                )
                ckpt = Path(rd) / "checkpoints" / "best_model.ckpt"
                marker = Path(rd) / ".complete"
                passed = ckpt.exists() and marker.exists()
                return dg.AssetCheckResult(
                    passed=passed,
                    metadata={
                        "path": dg.MetadataValue.path(str(ckpt)),
                        "has_marker": dg.MetadataValue.bool(marker.exists()),
                    },
                )
            return _check

        checks.append(_make_check(asset_name, cfg))
    return checks


# ---------------------------------------------------------------------------
# Component — assembles specs + behavior + resources into Definitions
# ---------------------------------------------------------------------------


class SlurmTrainingComponent(dg.Component, dg.Model, dg.Resolvable):
    """SLURM training pipeline. Reads pipeline.yaml + ablation.yaml,
    generates tagged assets with IOManager checkpoint handoff."""

    dry_run: bool = False
    poll_interval: int = 60

    def build_defs(self, context: dg.ComponentLoadContext) -> dg.Definitions:
        recipe = yaml.safe_load(RECIPE_PATH.read_text())

        # 1. Enumerate training configs (pure data)
        stage_configs = enumerate_assets(PIPELINE_YAML, recipe)

        # 1b. Validate all assets have resource profiles (fail at definition time, not submit time)
        for cfg in stage_configs:
            try:
                get_resources(cfg.model_type, cfg.scale, cfg.stage)
            except KeyError as e:
                raise KeyError(
                    f"Asset '{cfg.asset_name}' has no resource profile: {e}. "
                    f"Add entry to config/resources.yaml."
                ) from None

        # 2. Partitions
        datasets = [k for k in yaml.safe_load(CATALOG_PATH.read_text())
                     if not k.startswith("_")]
        seeds = [str(s) for s in recipe.get("sweep", {}).get("seeds", [42])]
        partitions = dg.MultiPartitionsDefinition({
            "dataset": dg.StaticPartitionsDefinition(datasets),
            "seed": dg.StaticPartitionsDefinition(seeds),
        })

        lake_root = os.environ.get("KD_GAT_LAKE_ROOT", LAKE_ROOT)
        user = os.environ.get("USER", "unknown")

        # 3. Build assets — one @asset per StageConfig, IOManager wires checkpoint paths
        assets = [_make_asset(cfg, partitions, lake_root, user) for cfg in stage_configs]

        # 4. Build asset checks
        cfg_lookup = {cfg.asset_name: cfg for cfg in stage_configs}
        checks = _make_checkpoint_checks(cfg_lookup, partitions, lake_root, user)

        # 6. Resources
        return dg.Definitions(
            assets=assets,
            asset_checks=checks,
            resources={
                "slurm": SlurmTrainingResource(
                    dry_run=self.dry_run,
                    poll_interval=self.poll_interval,
                ),
                "io_manager": CheckpointPathIOManager(
                    base_dir=f"{lake_root}/.dagster/io",
                ),
            },
        )
