"""Dagster asset definitions for KD-GAT pipeline orchestration.

Reads ablation recipe directly, computes topology and config chains in-process.
Each asset submits one SLURM job with multi-config flags — no intermediate
expanded YAMLs or manifest.json.

Entry point: dagster asset materialize -m graphids.orchestrate.dagster_defs
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import dagster as dg
import yaml

from graphids.config import (
    CATALOG_PATH, CONFIG_DIR, LAKE_ROOT, STAGE_DEPENDENCIES,
    STAGE_MODEL_MAP, _CKPT_MODEL, compute_identity_hash, run_dir,
)
from .resources import ResourceSpec, get_failure_reactions, get_resources, scale_resources
from .slurm import generate_script, poll, submit

STAGES_DIR = CONFIG_DIR / "stages"
OVERLAYS_DIR = CONFIG_DIR / "overlays"
RECIPE_PATH = CONFIG_DIR / "ablation.yaml"

# ---------------------------------------------------------------------------
# Partitions
# ---------------------------------------------------------------------------

_datasets = [k for k in yaml.safe_load(CATALOG_PATH.read_text()) if not k.startswith("_")]

partitions = dg.MultiPartitionsDefinition({
    "dataset": dg.StaticPartitionsDefinition(_datasets),
    "seed": dg.StaticPartitionsDefinition(["42"]),
})

_FAILURE_REACTIONS = get_failure_reactions()
_DRY_RUN = os.environ.get("KD_GAT_DRY_RUN", "").lower() in ("1", "true")


# ---------------------------------------------------------------------------
# Recipe helpers (ported from expand.py, no torch/Lightning dependency)
# ---------------------------------------------------------------------------

def _cli_val(v) -> str:
    """Convert Python value to CLI-safe string (booleans lowercase for jsonargparse)."""
    return str(v).lower() if isinstance(v, bool) else str(v)


def _stage_args(
    stage: str, merged: dict, stages: list[str],
) -> tuple[list[str], dict[str, str], dict]:
    """Map recipe config to (config_files, model_overrides, identity_cfg) for a stage.

    identity_cfg has Python-typed values matching what jsonargparse resolves,
    so compute_identity_hash produces identical hashes to the old expand path.
    """
    scale = merged.get("scale", "small")
    model_type = merged.get("model_type", "vgae")
    has_kd = "auxiliaries" in merged

    configs: list[str] = [str(STAGES_DIR / f"{stage}.yaml")]
    overrides: dict[str, str] = {}

    # Identity values: Python types (not CLI strings) for hash computation.
    # Superset of all stages' identity_keys — compute_identity_hash picks
    # only the keys listed in pipeline.yaml for the given stage.
    id_cfg: dict = {
        "scale": scale,
        "conv_type": merged.get("conv_type", "gatv2"),
        "variational": merged.get("variational", True),
        "loss_fn": merged.get("loss_fn", "ce"),
    }

    if stage == "autoencoder":
        configs.append(str(OVERLAYS_DIR / f"{scale}_{model_type}.yaml"))
        for k in ("conv_type", "variational"):
            if k in merged:
                overrides[k] = _cli_val(merged[k])
        if has_kd:
            configs.append(str(OVERLAYS_DIR / f"kd_{model_type}.yaml"))

    elif stage in ("normal", "curriculum"):
        configs.append(str(OVERLAYS_DIR / f"{scale}_gat.yaml"))
        for k in ("conv_type", "loss_fn"):
            if k in merged:
                overrides[k] = _cli_val(merged[k])
        if stage == "curriculum" and "variational" in merged:
            overrides["variational"] = _cli_val(merged["variational"])
        if has_kd:
            configs.append(str(OVERLAYS_DIR / "kd_gat.yaml"))

    elif stage == "fusion":
        if "fusion_method" in merged:
            overrides["method"] = merged["fusion_method"]
        overrides["scale"] = scale
        overrides["loss_fn"] = merged.get("loss_fn", "ce")
        overrides["gat_stage"] = "curriculum" if "curriculum" in stages else "normal"
        overrides["conv_type"] = merged.get("conv_type", "gatv2")
        overrides["variational"] = _cli_val(merged.get("variational", True))
        # Fusion-specific identity values
        id_cfg["method"] = merged.get("fusion_method", "bandit")
        id_cfg["gat_stage"] = overrides["gat_stage"]

    return configs, overrides, id_cfg


# ---------------------------------------------------------------------------
# Checkpoint path wiring
# ---------------------------------------------------------------------------

_CKPT_OVERRIDES = {
    "autoencoder": "--data.init_args.vgae_ckpt_path",
    "curriculum": "--data.init_args.gat_ckpt_path",
    "normal": "--data.init_args.gat_ckpt_path",
}


def _resolve_upstream_ckpts(dep_infos: list[dict], dataset: str, seed: int) -> list[str]:
    """Build CLI overrides pointing to upstream checkpoint paths."""
    overrides = []
    for dep in dep_infos:
        dep_rd = dep["run_dir_fn"](dataset, seed)
        ckpt = f"{dep_rd}/checkpoints/best_model.ckpt"
        if dep["is_kd_teacher"]:
            overrides.append(
                f"--model.init_args.auxiliaries=[{{model_path: {ckpt}}}]")
        elif dep["stage"] in _CKPT_OVERRIDES:
            overrides.append(f"{_CKPT_OVERRIDES[dep['stage']]}={ckpt}")
    return overrides


def _extract_partition(context) -> tuple[str, int]:
    if not context.has_partition_key:
        raise RuntimeError("Partition key required. Use --partition 'dataset|seed'")
    keys = context.partition_key
    if not isinstance(keys, dg.MultiPartitionKey):
        raise RuntimeError(f"Expected MultiPartitionKey, got {type(keys).__name__}")
    return keys.keys_by_dimension["dataset"], int(keys.keys_by_dimension["seed"])


# ---------------------------------------------------------------------------
# Asset factory
# ---------------------------------------------------------------------------

def _make_asset(
    name: str, stage: str, model_type: str, scale: str,
    config_files: list[str], model_overrides: dict[str, str],
    identity: str, kd_tag: str,
    dep_names: list[str], dep_infos: list[dict],
) -> dg.AssetsDefinition:
    """One @dg.asset per unique pipeline stage. Builds multi-config SLURM command."""

    @dg.asset(
        name=name,
        deps=[dg.AssetKey(d) for d in dep_names],
        partitions_def=partitions,
    )
    def _asset(context) -> dg.MaterializeResult:
        dataset, seed = _extract_partition(context)
        user = os.environ.get("USER", "unknown")
        rd = run_dir(LAKE_ROOT, user, dataset, model_type, scale,
                     stage, identity, kd_tag, seed)
        rd_path = Path(rd)

        # Skip if done
        if (rd_path / "checkpoints" / "best_model.ckpt").exists():
            context.log.info(f"Skipping {name} — already complete")
            metrics_file = rd_path / "metrics.json"
            metrics = json.loads(metrics_file.read_text()) if metrics_file.exists() else {}
            return dg.MaterializeResult(metadata=metrics)

        # Upstream checkpoint paths
        ckpt_overrides = _resolve_upstream_ckpts(dep_infos, dataset, seed)

        # Build CLI args: dataset, seed, run dir, model overrides, upstream ckpts
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

        # Resume from last.ckpt if exists
        ckpt = rd_path / "checkpoints" / "last.ckpt"
        script = generate_script(
            config_files, resources,
            ckpt_path=ckpt if ckpt.exists() else None,
            cli_overrides=cli_overrides,
        )
        job_id = submit(script, resources,
                        job_name=f"{name}_{dataset}_s{seed}", dry_run=_DRY_RUN)

        if _DRY_RUN:
            return dg.MaterializeResult(metadata={"dry_run": True, "job_name": name})

        # Poll + handle failure
        state = poll(job_id)
        if state != "COMPLETED":
            reaction = _FAILURE_REACTIONS.get(state, {})
            if reaction.get("max_retries", 0) > 0:
                raise dg.RetryRequested(
                    max_retries=reaction["max_retries"], seconds_to_wait=30)
            raise RuntimeError(f"SLURM job {job_id} failed: {state}")

        metrics_file = rd_path / "metrics.json"
        metrics = json.loads(metrics_file.read_text()) if metrics_file.exists() else {}
        return dg.MaterializeResult(metadata=metrics)

    return _asset


# ---------------------------------------------------------------------------
# Recipe loading + topology computation (no torch, no Lightning)
# ---------------------------------------------------------------------------

def load_recipe(recipe_path: Path = RECIPE_PATH) -> dict[str, dict]:
    """Parse ablation recipe and compute asset topology.

    Returns dict: asset_name → {stage, model_type, scale, config_files,
    model_overrides, identity, kd_tag, deps}.
    """
    recipe = yaml.safe_load(recipe_path.read_text())
    defaults = recipe.get("defaults", {})
    default_stages = defaults.get("stages", ["autoencoder", "curriculum", "fusion"])

    assets: dict[str, dict] = {}
    config_chains: dict[str, dict[str, str]] = {}  # config_name → {stage: asset_name}

    for config_name, config_overrides in recipe["configs"].items():
        merged = {**defaults, **(config_overrides or {})}
        stages = merged.get("stages", default_stages)
        has_kd = "auxiliaries" in merged

        for stage in stages:
            config_files, model_overrides, id_cfg = _stage_args(stage, merged, stages)
            identity = compute_identity_hash(stage, id_cfg)
            kd_tag = "_kd" if has_kd else ""
            asset_name = f"{stage}{identity}{kd_tag}"

            if asset_name not in assets:
                model_dir = _CKPT_MODEL.get(
                    STAGE_MODEL_MAP[stage], STAGE_MODEL_MAP[stage])
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
# Build dagster Definitions from recipe
# ---------------------------------------------------------------------------

def _build_assets() -> list[dg.AssetsDefinition]:
    if not RECIPE_PATH.exists():
        return []

    assets_info = load_recipe()
    result = []

    for asset_name, info in assets_info.items():
        dep_infos = []
        for dep_name in info["deps"]:
            dep = assets_info[dep_name]
            is_kd_teacher = (dep["stage"] == info["stage"])
            # Capture dep values in closure via default args
            dep_infos.append({
                "stage": dep["stage"],
                "is_kd_teacher": is_kd_teacher,
                "run_dir_fn": (
                    lambda ds, sd, _d=dep: run_dir(
                        LAKE_ROOT, os.environ.get("USER", "unknown"),
                        ds, _d["model_type"], _d["scale"], _d["stage"],
                        _d["identity"], _d["kd_tag"], sd)
                ),
            })

        result.append(_make_asset(
            name=asset_name,
            stage=info["stage"],
            model_type=info["model_type"],
            scale=info["scale"],
            config_files=info["config_files"],
            model_overrides=info["model_overrides"],
            identity=info["identity"],
            kd_tag=info["kd_tag"],
            dep_names=info["deps"],
            dep_infos=dep_infos,
        ))

    return result


defs = dg.Definitions(
    assets=_build_assets(),
    executor=dg.multiprocess_executor.configured({"max_concurrent": 8}),
)


# ---------------------------------------------------------------------------
# Validation (lazy torch import — called on demand, not at definition time)
# ---------------------------------------------------------------------------

_LOGGER_REQUIRED_CALLBACKS = {
    "pytorch_lightning.callbacks.LearningRateMonitor",
    "lightning.pytorch.callbacks.LearningRateMonitor",
}


def validate_recipe(recipe_path: Path = RECIPE_PATH) -> list[str]:
    """Validate all config chains in the recipe parse without error.

    Bootstraps LightningCLI parser (imports torch) to verify each unique config
    chain resolves correctly. Also checks callback/logger compatibility and
    null list fields in model init_args.
    """
    from graphids.cli import GraphIDSCLI, CLI_KWARGS

    _saved_argv = sys.argv
    sys.argv = [sys.argv[0]]
    _cli = GraphIDSCLI(
        **{**CLI_KWARGS, "run": False},
        args=["--config", str(STAGES_DIR / "autoencoder.yaml"),
              "--config", str(OVERLAYS_DIR / "small_vgae.yaml"),
              "--data.init_args.dataset=hcrl_ch", "--seed_everything=42"],
    )
    parser = _cli.parser
    sys.argv = _saved_argv

    assets_info = load_recipe(recipe_path)
    errors: list[str] = []
    seen: set[tuple] = set()
    _NULL_LIST_FIELDS = {"pool_aggrs", "hidden_dims", "auxiliaries", "dqn_vgae_error_weights"}

    for asset_name, info in assets_info.items():
        chain_key = (
            tuple(info["config_files"])
            + tuple(sorted(info["model_overrides"].items()))
        )
        if chain_key in seen:
            continue
        seen.add(chain_key)

        args: list[str] = []
        for f in info["config_files"]:
            args += ["--config", f]
        args += ["--data.init_args.dataset=hcrl_ch", "--seed_everything=42"]
        for k, v in info["model_overrides"].items():
            args += [f"--model.init_args.{k}={v}"]

        try:
            parsed = parser.parse_args(args)
            cfg = yaml.safe_load(parser.dump(
                parsed, skip_link_targets=False, skip_none=False))
        except Exception as e:
            errors.append(f"{asset_name}: parse error: {e}")
            continue

        # Callback/logger compatibility
        trainer = cfg.get("trainer", {})
        logger_enabled = trainer.get("logger", True) is not False
        for cb in trainer.get("callbacks") or []:
            cp = cb.get("class_path", "")
            if cp in _LOGGER_REQUIRED_CALLBACKS and not logger_enabled:
                errors.append(f"{asset_name}: {cp.split('.')[-1]} requires logger but logger=false")

        # Null list fields
        model_args = cfg.get("model", {}).get("init_args", {})
        for field in _NULL_LIST_FIELDS:
            if field in model_args and model_args[field] is None:
                errors.append(f"{asset_name}: model.init_args.{field} is null")

    return errors


# ---------------------------------------------------------------------------
# Smoke test (pre-submission gate)
# ---------------------------------------------------------------------------

def smoke_test(*, dry_run: bool = False, dataset: str = "set_01",
               seed: int = 42, max_epochs: int = 3) -> bool:
    """Run one complete chain (autoencoder→curriculum→fusion) on gpudebug."""
    assets_info = load_recipe()

    # Find a fusion with a curriculum dep (3-stage chain)
    fusion_asset = next(
        (n for n, i in assets_info.items()
         if i["stage"] == "fusion" and "_kd" not in n
         and any(assets_info[d]["stage"] == "curriculum" for d in i["deps"])),
        None,
    )
    if not fusion_asset:
        fusion_asset = next(
            (n for n, i in assets_info.items()
             if i["stage"] == "fusion" and "_kd" not in n), None)
    if not fusion_asset:
        print("No fusion asset — cannot build chain", file=sys.stderr)
        return False

    chain: list[str] = []

    def _trace(asset: str):
        for dep in assets_info[asset]["deps"]:
            _trace(dep)
        if asset not in chain:
            chain.append(asset)
    _trace(fusion_asset)

    user = os.environ.get("USER", "unknown")
    smoke_resources = ResourceSpec(
        partition="gpudebug", time="01:00:00", mem="24G",
        cpus_per_task=3, num_workers=2, gres="gpu:1",
    )

    print(f"Smoke chain ({len(chain)} stages, {dataset}, seed {seed}, {max_epochs} epochs):")
    for asset_name in chain:
        info = assets_info[asset_name]
        rd = run_dir(LAKE_ROOT, user, dataset, info["model_type"], info["scale"],
                     info["stage"], info["identity"], info["kd_tag"], seed)

        cli_overrides = [
            f"--data.init_args.dataset={dataset}",
            f"--seed_everything={seed}",
            f"--trainer.default_root_dir={rd}",
            f"--trainer.max_epochs={max_epochs}",
        ]
        for k, v in info["model_overrides"].items():
            cli_overrides.append(f"--model.init_args.{k}={v}")

        # Upstream checkpoint overrides
        for dep_name in info["deps"]:
            dep = assets_info[dep_name]
            dep_rd = run_dir(LAKE_ROOT, user, dataset, dep["model_type"], dep["scale"],
                             dep["stage"], dep["identity"], dep["kd_tag"], seed)
            ckpt = f"{dep_rd}/checkpoints/best_model.ckpt"
            is_kd_teacher = (dep["stage"] == info["stage"])
            if is_kd_teacher:
                cli_overrides.append(
                    f"--model.init_args.auxiliaries=[{{model_path: {ckpt}}}]")
            elif dep["stage"] in _CKPT_OVERRIDES:
                cli_overrides.append(f"{_CKPT_OVERRIDES[dep['stage']]}={ckpt}")

        script = generate_script(info["config_files"], smoke_resources,
                                 cli_overrides=cli_overrides)
        job_name = f"smoke_{info['stage']}_{asset_name[-8:]}"
        job_id = submit(script, smoke_resources, job_name=job_name, dry_run=dry_run)

        if dry_run:
            print(f"  {info['stage']} ({asset_name}): dry run")
            continue

        print(f"  {info['stage']} ({asset_name}): submitted job {job_id}, waiting...")
        state = poll(job_id, interval=15)
        status = "PASS" if state == "COMPLETED" else "FAIL"
        print(f"  {status}: {info['stage']} (job {job_id}) -> {state}")

        if state != "COMPLETED":
            print(f"  Stopping chain — {info['stage']} failed", file=sys.stderr)
            return False

    if dry_run:
        print(f"Dry run: would submit {len(chain)} smoke jobs in sequence")
    return True
