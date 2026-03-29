"""CLI: python -m graphids.orchestrate [run|validate|smoke]

Subcommands:
  run       — dagster asset materialize (via dg launch)
  validate  — verify all recipe config chains parse correctly
  smoke     — submit one chain on gpudebug as pre-submission gate
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

from graphids.config import CONFIG_DIR, LAKE_ROOT, run_dir


STAGES_DIR = CONFIG_DIR / "stages"
OVERLAYS_DIR = CONFIG_DIR / "overlays"
RECIPE_PATH = CONFIG_DIR / "ablation.yaml"


# ---------------------------------------------------------------------------
# Validation (lazy torch import — called on demand, not at definition time)
# ---------------------------------------------------------------------------

_LOGGER_REQUIRED_CALLBACKS = {
    "pytorch_lightning.callbacks.LearningRateMonitor",
    "lightning.pytorch.callbacks.LearningRateMonitor",
}

_NULL_LIST_FIELDS = {"pool_aggrs", "hidden_dims", "auxiliaries", "dqn_vgae_error_weights"}


def validate_recipe(recipe_path: Path = RECIPE_PATH) -> list[str]:
    """Validate all config chains in the recipe parse without error.

    Bootstraps LightningCLI parser (imports torch) to verify each unique config
    chain resolves correctly. Also checks callback/logger compatibility and
    null list fields in model init_args.
    """
    from graphids.cli import CLI_KWARGS, GraphIDSCLI
    from graphids.components.slurm_training_component import enumerate_assets

    pipeline = yaml.safe_load((CONFIG_DIR / "pipeline.yaml").read_text())
    recipe = yaml.safe_load(recipe_path.read_text())
    assets_info = enumerate_assets(pipeline, recipe)

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

    errors: list[str] = []
    seen: set[tuple] = set()

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
                errors.append(
                    f"{asset_name}: {cp.split('.')[-1]} requires logger but logger=false")

        # Null list fields
        model_args = cfg.get("model", {}).get("init_args", {})
        for field in _NULL_LIST_FIELDS:
            if field in model_args and model_args[field] is None:
                errors.append(f"{asset_name}: model.init_args.{field} is null")

    return errors


# ---------------------------------------------------------------------------
# Smoke test (pre-submission gate)
# ---------------------------------------------------------------------------

def smoke_test(
    *, dry_run: bool = False, dataset: str = "set_01",
    seed: int = 42, max_epochs: int = 3,
) -> bool:
    """Run one complete chain (autoencoder→curriculum→fusion) on gpudebug."""
    from graphids.components.slurm_training_component import (
        _CKPT_CLI_FLAGS,
        enumerate_assets,
    )
    from graphids.orchestrate.resources import ResourceSpec
    from graphids.orchestrate.slurm import generate_script, poll, submit

    pipeline = yaml.safe_load((CONFIG_DIR / "pipeline.yaml").read_text())
    recipe = yaml.safe_load(RECIPE_PATH.read_text())
    assets_info = enumerate_assets(pipeline, recipe)

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

    def _trace(asset: str) -> None:
        for dep in assets_info[asset]["deps"]:
            _trace(dep)
        if asset not in chain:
            chain.append(asset)
    _trace(fusion_asset)

    user = os.environ.get("USER", "unknown")
    lake_root = os.environ.get("KD_GAT_LAKE_ROOT", LAKE_ROOT)
    smoke_resources = ResourceSpec(
        partition="gpudebug", time="01:00:00", mem="24G",
        cpus_per_task=3, num_workers=2, gres="gpu:1",
    )

    print(f"Smoke chain ({len(chain)} stages, {dataset}, seed {seed}, {max_epochs} epochs):")
    for asset_name in chain:
        info = assets_info[asset_name]
        rd = run_dir(lake_root, user, dataset, info["model_type"], info["scale"],
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
            dep_rd = run_dir(lake_root, user, dataset, dep["model_type"], dep["scale"],
                             dep["stage"], dep["identity"], dep["kd_tag"], seed)
            ckpt = f"{dep_rd}/checkpoints/best_model.ckpt"
            is_kd_teacher = (dep["stage"] == info["stage"])
            if is_kd_teacher:
                cli_overrides.append(
                    f"--model.init_args.auxiliaries=[{{model_path: {ckpt}}}]")
            elif dep["stage"] in _CKPT_CLI_FLAGS:
                cli_overrides.append(f"{_CKPT_CLI_FLAGS[dep['stage']]}={ckpt}")

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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="KD-GAT pipeline orchestrator")
    sub = p.add_subparsers(dest="command")

    sub.add_parser("run", help="Run dagster asset materialization")

    val_p = sub.add_parser("validate", help="Validate recipe config chains")
    val_p.add_argument("--recipe", default=str(RECIPE_PATH))

    smoke_p = sub.add_parser("smoke", help="Submit smoke test chain on gpudebug")
    smoke_p.add_argument("--dry-run", action="store_true")
    smoke_p.add_argument("--dataset", default="set_01")
    smoke_p.add_argument("--seed", type=int, default=42)
    smoke_p.add_argument("--max-epochs", type=int, default=3)

    args, remaining = p.parse_known_args()

    if args.command is None or args.command == "run":
        os.environ.setdefault("DAGSTER_HOME", "/fs/scratch/PAS1266/dagster")
        cmd = [
            sys.executable, "-m", "dagster", "asset", "materialize",
            "--select", "*",
            "-m", "graphids.orchestrate.definitions",
            *remaining,
        ]
        sys.exit(subprocess.call(cmd))

    elif args.command == "validate":
        errors = validate_recipe(Path(args.recipe))
        if errors:
            print(f"FAIL: {len(errors)} validation errors:", file=sys.stderr)
            for e in errors:
                print(f"  {e}", file=sys.stderr)
            sys.exit(1)
        print("OK: all config chains valid")

    elif args.command == "smoke":
        ok = smoke_test(
            dry_run=args.dry_run, dataset=args.dataset,
            seed=args.seed, max_epochs=args.max_epochs,
        )
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
