"""CLI: python -m graphids.orchestrate [run|validate|smoke]

Subcommands:
  run       — dagster asset materialize
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

_LOGGER_REQUIRED_CALLBACKS = {
    "pytorch_lightning.callbacks.LearningRateMonitor",
    "lightning.pytorch.callbacks.LearningRateMonitor",
}
_NULL_LIST_FIELDS = {"pool_aggrs", "hidden_dims", "auxiliaries", "dqn_vgae_error_weights"}


def validate_recipe(recipe_path: Path = RECIPE_PATH) -> list[str]:
    """Validate all config chains parse correctly (lazy torch import)."""
    from graphids.cli import CLI_KWARGS, GraphIDSCLI
    from graphids.orchestrate.component import enumerate_assets
    from graphids.config import PIPELINE_YAML

    recipe = yaml.safe_load(recipe_path.read_text())
    specs = enumerate_assets(PIPELINE_YAML, recipe)

    _saved = sys.argv
    sys.argv = [sys.argv[0]]
    _cli = GraphIDSCLI(
        **{**CLI_KWARGS, "run": False},
        args=["--config", str(STAGES_DIR / "autoencoder.yaml"),
              "--config", str(OVERLAYS_DIR / "small_vgae.yaml"),
              "--data.init_args.dataset=hcrl_ch", "--seed_everything=42"],
    )
    parser = _cli.parser
    sys.argv = _saved

    errors: list[str] = []
    seen: set[tuple] = set()

    for spec in specs:
        chain_key = (tuple(spec.config_files) + tuple(sorted(spec.model_overrides.items())))
        if chain_key in seen:
            continue
        seen.add(chain_key)

        args: list[str] = []
        for f in spec.config_files:
            args += ["--config", f]
        args += ["--data.init_args.dataset=hcrl_ch", "--seed_everything=42"]
        for k, v in spec.model_overrides.items():
            args += [f"--model.init_args.{k}={v}"]

        try:
            parsed = parser.parse_args(args)
            cfg = yaml.safe_load(parser.dump(parsed, skip_link_targets=False, skip_none=False))
        except Exception as e:
            errors.append(f"{spec.asset_name}: parse error: {e}")
            continue

        trainer = cfg.get("trainer", {})
        logger_on = trainer.get("logger", True) is not False
        for cb in trainer.get("callbacks") or []:
            cp = cb.get("class_path", "")
            if cp in _LOGGER_REQUIRED_CALLBACKS and not logger_on:
                errors.append(f"{spec.asset_name}: {cp.split('.')[-1]} requires logger")

        model_args = cfg.get("model", {}).get("init_args", {})
        for fld in _NULL_LIST_FIELDS:
            if fld in model_args and model_args[fld] is None:
                errors.append(f"{spec.asset_name}: model.init_args.{fld} is null")

    return errors


def smoke_test(*, dry_run: bool = False, dataset: str = "set_01",
               seed: int = 0, max_epochs: int = 3) -> bool:
    """Run one 3-stage chain on gpudebug."""
    from graphids.orchestrate.component import enumerate_assets
    from graphids.config import PIPELINE_YAML
    from graphids.orchestrate.resources import ResourceSpec
    from graphids.orchestrate.slurm import generate_script, poll, submit

    recipe = yaml.safe_load(RECIPE_PATH.read_text())
    specs = {s.asset_name: s for s in enumerate_assets(PIPELINE_YAML, recipe)}

    # Find a fusion with a curriculum dep (3-stage chain preferred)
    fusion = next(
        (s for s in specs.values()
         if s.stage == "fusion" and "_kd" not in s.asset_name
         and any(specs.get(d, specs.get(d, type("", (), {"stage": ""})())).stage == "curriculum"
                 for d in s.upstream_asset_names)),
        None)
    if not fusion:
        fusion = next(
            (s for s in specs.values()
             if s.stage == "fusion" and "_kd" not in s.asset_name and s.upstream_asset_names),
            None)
    if not fusion:
        print("No fusion asset found", file=sys.stderr)
        return False

    chain: list[str] = []

    def _trace(name: str) -> None:
        for dep in specs[name].upstream_asset_names:
            if dep in specs:
                _trace(dep)
        if name not in chain:
            chain.append(name)
    _trace(fusion.asset_name)

    lake_root = os.environ.get("KD_GAT_LAKE_ROOT", LAKE_ROOT)
    user = os.environ.get("USER", "unknown")
    smoke_res = ResourceSpec(
        partition="gpudebug", time="01:00:00", mem="24G",
        cpus_per_task=3, num_workers=2, gres="gpu:1",
    )

    print(f"Smoke chain ({len(chain)} stages, {dataset}, seed {seed}, {max_epochs} epochs):")
    for name in chain:
        spec = specs[name]
        rd = run_dir(lake_root, user, dataset, spec.model_type, spec.scale,
                     spec.stage, spec.identity, spec.kd_tag, seed)
        ckpt_file = Path(rd) / "checkpoints" / "best_model.ckpt"
        if ckpt_file.exists():
            print(f"  SKIP: {spec.stage} ({name}) — checkpoint exists")
            continue
        cli_args = [
            f"--data.init_args.dataset={dataset}",
            f"--seed_everything={seed}",
            f"--trainer.default_root_dir={rd}",
            f"--trainer.max_epochs={max_epochs}",
        ]
        for k, v in spec.model_overrides.items():
            cli_args.append(f"--model.init_args.{k}={v}")
        for dep_name in spec.upstream_asset_names:
            dep = specs.get(dep_name)
            if dep:
                dep_rd = run_dir(lake_root, user, dataset, dep.model_type, dep.scale,
                                 dep.stage, dep.identity, dep.kd_tag, seed)
                flag = spec.upstream_ckpt_flags.get(dep_name)
                if flag:
                    cli_args.append(f"{flag}={dep_rd}/checkpoints/best_model.ckpt")

        script = generate_script(spec.config_files, smoke_res, cli_overrides=cli_args)
        job_name = f"smoke_{spec.stage}_{name[-8:]}"
        job_id = submit(script, smoke_res, job_name=job_name, dry_run=dry_run)

        if dry_run:
            print(f"  {spec.stage} ({name}): dry run")
            continue
        print(f"  {spec.stage} ({name}): job {job_id}, waiting...")
        state = poll(job_id, interval=15)
        print(f"  {'PASS' if state == 'COMPLETED' else 'FAIL'}: {spec.stage} -> {state}")
        if state != "COMPLETED":
            return False

    if dry_run:
        print(f"Dry run: would submit {len(chain)} smoke jobs")
    return True


def main() -> None:
    p = argparse.ArgumentParser(description="KD-GAT pipeline orchestrator")
    sub = p.add_subparsers(dest="command")
    run_p = sub.add_parser("run", help="Run dagster asset materialization")
    run_p.add_argument("--dataset", required=True, help="Dataset partition (e.g. set_01)")
    run_p.add_argument("--seed", type=int, default=42, help="Seed partition (default: 42)")
    run_p.add_argument("--select", default="*", help="Asset selection (default: all)")
    val_p = sub.add_parser("validate", help="Validate recipe config chains")
    val_p.add_argument("--recipe", default=str(RECIPE_PATH))
    smoke_p = sub.add_parser("smoke", help="Smoke test on gpudebug")
    smoke_p.add_argument("--dry-run", action="store_true")
    smoke_p.add_argument("--dataset", default="set_01")
    smoke_p.add_argument("--seed", type=int, default=0)
    smoke_p.add_argument("--max-epochs", type=int, default=3)
    args, remaining = p.parse_known_args()

    if args.command is None or args.command == "run":
        if args.command is None:
            p.print_help()
            sys.exit(1)
        os.environ.setdefault("DAGSTER_HOME", "/fs/scratch/PAS1266/dagster")
        partition = f"{args.dataset}|{args.seed}"
        dg_bin = Path(sys.executable).parent / "dg"
        cmd = [str(dg_bin), "launch", "--assets", args.select,
               "--partition", partition, *remaining]
        print(f"Materializing: select={args.select} partition={partition}")
        sys.exit(subprocess.call(cmd))
    elif args.command == "validate":
        errors = validate_recipe(Path(args.recipe))
        if errors:
            print(f"FAIL: {len(errors)} errors:", file=sys.stderr)
            for e in errors:
                print(f"  {e}", file=sys.stderr)
            sys.exit(1)
        print("OK: all config chains valid")
    elif args.command == "smoke":
        ok = smoke_test(dry_run=args.dry_run, dataset=args.dataset,
                        seed=args.seed, max_epochs=args.max_epochs)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
