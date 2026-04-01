"""Validate all recipe config chains parse correctly."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

from graphids.config import CONFIG_DIR, PIPELINE_YAML, TrainingRunConfig, expand_recipe_configs
from graphids.core.contracts import TrainingContract, TrainingSpec

STAGES_DIR = CONFIG_DIR / "stages"
MODELS_DIR = CONFIG_DIR / "models"
RECIPES_DIR = CONFIG_DIR / "recipes"
RECIPE_PATH = RECIPES_DIR / "ablation.yaml"

_LOGGER_REQUIRED_CALLBACKS = {
    "pytorch_lightning.callbacks.LearningRateMonitor",
    "lightning.pytorch.callbacks.LearningRateMonitor",
}
_NULL_LIST_FIELDS = {"pool_aggrs", "hidden_dims", "auxiliaries"}

# Stage conventions for monitor metrics.
# Fusion stages optimize accuracy; all others optimize loss.
_STAGE_MONITOR_CONVENTIONS: dict[str, tuple[str, str]] = {
    "autoencoder": ("val_loss", "min"),
    "normal": ("val_loss", "min"),
    "curriculum": ("val_loss", "min"),
    "fusion": ("val_acc", "max"),
}


def _check_monitor_conventions(
    asset_name: str, stage: str, cfg: dict,
) -> list[str]:
    """Warn if checkpoint/early_stopping monitor doesn't match stage conventions."""
    expected = _STAGE_MONITOR_CONVENTIONS.get(stage)
    if expected is None:
        return []

    expected_monitor, expected_mode = expected
    warnings: list[str] = []

    for namespace, label in [("checkpoint", "ModelCheckpoint"),
                             ("early_stopping", "EarlyStopping")]:
        ns_cfg = cfg.get(namespace, {})
        if not isinstance(ns_cfg, dict):
            continue
        monitor = ns_cfg.get("monitor")
        mode = ns_cfg.get("mode")
        if monitor is not None and monitor != expected_monitor:
            warnings.append(
                f"{asset_name}: {label} monitor={monitor!r}, "
                f"expected {expected_monitor!r} for {stage} stage"
            )
        if mode is not None and mode != expected_mode:
            warnings.append(
                f"{asset_name}: {label} mode={mode!r}, "
                f"expected {expected_mode!r} for {stage} stage"
            )

    return warnings


def validate_recipe(argv: list[str]) -> None:
    """Validate Dagster defs and/or recipe config chains."""
    import argparse

    from graphids.cli import CLI_KWARGS, GraphIDSCLI
    from graphids.orchestrate.component import SlurmTrainingComponent
    from graphids.orchestrate.planning import enumerate_assets

    p = argparse.ArgumentParser(prog="python -m graphids validate-recipe")
    p.add_argument("--recipe", default=str(RECIPE_PATH))
    p.add_argument("--skip-lightning", action="store_true")
    p.add_argument("--skip-dagster", action="store_true")
    args = p.parse_args(argv)

    recipe = expand_recipe_configs(yaml.safe_load(Path(args.recipe).read_text()))

    dagster_errors: list[str] = []
    if not args.skip_dagster:
        try:
            defs = SlurmTrainingComponent().build_defs(None)
            defs.get_repository_def()
        except Exception as e:
            dagster_errors.append(f"Dagster definitions are not loadable: {e}")

    # Early schema validation — catches typos and invalid values before CLI parsing
    schema_errors: list[str] = []
    try:
        default_cfg = TrainingRunConfig(**recipe.get("defaults", {}))
    except Exception as e:
        schema_errors.append(f"Recipe defaults: {e}")
    else:
        for name, overrides in recipe.get("configs", {}).items():
            try:
                default_cfg.merge(overrides or {})
            except Exception as e:
                schema_errors.append(f"Recipe config '{name}': {e}")
    if schema_errors:
        print(f"FAIL: {len(schema_errors)} recipe schema errors:", file=sys.stderr)
        for e in schema_errors:
            print(f"  {e}", file=sys.stderr)
        sys.exit(1)

    if dagster_errors:
        print(f"FAIL: {len(dagster_errors)} dagster load errors:", file=sys.stderr)
        for e in dagster_errors:
            print(f"  {e}", file=sys.stderr)
        sys.exit(1)

    if args.skip_lightning:
        print("OK: dagster definitions loadable")
        return

    specs = enumerate_assets(PIPELINE_YAML, recipe)

    _saved = sys.argv
    sys.argv = [sys.argv[0]]
    _cli = GraphIDSCLI(
        **{**CLI_KWARGS, "run": False, "auto_configure_optimizers": False},
        args=["--config", str(STAGES_DIR / "autoencoder.yaml"),
              "--config", str(MODELS_DIR / "vgae" / "base.yaml"),
              "--config", str(MODELS_DIR / "vgae" / "scales" / "small.yaml"),
              "--data.init_args.dataset=hcrl_ch", "--seed_everything=42"],
    )
    parser = _cli.parser
    sys.argv = _saved

    errors: list[str] = []
    warnings: list[str] = []
    seen: set[tuple] = set()

    for spec in specs:
        chain_key = (tuple(spec.config_files)
                     + tuple(sorted(spec.model_init_overrides.items())))
        if chain_key in seen:
            continue
        seen.add(chain_key)

        cli_args: list[str] = []
        for f in spec.config_files:
            cli_args += ["--config", f]
        parse_spec = TrainingSpec(
            stage=spec.stage,
            model_family=spec.model_type,
            scale=spec.scale,
            dataset="hcrl_ch",
            seed=42,
            run_dir="/tmp/graphids-validate",
            config_files=spec.config_files,
            model_init_overrides=spec.model_init_overrides,
        )
        cli_args += TrainingContract.to_cli_overrides(parse_spec)

        try:
            parsed = parser.parse_args(cli_args)
            cfg = yaml.safe_load(
                parser.dump(parsed, skip_link_targets=False, skip_none=False))
        except (Exception, SystemExit) as e:
            errors.append(f"{spec.asset_name}: parse error: {e}")
            continue

        trainer = cfg.get("trainer", {})
        logger_on = trainer.get("logger", True) is not False
        for cb in trainer.get("callbacks") or []:
            cp = cb.get("class_path", "")
            if cp in _LOGGER_REQUIRED_CALLBACKS and not logger_on:
                errors.append(
                    f"{spec.asset_name}: {cp.split('.')[-1]} requires logger")

        model_args = cfg.get("model", {}).get("init_args", {})
        for fld in _NULL_LIST_FIELDS:
            if fld in model_args and model_args[fld] is None:
                errors.append(f"{spec.asset_name}: model.init_args.{fld} is null")

        warnings.extend(_check_monitor_conventions(spec.asset_name, spec.stage, cfg))

    if warnings:
        print(f"WARN: {len(warnings)} monitor convention warnings:", file=sys.stderr)
        for w in warnings:
            print(f"  {w}", file=sys.stderr)
    if errors:
        print(f"FAIL: {len(errors)} errors:", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        sys.exit(1)
    print("OK: all config chains valid")


main = validate_recipe
