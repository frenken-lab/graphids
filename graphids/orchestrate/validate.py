"""Validate all recipe config chains parse correctly."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

from graphids.config import CONFIG_DIR, PIPELINE_YAML

STAGES_DIR = CONFIG_DIR / "stages"
OVERLAYS_DIR = CONFIG_DIR / "overlays"
RECIPES_DIR = CONFIG_DIR / "recipes"
RECIPE_PATH = RECIPES_DIR / "ablation.yaml"

_LOGGER_REQUIRED_CALLBACKS = {
    "pytorch_lightning.callbacks.LearningRateMonitor",
    "lightning.pytorch.callbacks.LearningRateMonitor",
}
_NULL_LIST_FIELDS = {"pool_aggrs", "hidden_dims", "auxiliaries"}


def validate_recipe(argv: list[str]) -> None:
    """Parse every config chain in a recipe. Exit 1 on errors."""
    import argparse

    from graphids.cli import CLI_KWARGS, GraphIDSCLI
    from graphids.orchestrate.component import enumerate_assets

    p = argparse.ArgumentParser(prog="python -m graphids validate-recipe")
    p.add_argument("--recipe", default=str(RECIPE_PATH))
    args = p.parse_args(argv)

    recipe = yaml.safe_load(Path(args.recipe).read_text())
    specs = enumerate_assets(PIPELINE_YAML, recipe)

    _saved = sys.argv
    sys.argv = [sys.argv[0]]
    _cli = GraphIDSCLI(
        **{**CLI_KWARGS, "run": False, "auto_configure_optimizers": False},
        args=["--config", str(STAGES_DIR / "autoencoder.yaml"),
              "--config", str(OVERLAYS_DIR / "small_vgae.yaml"),
              "--data.init_args.dataset=hcrl_ch", "--seed_everything=42"],
    )
    parser = _cli.parser
    sys.argv = _saved

    errors: list[str] = []
    seen: set[tuple] = set()

    for spec in specs:
        chain_key = (tuple(spec.config_files)
                     + tuple(sorted(spec.model_overrides.items())))
        if chain_key in seen:
            continue
        seen.add(chain_key)

        cli_args: list[str] = []
        for f in spec.config_files:
            cli_args += ["--config", f]
        cli_args += ["--data.init_args.dataset=hcrl_ch", "--seed_everything=42"]
        for k, v in spec.model_overrides.items():
            cli_args += [f"--model.init_args.{k}={v}"]

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

    if errors:
        print(f"FAIL: {len(errors)} errors:", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        sys.exit(1)
    print("OK: all config chains valid")
