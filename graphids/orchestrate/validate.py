"""Validate dagster defs and recipe config chains (dev tool).

Thin wrapper around ``ConfigResolver.resolve_and_validate``, which every
dagster-submitted asset also runs automatically (ADR 0009). Use it locally
before submitting recipes to catch typos, null list fields, and callback/logger
wiring issues without spinning up a dagster worker.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from graphids.config import CONFIG_DIR, PIPELINE_YAML, TrainingRunConfig, expand_recipe_configs

RECIPE_PATH = CONFIG_DIR / "recipes" / "ablation.yaml"


def _fail(label: str, errors: list[str]) -> None:
    if not errors:
        return
    print(f"FAIL: {len(errors)} {label}:", file=sys.stderr)
    for e in errors:
        print(f"  {e}", file=sys.stderr)
    sys.exit(1)


def validate_recipe(argv: list[str]) -> None:
    """Validate Dagster defs and/or recipe config chains."""
    from graphids.orchestrate.component import SlurmTrainingComponent
    from graphids.orchestrate.planning import enumerate_assets
    from graphids.orchestrate.resolve import ConfigResolver

    p = argparse.ArgumentParser(prog="python -m graphids validate-recipe")
    p.add_argument("--recipe", default=str(RECIPE_PATH))
    p.add_argument("--skip-lightning", action="store_true")
    p.add_argument("--skip-dagster", action="store_true")
    args = p.parse_args(argv)

    recipe = expand_recipe_configs(yaml.safe_load(Path(args.recipe).read_text()))

    if not args.skip_dagster:
        try:
            SlurmTrainingComponent().build_defs(None).get_repository_def()
        except Exception as e:
            _fail("dagster load errors", [f"Dagster definitions are not loadable: {e}"])

    # --- Recipe schema (Pydantic) ---
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
    _fail("recipe schema errors", schema_errors)

    if args.skip_lightning:
        print("OK: dagster definitions loadable")
        return

    # --- CLI-chain validation via ConfigResolver ---
    # Dedupe by unique (config_files + model_init_overrides) chain so we don't
    # re-parse the same chain for every dataset/seed partition.
    resolver = ConfigResolver(lake_root="/tmp/validate", user="validate")
    errors: list[str] = []
    seen: set[tuple] = set()
    for cfg in enumerate_assets(PIPELINE_YAML, recipe):
        chain_key = (
            tuple(cfg.config_files)
            + tuple(sorted(cfg.model_init_overrides.items()))
        )
        if chain_key in seen:
            continue
        seen.add(chain_key)
        try:
            resolver.resolve_and_validate(cfg, dataset="hcrl_ch", seed=42)
        except (ValueError, FileNotFoundError) as e:
            errors.append(f"{cfg.asset_name}: {e}")

    _fail("errors", errors)
    print(f"OK: {len(seen)} unique config chains valid")


main = validate_recipe
