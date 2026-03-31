"""Dagster asset materialization via dg CLI."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from graphids.config import CONFIG_DIR

RECIPES_DIR = CONFIG_DIR / "recipes"
RECIPE_PATH = RECIPES_DIR / "ablation.yaml"


def run_orchestrate(argv: list[str]) -> None:
    p = argparse.ArgumentParser(prog="python -m graphids run")
    p.add_argument("--recipe", default=str(RECIPE_PATH))
    p.add_argument("--dataset", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--select", default="*")
    args, remaining = p.parse_known_args(argv)

    if "DAGSTER_HOME" not in os.environ:
        raise RuntimeError("DAGSTER_HOME not set — source .env first")
    os.environ["KD_GAT_RECIPE"] = args.recipe
    partition = f"{args.dataset}|{args.seed}"
    dg_bin = Path(sys.executable).parent / "dg"
    cmd = [str(dg_bin), "launch", "--assets", args.select,
           "--partition", partition, *remaining]
    print(f"Materializing: select={args.select} partition={partition}")
    sys.exit(subprocess.call(cmd))
