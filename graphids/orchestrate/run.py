"""Dagster asset materialization via dg CLI."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

from graphids.config import CONFIG_DIR

RECIPES_DIR = CONFIG_DIR / "recipes"
RECIPE_PATH = RECIPES_DIR / "ablation.yaml"


def run_orchestrate(argv: list[str]) -> None:
    p = argparse.ArgumentParser(prog="python -m graphids run")
    p.add_argument("--recipe", default=str(RECIPE_PATH))
    p.add_argument("--dataset", nargs="*", default=None,
                   help="Subset of datasets (default: all from recipe)")
    p.add_argument("--seed", nargs="*", type=int, default=None,
                   help="Subset of seeds (default: all from recipe)")
    p.add_argument("--select", default="*")
    args, remaining = p.parse_known_args(argv)

    if "DAGSTER_HOME" not in os.environ:
        raise RuntimeError("DAGSTER_HOME not set — source .env first")
    os.environ["KD_GAT_RECIPE"] = args.recipe

    recipe = yaml.safe_load(Path(args.recipe).read_text())
    datasets = args.dataset or recipe.get("sweep", {}).get("datasets", [])
    seeds = [str(s) for s in (args.seed or recipe.get("sweep", {}).get("seeds", [42]))]

    if not datasets:
        raise RuntimeError("No datasets — set sweep.datasets in recipe or pass --dataset")

    dg_bin = Path(sys.executable).parent / "dg"
    partitions = [f"{ds}|{seed}" for ds in datasets for seed in seeds]

    print(f"Recipe: {args.recipe}")
    print(f"Datasets: {datasets}")
    print(f"Seeds: {seeds}")
    print(f"Launching {len(partitions)} partition(s)...")

    procs: list[tuple[str, subprocess.Popen]] = []
    for partition in partitions:
        cmd = [str(dg_bin), "launch", "--assets", args.select,
               "--partition", partition, *remaining]
        print(f"  -> {partition}")
        procs.append((partition, subprocess.Popen(cmd)))

    failures = []
    for partition, proc in procs:
        rc = proc.wait()
        if rc != 0:
            failures.append(partition)

    if failures:
        print(f"\nFailed ({len(failures)}/{len(procs)}): {failures}")
        sys.exit(1)
    print(f"\nAll {len(procs)} partition(s) launched.")
