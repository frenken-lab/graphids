#!/usr/bin/env python3
"""Build a full pipeline manifest across datasets, seeds, and scales.

Usage:
    python scripts/build_pipeline.py                          # all defaults
    python scripts/build_pipeline.py --datasets set_01 set_02 # subset of datasets
    python scripts/build_pipeline.py --seeds 42 123           # subset of seeds
    python scripts/build_pipeline.py --scales small large     # subset of scales
    python scripts/build_pipeline.py --dry-run                # build + preview DAG
    python scripts/build_pipeline.py --out my_sweep.yaml      # custom output
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

from graphids.config.manifest_builder import ManifestBuilder


ALL_DATASETS = list(yaml.safe_load(
    (Path(__file__).resolve().parents[1] / "graphids" / "config" / "datasets.yaml").read_text(),
).keys())

ALL_SEEDS = [42, 123, 456]

ALL_SCALES = ["large", "small"]


def build(datasets: list[str], seeds: list[int], scales: list[str]) -> ManifestBuilder:
    b = ManifestBuilder(
        sweep={"dataset": datasets, "seed": seeds},
        defaults={
            "stages": ["autoencoder", "curriculum", "fusion", "evaluation"],
            "scale": "small",
            "training.loss_fn": "focal",
            "fusion.method": "bandit",
        },
    )

    for scale in scales:
        b.add(scale, scale=scale)

    return b


def main():
    parser = argparse.ArgumentParser(description="Build full pipeline manifest")
    parser.add_argument("--datasets", nargs="+", default=ALL_DATASETS,
                        help=f"Datasets to sweep (default: all {len(ALL_DATASETS)})")
    parser.add_argument("--seeds", nargs="+", type=int, default=ALL_SEEDS,
                        help=f"Seeds to sweep (default: {ALL_SEEDS})")
    parser.add_argument("--scales", nargs="+", default=ALL_SCALES,
                        help=f"Scales to train (default: {ALL_SCALES})")
    parser.add_argument("--out", default="pipeline.yaml", help="Output path (default: pipeline.yaml)")
    parser.add_argument("--dry-run", action="store_true", help="Build then dry-run the DAG")
    args = parser.parse_args()

    b = build(args.datasets, args.seeds, args.scales)
    out = Path(args.out)
    b.write(out)
    n_configs = len(b._configs)
    n_combos = len(args.datasets) * len(args.seeds) * n_configs
    print(f"Wrote {n_configs} configs × {len(args.datasets)} datasets × {len(args.seeds)} seeds = {n_combos} combos to {out}")

    if args.dry_run:
        subprocess.run(
            [sys.executable, "-m", "graphids", "manifest", str(out), "--dry-run"],
            check=True,
        )


if __name__ == "__main__":
    main()
