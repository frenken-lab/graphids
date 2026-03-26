"""CLI entry point: single stage runs via resolve(), DAG submission via manifest.

Usage:
    python -m graphids stage=autoencoder model_type=vgae scale=large dataset=hcrl_sa
    python -m graphids manifest ablation.yaml --dry-run

JSON logs: set KD_GAT_JSON_LOGS=1 (auto-detected, no CLI flag needed).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)
# file_system strategy: safety net for main-process tensor sharing.
# Spawn workers don't inherit this — they get it via worker_init_fn in datamodule.py.
mp.set_sharing_strategy("file_system")


from graphids.logging import configure_logging
configure_logging()


def main(argv: list[str] | None = None) -> None:
    from graphids.config import STAGES, resolve
    from graphids.pipeline.stages import run_stage

    args = argv if argv is not None else sys.argv[1:]
    overrides = [a for a in args if "=" in a and not a.startswith("-")]

    cfg = resolve(*overrides)
    stage = cfg.stage
    if not stage or stage not in STAGES:
        raise SystemExit(f"stage= required. Valid: {list(STAGES.keys())}")

    result = run_stage(cfg, stage)
    metrics = result.get("metrics", {}) if isinstance(result, dict) else {}
    return metrics.get("val_loss", float("inf"))


def manifest(argv: list[str] | None = None) -> None:
    """Submit experiment manifest to SLURM."""
    import argparse

    from graphids.pipeline.manifest import submit_manifest

    parser = argparse.ArgumentParser(description="Submit experiment manifest to SLURM")
    parser.add_argument("manifest", type=Path, help="Path to manifest YAML")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without submitting")
    parser.add_argument("--filter", nargs="*", help="Only run these config names")
    args = parser.parse_args(argv)

    futures = submit_manifest(args.manifest, dry_run=args.dry_run, filter_configs=args.filter)
    if futures:
        print(f"Submitted {len(futures)} jobs")


if __name__ == "__main__":
    if sys.argv[1:2] == ["manifest"]:
        manifest(sys.argv[2:])
    else:
        main()
