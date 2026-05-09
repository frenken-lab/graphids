"""Migrate training-plan run dirs from ablations/ to training/ subdir.

Moves teacher, student_kd, student_nokd group dirs (all datasets, seed 42)
from {RUN_ROOT}/{ds}/ablations/{group} → {RUN_ROOT}/{ds}/training/{group},
then updates MLflow graphids.run_dir tags so health/show queries find runs.

Usage:
    python scripts/migrate_training_subdirs.py          # dry run
    python scripts/migrate_training_subdirs.py --apply  # execute moves + MLflow updates
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from mlflow.tracking import MlflowClient

from graphids._mlflow import configure_tracking_uri

TRAINING_GROUPS = ["teacher", "student_kd", "student_nokd"]
DATASETS = ["hcrl_sa", "set_01", "set_02", "set_03", "set_04"]


def _run_root() -> str:
    rr = os.environ.get("GRAPHIDS_RUN_ROOT")
    if not rr:
        sys.exit("GRAPHIDS_RUN_ROOT unset — source .env first")
    return rr


def plan_moves(run_root: str) -> list[tuple[Path, Path]]:
    moves: list[tuple[Path, Path]] = []
    for ds in DATASETS:
        for group in TRAINING_GROUPS:
            src = Path(run_root) / ds / "ablations" / group
            dst = Path(run_root) / ds / "training" / group
            if src.exists():
                moves.append((src, dst))
    return moves


def search_experiments_safe(client: MlflowClient) -> list[str]:
    """Return all experiment IDs matching graphids/* pattern."""
    exps = client.search_experiments(filter_string="name LIKE 'graphids/%'")
    return [e.experiment_id for e in exps]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Execute moves and MLflow updates")
    args = parser.parse_args()

    run_root = _run_root()
    configure_tracking_uri()
    client = MlflowClient()

    moves = plan_moves(run_root)
    if not moves:
        print("Nothing to move — no training-plan group dirs found under ablations/")
        return

    print(f"{'DRY RUN' if not args.apply else 'APPLYING'} — {len(moves)} directory move(s):\n")
    for src, dst in moves:
        print(f"  mv {src}")
        print(f"     → {dst}")

    print()
    exp_ids = search_experiments_safe(client)
    print(f"Scanning {len(exp_ids)} MLflow experiments for graphids.run_dir tags to update...\n")

    for src, dst in moves:
        print(f"[{src.parent.parent.name}/{src.name}]")
        filter_str = f"tags.`graphids.run_dir` LIKE '{src}/%'"
        runs = client.search_runs(
            experiment_ids=exp_ids,
            filter_string=filter_str,
            max_results=500,
        )
        if not runs:
            print(f"  MLflow: no runs found under {src}")
        else:
            for run in runs:
                old_rd = run.data.tags.get("graphids.run_dir", "")
                new_rd = old_rd.replace(str(src), str(dst), 1)
                print(f"  run {run.info.run_id[:8]}  {Path(old_rd).name}  →  {Path(new_rd).name}")
                if args.apply:
                    client.set_tag(run.info.run_id, "graphids.run_dir", new_rd)

        if args.apply:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            print(f"  moved ✓")
        print()

    if not args.apply:
        print("Dry run complete. Pass --apply to execute.")
    else:
        print("Migration complete.")


if __name__ == "__main__":
    main()
