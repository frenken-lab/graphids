"""CLI: python -m graphids.orchestrate [run|validate|smoke]

Subcommands:
  run       — dagster asset materialize (default)
  validate  — verify all recipe config chains parse correctly
  smoke     — submit one chain on gpudebug as pre-submission gate
"""

import argparse
import os
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description="KD-GAT pipeline orchestrator")
    sub = p.add_subparsers(dest="command")

    sub.add_parser("run", help="Run dagster asset materialization")

    val_p = sub.add_parser("validate", help="Validate recipe config chains")
    val_p.add_argument("--recipe", default="graphids/config/ablation.yaml")

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
            "-m", "graphids.orchestrate.dagster_defs",
            *remaining,
        ]
        sys.exit(subprocess.call(cmd))

    elif args.command == "validate":
        from .dagster_defs import validate_recipe
        from pathlib import Path

        errors = validate_recipe(Path(args.recipe))
        if errors:
            print(f"FAIL: {len(errors)} validation errors:", file=sys.stderr)
            for e in errors:
                print(f"  {e}", file=sys.stderr)
            sys.exit(1)
        print("OK: all config chains valid")

    elif args.command == "smoke":
        from .dagster_defs import smoke_test

        ok = smoke_test(
            dry_run=args.dry_run, dataset=args.dataset,
            seed=args.seed, max_epochs=args.max_epochs,
        )
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
