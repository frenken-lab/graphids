"""CLI entry point: python -m graphids.orchestrate <config_dir> [options]."""

import argparse
import sys

from .submit import run_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Submit LightningCLI stages to SLURM with DAG ordering and retry.",
    )
    parser.add_argument("config_dir", help="Directory containing per-stage YAML configs")
    parser.add_argument("--datasets", nargs="+", default=None, help="Override datasets (default: from configs)")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Override seeds (default: from configs)")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--poll-interval", type=int, default=300, help="Seconds between sacct polls")
    parser.add_argument("--dry-run", action="store_true", help="Print sbatch commands without submitting")
    args = parser.parse_args()

    from pathlib import Path

    run_pipeline(
        config_dir=Path(args.config_dir),
        datasets=args.datasets,
        seeds=args.seeds,
        max_retries=args.max_retries,
        poll_interval=args.poll_interval,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
