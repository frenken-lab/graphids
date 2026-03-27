"""CLI entry point: unified jsonargparse dispatch for all subcommands.

Usage:
    python -m graphids stage=autoencoder model_type=vgae scale=large dataset=hcrl_sa
    python -m graphids build ablation --dry-run
    python -m graphids build pipeline --datasets set_01 set_02 --dry-run
    python -m graphids manifest ablation.yaml --dry-run
    python -m graphids manifest ablation.yaml --filter fusion_bandit fusion_dqn

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


import structlog
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ],
    cache_logger_on_first_use=True,
)


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_run(argv: list[str]) -> float:
    """Run a single pipeline stage. Accepts key=value overrides."""
    from graphids.config import STAGES, resolve
    from graphids.pipeline.stages import run_stage

    cfg = resolve(*argv)
    stage = cfg.stage
    if not stage or stage not in STAGES:
        raise SystemExit(f"stage= required. Valid: {list(STAGES.keys())}")

    result = run_stage(cfg, stage)
    metrics = result.get("metrics", {}) if isinstance(result, dict) else {}
    return metrics.get("val_loss", float("inf"))


def cmd_build(argv: list[str]) -> None:
    """Build an experiment manifest from a preset recipe."""
    from jsonargparse import ArgumentParser

    from graphids.pipeline.build import PRESETS

    parser = ArgumentParser(prog="graphids build", env_prefix="KD_GAT", default_env=True)
    parser.add_argument("preset", choices=list(PRESETS), help="Builder preset name")
    parser.add_argument("--out", type=Path, default=None, help="Output path (default: <preset>.yaml)")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Build then dry-run the DAG")
    parser.add_argument("--datasets", nargs="+", default=None, help="Datasets to sweep (pipeline preset)")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Seeds to sweep (pipeline preset)")
    parser.add_argument("--scales", nargs="+", default=None, help="Scales to train (pipeline preset)")
    args = parser.parse_args(argv)

    entry = PRESETS[args.preset]
    build_fn = entry["build_fn"]
    out = args.out or Path(entry["default_out"])

    if args.preset == "pipeline":
        manifest = build_fn(datasets=args.datasets, seeds=args.seeds, scales=args.scales)
    else:
        manifest = build_fn()

    manifest.write(out)
    print(f"Wrote {len(manifest._configs)} configs to {out}")

    if args.dry_run:
        manifest.run(dry_run=True)


def cmd_manifest(argv: list[str]) -> None:
    """Submit experiment manifest to SLURM (poll-based orchestration)."""
    from jsonargparse import ArgumentParser

    from graphids.pipeline.manifest import Manifest

    parser = ArgumentParser(prog="graphids manifest", env_prefix="KD_GAT", default_env=True)
    parser.add_argument("path", type=Path, help="Path to manifest YAML")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Print plan without submitting")
    parser.add_argument("--filter", nargs="*", help="Only run these config names")
    args = parser.parse_args(argv)

    Manifest.from_yaml(args.path).run(
        dry_run=args.dry_run, filter_configs=getattr(args, "filter", None),
    )


_SUBCOMMANDS = {
    "build": cmd_build,
    "manifest": cmd_manifest,
}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    args = argv if argv is not None else sys.argv[1:]

    if args and args[0] in _SUBCOMMANDS:
        return _SUBCOMMANDS[args[0]](args[1:])

    # Default: run a stage (backward compat — no "run" prefix needed)
    return cmd_run(args)


if __name__ == "__main__":
    main()
