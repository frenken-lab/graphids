"""Ablation launcher CLI — thin wrapper around ``graphids.slurm.dag``."""

from __future__ import annotations

import sys
from typing import Annotated

import typer

from graphids.cli.app import app


@app.command("launch-ablation", rich_help_panel="SLURM")
def launch_ablation_cli(
    dataset: Annotated[str, typer.Option(help="Dataset name")] = "set_01",
    seed: Annotated[
        list[int] | None,
        typer.Option(help="Repeat for multiple seeds; default (42, 123, 777)"),
    ] = None,
    cluster: Annotated[str, typer.Option(help="Target cluster, e.g. cardinal")] = "",
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Print commands, do not submit")
    ] = False,
    force_refit: Annotated[
        bool,
        typer.Option(
            "--force-refit",
            help=(
                "Re-fit even if MLflow shows a FINISHED prior run. Use after "
                "a code change that invalidates earlier ckpts (e.g. scaler / "
                "decoder / objective changes) — status alone doesn't carry "
                "code-version semantics."
            ),
        ),
    ] = False,
) -> None:
    """Submit the OFAT ablation DAG to SLURM.

    Topology + per-group walltime overrides live in
    ``graphids.slurm.dag.OFAT_DAG``. Idempotent by default — skips any
    ``(variant, seed)`` whose latest fit attempt is already FINISHED in
    MLflow. Pass ``--force-refit`` to override.
    """
    from graphids.slurm.dag import DEFAULT_SEEDS, launch_ablation

    seeds = tuple(seed) if seed else DEFAULT_SEEDS
    result = launch_ablation(
        dataset=dataset,
        seeds=seeds,
        cluster=cluster,
        dry_run=dry_run,
        force_refit=force_refit,
    )

    print("", file=sys.stderr)
    print("=== Launched ===", file=sys.stderr)
    print(f"Dataset:  {dataset}", file=sys.stderr)
    print(f"Seeds:    {list(seeds)}", file=sys.stderr)
    print(f"Nodes:    {len({name for name, _ in result.jids})}", file=sys.stderr)
    print(f"Skipped:  {len(result.skipped)} (already FINISHED in MLflow)", file=sys.stderr)
