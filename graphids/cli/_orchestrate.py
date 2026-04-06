"""Orchestration commands: from-spec, pipeline-status, rebuild-catalog, _finalize-record."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from graphids.cli.app import app


@app.command("from-spec", rich_help_panel="Orchestration")
def from_spec(
    phase: Annotated[str, typer.Option(help="Phase to run: train, test, or analyze")],
    spec_file: Annotated[Path, typer.Option(help="Path to JSON spec envelope")],
) -> None:
    """Run a pipeline stage from a canonical spec envelope (dagster -> SLURM transport)."""
    if phase not in ("train", "test", "analyze"):
        raise typer.BadParameter(f"--phase must be train, test, or analyze (got {phase!r})")

    from graphids.orchestrate.ops.entrypoint import run_from_spec

    run_from_spec(phase, spec_file)


@app.command("pipeline-status", rich_help_panel="Orchestration")
def pipeline_status(
    dataset: Annotated[str | None, typer.Option(help="Filter by dataset")] = None,
    seed: Annotated[int, typer.Option(help="Seed to display")] = 42,
    json_: Annotated[bool, typer.Option("--json", help="Output as JSON")] = False,
    log: Annotated[
        str | None,
        typer.Option(help="Show log (all/failures/retries/completions/submissions/polls)"),
    ] = None,
    log_file: Annotated[Path | None, typer.Option(help="Path to orchestrator log file")] = None,
    follow: Annotated[bool, typer.Option("-f", help="Follow log output (like tail -f)")] = False,
) -> None:
    """Show aggregated pipeline status from DuckDB catalog."""
    from graphids.orchestrate.ops.status import show_pipeline_status

    show_pipeline_status(
        dataset=dataset,
        seed=seed,
        as_json=json_,
        log_filter=log,
        log_file=log_file,
        follow=follow,
    )


@app.command("rebuild-catalog", rich_help_panel="Orchestration")
def rebuild_catalog(
    lake_root: Annotated[str | None, typer.Option(help="Lake root path")] = None,
    backfill_only: Annotated[bool, typer.Option(help="Only backfill legacy sidecars")] = False,
    skip_backfill: Annotated[bool, typer.Option(help="Skip legacy sidecar backfill")] = False,
    dry_run: Annotated[bool, typer.Option(help="Print actions without executing")] = False,
) -> None:
    """Rebuild DuckDB catalog from run_record.json sidecars."""
    from graphids.config.constants import LAKE_ROOT
    from graphids.orchestrate.ops.catalog import rebuild_catalog as _rebuild

    _rebuild(
        lake_root=lake_root or LAKE_ROOT,
        backfill_only=backfill_only,
        skip_backfill=skip_backfill,
        dry_run=dry_run,
    )


@app.command("_finalize-record", hidden=True)
def finalize_record(
    run_dir: Annotated[Path, typer.Option(help="Path to run directory")],
) -> None:
    """(Internal) Update run_record.json sidecar with phase markers + wall time."""
    from graphids.orchestrate.ops.finalize import finalize_run_record

    finalize_run_record(run_dir)
