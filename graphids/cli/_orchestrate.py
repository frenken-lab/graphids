"""Orchestration commands: pipeline-status, rebuild-catalog."""

from __future__ import annotations

from typing import Annotated

import typer

from graphids.cli.app import app


@app.command("pipeline-status", rich_help_panel="Orchestration")
def pipeline_status(
    dataset: Annotated[str | None, typer.Option(help="Filter by dataset")] = None,
    seed: Annotated[int, typer.Option(help="Seed to display")] = 42,
) -> None:
    """Show aggregated pipeline status from DuckDB catalog."""
    from graphids.orchestrate.ops.status import show_pipeline_status

    show_pipeline_status(dataset=dataset, seed=seed)


@app.command("rebuild-catalog", rich_help_panel="Orchestration")
def rebuild_catalog(
    lake_root: Annotated[str | None, typer.Option(help="Lake root path")] = None,
    dry_run: Annotated[bool, typer.Option(help="Print actions without executing")] = False,
) -> None:
    """Rebuild DuckDB catalog from traces.jsonl span data."""
    from graphids.config.constants import LAKE_ROOT
    from graphids.orchestrate.ops.catalog import rebuild_catalog as _rebuild

    _rebuild(lake_root=lake_root or LAKE_ROOT, dry_run=dry_run)
