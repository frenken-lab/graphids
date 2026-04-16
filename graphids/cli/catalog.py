"""Catalog commands: query, rebuild."""

from __future__ import annotations

import json as _json
import sys
from pathlib import Path
from typing import Annotated

import typer

from graphids.cli.app import _complete_dataset, app


@app.command("catalog-query", rich_help_panel="Catalog")
def catalog_query(
    group: Annotated[str | None, typer.Option(help="Filter by ablation group")] = None,
    variant: Annotated[str | None, typer.Option(help="Filter by variant")] = None,
    dataset: Annotated[
        str | None,
        typer.Option(help="Filter by dataset", autocompletion=_complete_dataset),
    ] = None,
    seed: Annotated[int | None, typer.Option(help="Filter by seed")] = None,
    since_ns: Annotated[
        int | None,
        typer.Option(help="Filter started_at >= this epoch-ns value"),
    ] = None,
    limit: Annotated[int, typer.Option(help="Max rows to print")] = 50,
) -> None:
    """Print filtered rows from the lake-wide runs table."""
    from graphids.catalog import Catalog
    from graphids.config.constants import LAKE_ROOT

    df = Catalog(LAKE_ROOT).query_runs(
        group=group,
        variant=variant,
        dataset=dataset,
        seed=seed,
        since_ns=since_ns,
        limit=limit,
    )
    if df.is_empty():
        typer.echo("no rows")
        return
    typer.echo(df)


@app.command("catalog-rebuild", rich_help_panel="Catalog")
def catalog_rebuild(
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation")] = False,
) -> None:
    """Backfill catalog rows from run_dirs that have summary.json but no DB row.

    Scans the ablation tree under ``{lake_root}`` for ``summary.json``,
    diffs against the catalog, and writes the missing rows. Non-ablation
    run_dirs skip silently (same contract as the orchestrator hook).
    Timestamps default to the ``summary.json`` mtime for runs that
    completed before the catalog existed.
    """
    from graphids.catalog import Catalog, parse_run_dir, run_id_for
    from graphids.config.constants import LAKE_ROOT
    from graphids.config.settings import get_settings

    cat = Catalog(LAKE_ROOT)
    existing = cat.existing_run_ids()
    cluster = get_settings().cluster or None

    missing: list[Path] = []
    for ablations_dir in Path(LAKE_ROOT).glob("*/ablations"):
        for summary in ablations_dir.glob("*/*/seed_*/summary.json"):
            run_dir = summary.parent
            identity = parse_run_dir(run_dir)
            if identity is None:
                continue
            if run_id_for(identity, cluster=cluster) not in existing:
                missing.append(run_dir)

    if not missing:
        typer.echo("catalog up to date")
        return

    typer.echo(f"would backfill {len(missing)} run(s):")
    for run_dir in missing[:20]:
        typer.echo(f"  {run_dir}")
    if len(missing) > 20:
        typer.echo(f"  ... and {len(missing) - 20} more")

    if not yes:
        if sys.stdin.isatty():
            typer.confirm("proceed?", abort=True)
        else:
            raise typer.BadParameter("non-interactive shell requires --yes/-y")

    written = 0
    for run_dir in missing:
        summary_path = run_dir / "summary.json"
        summary = _json.loads(summary_path.read_text())
        mtime_ns = int(summary_path.stat().st_mtime * 1e9)
        rid = cat.record_run(
            run_dir,
            metrics=summary.get("metrics") or {},
            git_sha=summary.get("git_sha", ""),
            status=summary.get("status", "ok"),
            started_at_ns=mtime_ns,
            ended_at_ns=mtime_ns,
        )
        if rid:
            written += 1
    typer.echo(f"wrote {written} row(s)")
