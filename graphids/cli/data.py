"""Data commands: rebuild-caches, validate-metadata.

Fusion-state extraction is no longer a Typer command — it's an
``ExtractRow`` in fusion plan jsonnets, dispatched by ``graphids exec/submit``.
"""

from __future__ import annotations

import sys
from typing import Annotated

import typer

from graphids.cli.app import _complete_dataset, app


@app.command("rebuild-caches", rich_help_panel="Data")
def rebuild_caches(
    dataset: Annotated[
        list[str] | None,
        typer.Option(help="Dataset name(s) to rebuild", autocompletion=_complete_dataset),
    ] = None,
    all_: Annotated[bool, typer.Option("--all", help="Rebuild all datasets")] = False,
    delete_existing: Annotated[
        bool, typer.Option(help="Delete stale cache before rebuilding")
    ] = False,
    yes: Annotated[
        bool,
        typer.Option(
            "--yes", "-y", help="Skip confirmation prompt (required for non-interactive use)"
        ),
    ] = False,
) -> None:
    """Rebuild preprocessed graph caches from raw dataset files."""
    from graphids.config.catalog import dataset_names
    from graphids.core.data.rebuild import rebuild_caches as _rebuild

    datasets = list(dataset_names()) if all_ else (dataset or [])
    if not datasets:
        raise typer.BadParameter("Provide --dataset names or --all")

    if delete_existing and not yes:
        prompt = f"Delete existing cache directories for: {', '.join(datasets)}?"
        if sys.stdin.isatty():
            typer.confirm(prompt, abort=True)
        else:
            raise typer.BadParameter(
                "--delete-existing in a non-interactive shell requires --yes/-y"
            )

    _rebuild(datasets, delete_existing=delete_existing)


@app.command("validate-metadata", rich_help_panel="Data")
def validate_metadata_cli(
    dataset: Annotated[
        str,
        typer.Option(help="Dataset name to validate", autocompletion=_complete_dataset),
    ],
) -> None:
    """Validate cache_metadata.json against v2 schema + catalog expectations."""
    from graphids.config.catalog import cache_dir, lake_root, load_catalog
    from graphids.config.constants import PREPROCESSING_VERSION
    from graphids.core.data.metadata import load_metadata, validate_metadata

    catalog = load_catalog()
    if dataset not in catalog:
        raise typer.BadParameter(f"Unknown dataset {dataset!r}")

    cdir = cache_dir(lake_root(), dataset)
    try:
        meta = load_metadata(cdir)
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"FAIL: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    errors = validate_metadata(
        meta,
        dataset=dataset,
        test_subdirs=catalog[dataset].get("test_subdirs") or [],
        preprocessing_version=PREPROCESSING_VERSION,
    )
    if errors:
        typer.echo(f"FAIL: {len(errors)} validation error(s) for {dataset}:", err=True)
        for err in errors:
            typer.echo(f"  - {err}", err=True)
        raise typer.Exit(code=1)
    splits = list(meta.get("splits") or {})
    typer.echo(f"OK: {dataset} — {len(splits)} splits ({', '.join(splits)})")


