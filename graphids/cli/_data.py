"""Data commands: rebuild-caches, stage-data, extract-fusion-states."""

from __future__ import annotations

from typing import Annotated

import typer

from graphids.cli.app import app


@app.command("rebuild-caches", rich_help_panel="Data")
def rebuild_caches(
    dataset: Annotated[
        list[str] | None,
        typer.Option(help="Dataset name(s) to rebuild"),
    ] = None,
    all_: Annotated[bool, typer.Option("--all", help="Rebuild all datasets")] = False,
    delete_existing: Annotated[
        bool, typer.Option(help="Delete stale cache before rebuilding")
    ] = False,
) -> None:
    """Rebuild preprocessed graph caches from raw dataset files."""
    from graphids.config.paths import dataset_names
    from graphids.core.data.cache import rebuild_caches as _rebuild

    datasets = list(dataset_names()) if all_ else (dataset or [])
    if not datasets:
        raise typer.BadParameter("Provide --dataset names or --all")
    _rebuild(datasets, delete_existing=delete_existing)


@app.command("stage-data", rich_help_panel="Data")
def stage_data(
    cache: Annotated[bool, typer.Option(help="Stage cached (preprocessed) data only")] = False,
    raw: Annotated[bool, typer.Option(help="Stage raw data only")] = False,
    skip_tmpdir: Annotated[bool, typer.Option(help="Skip TMPDIR staging")] = False,
    dataset: Annotated[str, typer.Option(help="Single dataset to stage")] = "",
) -> None:
    """Stage data from NFS to scratch/TMPDIR for fast training I/O."""
    from graphids.slurm.staging import stage_data as _stage

    _stage(cache_only=cache, raw_only=raw, skip_tmpdir=skip_tmpdir, dataset=dataset)


@app.command("extract-fusion-states", rich_help_panel="Data")
def extract_fusion_states(
    vgae_ckpt: Annotated[str, typer.Option(help="Path to VGAE checkpoint")],
    gat_ckpt: Annotated[str, typer.Option(help="Path to GAT checkpoint")],
    dataset: Annotated[str, typer.Option(help="Dataset name")],
    output_dir: Annotated[str, typer.Option(help="Output directory for fusion states")],
    max_samples: Annotated[int, typer.Option(help="Max training samples")] = 150_000,
    max_val_samples: Annotated[int, typer.Option(help="Max validation samples")] = 30_000,
    batch_size: Annotated[int, typer.Option(help="Batch size for extraction")] = 256,
    seed: Annotated[int, typer.Option(help="Random seed")] = 42,
    window_size: Annotated[int, typer.Option(help="Sliding window size")] = 100,
    stride: Annotated[int, typer.Option(help="Sliding window stride")] = 100,
    val_fraction: Annotated[float, typer.Option(help="Validation split fraction")] = 0.2,
) -> None:
    """Extract VGAE + GAT latent states for fusion model training."""
    from graphids.core.models.fusion.states import extract_fusion_states as _extract

    _extract(
        vgae_ckpt=vgae_ckpt,
        gat_ckpt=gat_ckpt,
        dataset=dataset,
        output_dir=output_dir,
        max_samples=max_samples,
        max_val_samples=max_val_samples,
        batch_size=batch_size,
        seed=seed,
        window_size=window_size,
        stride=stride,
        val_fraction=val_fraction,
    )
