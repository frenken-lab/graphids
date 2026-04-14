"""Pipeline CLI — runs the full 3-stage chain in-process."""

from __future__ import annotations

from typing import Annotated

import typer

from graphids.cli.app import (
    _complete_conv_type,
    _complete_dataset,
    _complete_fusion_method,
    _complete_loss_fn,
    _complete_scale,
    _parse_kv_pair,
    app,
)
from graphids.orchestrate.config import PipelineConfig

# Pydantic is the single source of truth for defaults (axes.json → PipelineConfig).
# Instantiating a default ``PipelineConfig`` once exposes every field's default via
# plain attribute access, so the Typer options below never duplicate the axes dict.
_defaults = PipelineConfig()


@app.command("pipeline-run", rich_help_panel="Orchestration")
def pipeline_run(
    dataset: Annotated[
        str,
        typer.Option(help="Dataset name", autocompletion=_complete_dataset),
    ] = _defaults.dataset,
    seed: Annotated[int, typer.Option(help="Random seed")] = _defaults.seed,
    scale: Annotated[
        str,
        typer.Option(help="Model scale", autocompletion=_complete_scale),
    ] = _defaults.scale,
    fusion_method: Annotated[
        str,
        typer.Option(
            "--fusion-method", help="Fusion method", autocompletion=_complete_fusion_method
        ),
    ] = _defaults.fusion_method,
    stages: Annotated[str, typer.Option(help="Comma-separated stages to run")] = ",".join(
        _defaults.stages
    ),
    conv_type: Annotated[
        str,
        typer.Option("--conv-type", help="Conv type", autocompletion=_complete_conv_type),
    ] = _defaults.conv_type,
    variational: Annotated[
        bool, typer.Option("--variational/--no-variational", help="VGAE variational mode")
    ] = _defaults.variational,
    loss_fn: Annotated[
        str,
        typer.Option("--loss-fn", help="Loss function", autocompletion=_complete_loss_fn),
    ] = _defaults.loss_fn,
    lake_root: Annotated[str, typer.Option(help="Lake root override")] = _defaults.lake_root,
    max_retries: Annotated[
        int, typer.Option(help="Per-stage retry budget")
    ] = _defaults.max_retries,
    trainer_override: Annotated[
        list[str] | None,
        typer.Option(
            "--trainer-override",
            "-O",
            parser=_parse_kv_pair,
            metavar="KEY=JSON",
            help="Dotted trainer override (e.g. trainer.max_epochs=3)",
        ),
    ] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Print stage list only")] = False,
) -> None:
    """Run the full pipeline in-process inside the current SLURM allocation."""
    from graphids.orchestrate import build_pipeline_stages, run_pipeline

    overrides = dict(trainer_override or [])
    stage_list = [s.strip() for s in stages.split(",")]

    cfg = PipelineConfig(
        dataset=dataset,
        seed=seed,
        scale=scale,
        fusion_method=fusion_method,
        stages=stage_list,
        conv_type=conv_type,
        variational=variational,
        loss_fn=loss_fn,
        lake_root=lake_root,
        max_retries=max_retries,
        tla_overrides=overrides,
    )

    if dry_run:
        stage_cfgs = build_pipeline_stages(cfg)
        typer.echo(f"Stages:     {[s.asset_name for s in stage_cfgs]}")
        if overrides:
            typer.echo(f"Overrides:  {overrides}")
        raise typer.Exit()

    result = run_pipeline(cfg)
    for stage_name, ckpt in result.checkpoints_by_stage().items():
        typer.echo(f"{stage_name}: {ckpt}")
