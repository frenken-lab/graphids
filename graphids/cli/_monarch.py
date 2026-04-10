"""Monarch pipeline CLI commands."""

from __future__ import annotations

from typing import Annotated

import typer

from graphids.cli.app import app, parse_tla
from graphids.config.constants import PIPELINE_DEFAULTS

_D = PIPELINE_DEFAULTS


def _print_spec(spec: object) -> None:
    """Print a JobSpec for dry-run output."""
    typer.echo(f"Partition:  {spec.partition}")
    typer.echo(f"Time:       {spec.time}")
    typer.echo(f"Memory:     {spec.mem}")
    typer.echo(f"CPUs:       {spec.cpus}")
    typer.echo(f"GPUs/node:  {spec.gpus_per_node}")
    typer.echo(f"Account:    {spec.account}")
    typer.echo(f"Job name:   {spec.job_name}")


def _check_monarch() -> None:
    """Exit with error if monarch is not installed."""
    from graphids.orchestrate import available

    if not available():
        typer.echo(
            "Error: monarch is not installed. Install with: uv pip install torchmonarch",
            err=True,
        )
        raise typer.Exit(code=1)


@app.command("monarch-run", rich_help_panel="Orchestration")
def monarch_run(
    dataset: Annotated[str, typer.Option(help="Dataset name")] = _D.get("dataset", "hcrl_ch"),
    seed: Annotated[int, typer.Option(help="Random seed")] = _D.get("seed", 42),
    scale: Annotated[str, typer.Option(help="Model scale (small/large)")] = _D.get(
        "scale", "small"
    ),
    fusion_method: Annotated[str, typer.Option("--fusion-method", help="Fusion method")] = _D.get(
        "fusion_method", "bandit"
    ),
    stages: Annotated[str, typer.Option(help="Comma-separated stages to run")] = ",".join(
        _D.get("stages", ["autoencoder", "supervised", "fusion"])
    ),
    conv_type: Annotated[
        str, typer.Option("--conv-type", help="Conv type (gat/gatv2/transformer)")
    ] = _D.get("conv_type", "gatv2"),
    variational: Annotated[
        bool, typer.Option("--variational/--no-variational", help="VGAE variational mode")
    ] = _D.get("variational", True),
    loss_fn: Annotated[
        str, typer.Option("--loss-fn", help="Loss function (focal/ce/weighted_ce)")
    ] = _D.get("loss_fn", "focal"),
    trainer_override: Annotated[
        list[str] | None,
        typer.Option(
            "--trainer-override", "-O", help="Dotted trainer override (e.g. trainer.max_epochs=3)"
        ),
    ] = None,
    partition: Annotated[
        str | None, typer.Option(help="Override SLURM partition (e.g. gpudebug)")
    ] = None,
    time: Annotated[str | None, typer.Option(help="Override wall time (e.g. 1:00:00)")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Print allocation spec only")] = False,
) -> None:
    """Run the 3-stage pipeline in a single SLURM allocation via Monarch."""
    from graphids.orchestrate.allocate import JobSpec
    from graphids.orchestrate.run import PipelineConfig, build_pipeline_stages, run_pipeline

    overrides = parse_tla(trainer_override)
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
        tla_overrides=overrides,
    )

    spec = JobSpec(
        partition=partition or "gpu",
        time=time or "4:00:00",
        mem="40G",
        cpus=8,
    )

    if dry_run:
        # Show what would run without touching Monarch
        stage_cfgs = build_pipeline_stages(cfg)
        typer.echo(f"Stages:     {[s.asset_name for s in stage_cfgs]}")
        if overrides:
            typer.echo(f"Overrides:  {overrides}")
        _print_spec(spec)
        raise typer.Exit()

    _check_monarch()

    result = run_pipeline(cfg, spec)
    for stage_name, ckpt in result.checkpoints_by_stage().items():
        typer.echo(f"{stage_name}: {ckpt}")
    if result.analyzed_assets:
        typer.echo(f"analyzed: {', '.join(result.analyzed_assets)}")
