"""Monarch pipeline CLI commands."""

from __future__ import annotations

from typing import Annotated

import typer

from graphids.cli.app import app


@app.command("monarch-run", rich_help_panel="Orchestration")
def monarch_run(
    dataset: Annotated[str, typer.Option(help="Dataset name")] = "hcrl_ch",
    seed: Annotated[int, typer.Option(help="Random seed")] = 42,
    scale: Annotated[str, typer.Option(help="Model scale (small/large)")] = "small",
    fusion_method: Annotated[str, typer.Option("--fusion-method", help="Fusion method")] = "bandit",
    stages: Annotated[
        str, typer.Option(help="Comma-separated stages to run")
    ] = "autoencoder,supervised,fusion",
    conv_type: Annotated[
        str, typer.Option("--conv-type", help="Conv type (gat/gatv2/transformer)")
    ] = "gatv2",
    variational: Annotated[
        bool, typer.Option("--variational/--no-variational", help="VGAE variational mode")
    ] = True,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Print allocation spec only")] = False,
) -> None:
    """Run the 3-stage pipeline in a single SLURM allocation via Monarch."""
    from graphids.monarch.job import pipeline_job_spec

    spec = pipeline_job_spec(scale, fusion_method=fusion_method)

    if dry_run:
        typer.echo(f"Partition:  {spec.partition}")
        typer.echo(f"Time:       {spec.time}")
        typer.echo(f"Memory:     {spec.mem}")
        typer.echo(f"CPUs:       {spec.cpus}")
        typer.echo(f"GPUs/node:  {spec.gpus_per_node}")
        typer.echo(f"Account:    {spec.account}")
        typer.echo(f"Job name:   {spec.job_name}")
        raise typer.Exit()

    from graphids.monarch import available

    if not available():
        typer.echo(
            "Error: monarch is not installed. Install with: uv pip install torchmonarch",
            err=True,
        )
        raise typer.Exit(code=1)

    from graphids.monarch.pipeline import PipelineConfig, run_pipeline

    stage_list = [s.strip() for s in stages.split(",")]
    cfg = PipelineConfig(
        dataset=dataset,
        seed=seed,
        scale=scale,
        fusion_method=fusion_method,
        stages=stage_list,
        conv_type=conv_type,
        variational=variational,
    )
    checkpoints = run_pipeline(cfg)
    for stage_name, ckpt in checkpoints.items():
        typer.echo(f"{stage_name}: {ckpt}")
