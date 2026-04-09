"""Monarch pipeline CLI commands."""

from __future__ import annotations

from pathlib import Path
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
    from graphids.orchestrate.job import chain_job_spec
    from graphids.orchestrate.pipeline import build_pipeline_stages, run_chain
    from graphids.orchestrate.schemas import PipelineConfig

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

    from dataclasses import replace

    stage_cfgs = build_pipeline_stages(cfg)
    spec = chain_job_spec(stage_cfgs, dataset=dataset)
    if partition:
        spec = replace(spec, partition=partition)
    if time:
        spec = replace(spec, time=time)

    if dry_run:
        if overrides:
            typer.echo(f"Overrides:  {overrides}")
        _print_spec(spec)
        raise typer.Exit()

    _check_monarch()

    from graphids.orchestrate.sweep import ChainSpec

    chain = ChainSpec(
        chain_id=f"pipeline_{dataset}_s{seed}",
        stages=stage_cfgs,
        dataset=dataset,
        seed=seed,
    )
    checkpoints = run_chain(
        chain, max_retries=cfg.max_retries, lake_root=cfg.lake_root, job_spec_override=spec
    )
    for stage_name, ckpt in checkpoints.items():
        typer.echo(f"{stage_name}: {ckpt}")


@app.command("monarch-sweep", rich_help_panel="Orchestration")
def monarch_sweep(
    recipe: Annotated[str, typer.Option(help="Path to recipe jsonnet file")],
    datasets: Annotated[
        str, typer.Option(help="Comma-separated dataset names (default: from registry)")
    ] = "",
    seeds: Annotated[str, typer.Option(help="Comma-separated seeds")] = "42",
    max_concurrent: Annotated[
        int, typer.Option("--max-concurrent", help="Max parallel chains (0=all)")
    ] = 0,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Plan only, print chains and specs")
    ] = False,
) -> None:
    """Run a recipe sweep via Monarch (replaces dg launch)."""
    from graphids.orchestrate.sweep import plan_chains

    recipe_path = str(Path(recipe).resolve())
    seed_list = [int(s.strip()) for s in seeds.split(",")]
    if datasets:
        dataset_list = [d.strip() for d in datasets.split(",")]
    else:
        from graphids.config.topology import dataset_names

        dataset_list = dataset_names()

    chains = plan_chains(recipe_path, dataset_list, seed_list)

    if dry_run:
        from graphids.orchestrate.job import chain_job_spec

        typer.echo(f"Recipe:   {recipe}")
        typer.echo(f"Datasets: {dataset_list}")
        typer.echo(f"Seeds:    {seed_list}")
        typer.echo(f"Chains:   {len(chains)}")
        typer.echo()
        for chain in chains:
            spec = chain_job_spec(chain.stages, dataset=chain.dataset)
            stage_names = [s.stage for s in chain.stages]
            typer.echo(f"  {chain.chain_id}")
            typer.echo(f"    stages:    {' → '.join(stage_names)}")
            typer.echo(f"    partition: {spec.partition}  time: {spec.time}  mem: {spec.mem}")
        raise typer.Exit()

    _check_monarch()

    from graphids.orchestrate.pipeline import run_sweep
    from graphids.orchestrate.schemas import SweepConfig

    cfg = SweepConfig(
        recipe_path=recipe_path,
        datasets=dataset_list,
        seeds=seed_list,
        max_retries=2,
        max_concurrent=max_concurrent,
    )
    results = run_sweep(cfg)

    ok = sum(1 for v in results.values() if isinstance(v, dict))
    failed = len(results) - ok
    typer.echo(f"\nSweep complete: {ok}/{len(results)} succeeded, {failed} failed")

    for chain_id, v in results.items():
        if isinstance(v, str):
            typer.echo(f"  FAILED: {chain_id}: {v}", err=True)
