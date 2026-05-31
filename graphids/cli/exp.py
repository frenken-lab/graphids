"""New experiment-facing CLI.

This is the first replacement surface for the old row/submit mental model.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Annotated

import mlflow
import typer
from rich.console import Console
from rich.table import Table

from graphids.cli.app import app
from graphids.exp.config import ExperimentConfig
from graphids.exp.journal import load_manifest
from graphids.exp.results import query_result_view, result_rows_as_json, sort_rows
from graphids.exp.runtime import launch_run, summarize_run
from graphids.exp.slurm import submit_experiment

exp_app = typer.Typer(
    name="exp",
    help="Experiment manifests, status, and future Ray/Hydra launch surface.",
    no_args_is_help=True,
)
app.add_typer(exp_app, name="exp")
console = Console()


@exp_app.command("config")
def config(
    path: Annotated[Path, typer.Argument(help="YAML config file to validate as ExperimentConfig")],
) -> None:
    """Load a YAML config through OmegaConf and validate it as ``ExperimentConfig``."""
    console.print_json(data=ExperimentConfig.from_yaml(path).model_dump(mode="json"))


@exp_app.command("status")
def status(
    run_dir: Annotated[Path, typer.Argument(help="Run directory to inspect")],
) -> None:
    """Print manifest + latest event summary for one run."""
    summary = summarize_run(run_dir)
    if summary is None:
        raise typer.BadParameter(f"no manifest found in {run_dir}")

    table = Table(title="run status", show_lines=False)
    table.add_column("field")
    table.add_column("value")
    table.add_row("name", summary.name)
    table.add_row("stage", summary.stage)
    table.add_row("status", summary.status)
    table.add_row("last_event", summary.last_event or "—")
    table.add_row("error", summary.error or "—")
    table.add_row("run_dir", summary.run_dir)
    table.add_row("git_sha", summary.extra.get("git_sha", "—"))
    table.add_row("run_id", summary.extra.get("run_id", "—"))
    console.print(table)


@exp_app.command("manifest")
def manifest(
    run_dir: Annotated[Path, typer.Argument(help="Run directory to inspect")],
) -> None:
    """Dump the manifest JSON for a run."""
    manifest = load_manifest(run_dir)
    if manifest is None:
        raise typer.BadParameter(f"no manifest found in {run_dir}")
    console.print_json(data=manifest.model_dump(mode="json"))


@exp_app.command("launch")
def launch(
    path: Annotated[Path, typer.Argument(help="YAML experiment config")],
) -> None:
    """Launch one experiment config through the new primitive surface."""
    exp_cfg = ExperimentConfig.from_yaml(path)
    run = exp_cfg.build_run(
        name=exp_cfg.experiment_name,
        stage=exp_cfg.stage,
        config=exp_cfg.config,
    )
    result = launch_run(run)
    if result is not None:
        payload = asdict(result) if is_dataclass(result) else {"result": str(result)}
        console.print_json(data=payload)


@exp_app.command("submit")
def submit(
    path: Annotated[Path, typer.Argument(help="YAML experiment config")],
    cluster: Annotated[str | None, typer.Option("--cluster", "-C", help="SLURM cluster override")] = None,
    partition: Annotated[str | None, typer.Option("--partition", "-p", help="SLURM partition override")] = None,
    time_limit: Annotated[str | None, typer.Option("--time", "-t", help="SLURM walltime override")] = None,
    gres: Annotated[str | None, typer.Option("--gres", help="SLURM gres override")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Print the sbatch script without submitting")] = False,
) -> None:
    """Submit one experiment YAML as a SLURM batch job."""
    exp_cfg = ExperimentConfig.from_yaml(path)
    result = submit_experiment(
        exp_cfg,
        path,
        cluster=cluster,
        partition=partition,
        time_limit=time_limit,
        gres=gres,
        dry_run=dry_run,
    )
    if dry_run:
        typer.echo(result.script, nl=False)
        return
    console.print_json(
        data={
            "job_id": result.job_id,
            "script_path": str(result.script_path),
            "command": list(result.command),
            "stdout": result.stdout.strip(),
        }
    )


@exp_app.command("results")
def results(
    view: Annotated[str, typer.Option("--view", "-v", help="Result view in configs/result_views.yml")] = "fusion",
    dataset: Annotated[
        list[str] | None,
        typer.Option("--dataset", "-d", help="Dataset to query; repeat for multiple datasets"),
    ] = None,
    variant: Annotated[
        list[str] | None,
        typer.Option("--variant", help="Variant to include; repeat for multiple variants"),
    ] = None,
    all_runs: Annotated[bool, typer.Option("--all", help="Show all matching runs, not latest per variant")] = False,
    tracking_uri: Annotated[str | None, typer.Option("--tracking-uri", help="Override MLflow tracking URI")] = None,
    output_format: Annotated[str, typer.Option("--format", help="table or json")] = "table",
) -> None:
    """Query configured MLflow result views."""
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    datasets = dataset or ["hcrl_sa", "set_01", "set_02", "set_03", "set_04"]
    rows = sort_rows(
        query_result_view(
            view=view,
            datasets=datasets,
            variants=variant,
            latest=not all_runs,
        )
    )
    if output_format == "json":
        console.print_json(data=result_rows_as_json(rows))
        return
    if output_format != "table":
        raise typer.BadParameter("--format must be table or json")

    table = Table(title=f"{view} results", show_lines=False)
    base_cols = ["dataset", "variant", "status", "run_id"]
    metric_cols = list(rows[0].metrics) if rows else []
    for col in [*base_cols, *metric_cols]:
        table.add_column(col)
    for row in rows:
        payload = row.flat()
        values = []
        for col in [*base_cols, *metric_cols]:
            value = payload.get(col)
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            elif value is None:
                values.append("n/a")
            else:
                values.append(str(value))
        table.add_row(*values)
    console.print(table)
