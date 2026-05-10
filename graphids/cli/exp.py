"""New experiment-facing CLI.

This is the first replacement surface for the old row/submit mental model.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from graphids.cli.app import app
from graphids.exp.config import ExperimentConfig
from graphids.exp.journal import load_manifest
from graphids.exp.runtime import launch_run, summarize_run

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
