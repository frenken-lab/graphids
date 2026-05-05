"""Read-only views over rendered plans — list distinct plan_ids and
drill into one plan's per-row state.

Pure queries. No submission, no MLflow writes, no SLURM mutations —
``single-submission-primitive.md`` blocks read-and-act helpers, not
read-only views.

Two data sources joined per plan:
- **MLflow** — runs with ``tags.graphids.plan_id = <id>`` (open at
  fit-start; covers all rows that reached ``trainer.fit``).
- **sacct** — SLURM jobs with ``--comment=graphids.plan_id=<id>``
  (covers queued / pending / failed-before-fit jobs that never opened
  an MLflow run).
"""

from __future__ import annotations

import shlex
import subprocess
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from graphids.cli.app import app

plans_app = typer.Typer(
    name="plans",
    help="Read-only views over rendered plans (list / status).",
    no_args_is_help=True,
)
app.add_typer(plans_app, name="plans")

_console = Console()


def _graphids_experiment_ids() -> list[str]:
    from mlflow.tracking import MlflowClient

    from graphids._mlflow import configure_tracking_uri

    configure_tracking_uri()
    client = MlflowClient()
    return [
        e.experiment_id
        for e in client.search_experiments(filter_string="name LIKE 'graphids/%'")
    ]


def _sacct_for_plan(plan_id: str, days: int = 7) -> list[dict[str, str]]:
    """Pull sacct rows whose ``Comment`` matches ``graphids.plan_id={id}``.

    Returns ``[]`` on sacct error (login node without SLURM accounting).
    """
    cmd = [
        "sacct", "-P", "--noheader", f"--starttime=now-{days}days",
        "-o", "JobID,JobName,State,Elapsed,Comment",
        "--user", subprocess.run(["whoami"], capture_output=True, text=True).stdout.strip(),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=10).stdout
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return []
    needle = f"graphids.plan_id={plan_id}"
    rows: list[dict[str, str]] = []
    for line in out.splitlines():
        if needle not in line:
            continue
        parts = line.split("|")
        if len(parts) < 5 or "." in parts[0]:  # skip .batch / .extern step rows
            continue
        rows.append({"jid": parts[0], "name": parts[1], "state": parts[2], "elapsed": parts[3]})
    return rows


@plans_app.command("list")
def list_plans(
    days: Annotated[int, typer.Option("--days", help="Look-back window in days")] = 7,
) -> None:
    """Distinct ``plan_id``s seen in MLflow within the last N days, newest first."""
    from mlflow.tracking import MlflowClient

    from graphids._mlflow import configure_tracking_uri

    configure_tracking_uri()
    client = MlflowClient()
    exp_ids = _graphids_experiment_ids()
    if not exp_ids:
        _console.print("[yellow]no graphids/* experiments found[/yellow]")
        raise typer.Exit(0)

    cutoff_ms = int((__import__("time").time() - days * 86400) * 1000)
    runs = client.search_runs(
        experiment_ids=exp_ids,
        filter_string=f"tags.`graphids.plan_id` != '' and attributes.start_time > {cutoff_ms}",
        max_results=1000,
        order_by=["attributes.start_time DESC"],
    )

    seen: dict[str, dict[str, str | int]] = {}
    for r in runs:
        pid = r.data.tags.get("graphids.plan_id", "")
        if not pid:
            continue
        entry = seen.setdefault(pid, {
            "plan_id": pid,
            "first_seen": r.info.start_time,
            "n_runs": 0,
            "dataset": r.data.tags.get("graphids.dataset", ""),
            "group": r.data.tags.get("graphids.group", ""),
        })
        entry["n_runs"] = int(entry["n_runs"]) + 1  # type: ignore[operator]

    if not seen:
        _console.print(f"[yellow]no plans in last {days} days[/yellow]")
        raise typer.Exit(0)

    table = Table(title=f"plans (last {days}d)", show_lines=False)
    table.add_column("plan_id", style="cyan", no_wrap=True)
    table.add_column("dataset")
    table.add_column("group")
    table.add_column("# runs", justify="right")
    table.add_column("first seen (UTC)")
    from datetime import UTC, datetime
    for pid, e in sorted(seen.items(), key=lambda kv: -int(kv[1]["first_seen"])):  # type: ignore[arg-type]
        ts = datetime.fromtimestamp(int(e["first_seen"]) / 1000, UTC).isoformat(timespec="seconds")
        table.add_row(pid, str(e["dataset"]), str(e["group"]), str(e["n_runs"]), ts)
    _console.print(table)


@plans_app.command("status")
def plan_status(
    plan_id: Annotated[str, typer.Argument(help="plan_id (uuid7) to inspect")],
    days: Annotated[int, typer.Option("--days", help="sacct look-back window")] = 7,
) -> None:
    """Per-row state for one plan — joins MLflow runs + sacct jobs."""
    from mlflow.tracking import MlflowClient

    from graphids._mlflow import build_search_filter, configure_tracking_uri

    configure_tracking_uri()
    client = MlflowClient()

    exp_ids = _graphids_experiment_ids()
    runs = client.search_runs(
        experiment_ids=exp_ids,
        filter_string=build_search_filter(plan_id=plan_id),
        max_results=1000,
        order_by=["attributes.start_time ASC"],
    ) if exp_ids else []

    sacct_rows = _sacct_for_plan(plan_id, days=days)

    if not runs and not sacct_rows:
        _console.print(f"[red]no MLflow runs or sacct jobs found for plan_id={plan_id}[/red]")
        raise typer.Exit(1)

    # MLflow side
    if runs:
        ml_table = Table(title=f"MLflow runs (plan_id={plan_id})")
        for col in ("run_name", "phase", "variant", "status", "duration_s"):
            ml_table.add_column(col)
        for r in runs:
            duration = (
                f"{(r.info.end_time - r.info.start_time) / 1000:.1f}"
                if r.info.end_time else "—"
            )
            ml_table.add_row(
                r.info.run_name or "",
                r.data.tags.get("graphids.phase", ""),
                r.data.tags.get("graphids.variant", ""),
                r.info.status,
                duration,
            )
        _console.print(ml_table)

    # SLURM side
    if sacct_rows:
        sl_table = Table(title=f"SLURM jobs (plan_id={plan_id}, last {days}d)")
        for col in ("jid", "job_name", "state", "elapsed"):
            sl_table.add_column(col)
        for s in sacct_rows:
            sl_table.add_row(s["jid"], s["name"], s["state"], s["elapsed"])
        _console.print(sl_table)
    else:
        _console.print(f"[dim](no sacct rows in last {days}d — login-node sacct may be unavailable)[/dim]")
