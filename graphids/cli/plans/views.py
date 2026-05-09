"""Read-only ``plans`` commands: available / describe / list / show / where / health.

All MLflow access goes through ``_mlflow_ctx()`` and
``_mlflow.build_search_filter(...)``; no ad-hoc filter strings.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from graphids.cli.plans import _mlflow_ctx, console, plans_app


@plans_app.command("available")
def list_available() -> None:
    """List plan modules under ``graphids.plan.plans`` (dotted names).

    Pure filesystem walk; no MLflow, no SLURM.
    """
    import pkgutil

    import graphids.plan.plans as pkg

    found = []
    for info in pkgutil.walk_packages(pkg.__path__, prefix=pkg.__name__ + "."):
        if info.ispkg:
            continue
        dotted = info.name.removeprefix(pkg.__name__ + ".")
        path = (
            info.module_finder.path  # type: ignore[union-attr]
            + "/"
            + info.name.rsplit(".", 1)[-1]
            + ".py"
        )
        found.append((dotted, path))
    if not found:
        console.print("[yellow]no plan modules found[/yellow]")
        raise typer.Exit(0)

    table = Table(title="available plan modules", show_lines=False)
    table.add_column("dotted name", style="cyan", no_wrap=True)
    table.add_column("path")
    for dotted, path in sorted(found):
        table.add_row(dotted, path)
    console.print(table)


@plans_app.command("describe")
def describe_plan(
    plan: Annotated[str, typer.Argument(help="Dotted module name (e.g. ablations.supervised)")],
    dataset: Annotated[str, typer.Option("--dataset", "-d", help="Dataset")],
    seed: Annotated[int, typer.Option("--seed", "-s", help="Seed")] = 42,
) -> None:
    """Render the plan and print the row table — no JSON write, no submit."""
    from graphids.plan.render import render_plan

    plan_obj = render_plan(plan, dataset=dataset, seed=seed, created_at="describe")
    console.print(
        f"[bold]{plan}[/bold]  dataset={dataset}  seed={seed}  [dim]({len(plan_obj)} rows)[/dim]"
    )
    table = Table(show_lines=False)
    for col in ("name", "action", "variant", "mode", "length"):
        table.add_column(col)
    for r in plan_obj.rows:
        meta = getattr(r, "meta", None)
        table.add_row(
            r.name,
            r.action,
            getattr(meta, "variant", "—") if meta else "—",
            r.resources.mode,
            r.resources.length,
        )
    console.print(table)


@plans_app.command("list")
def list_plans(
    days: Annotated[int, typer.Option("--days", help="Look-back window in days")] = 7,
) -> None:
    """Distinct ``plan_id``s seen in MLflow within the last N days, newest first."""
    client, exp_ids = _mlflow_ctx(exit_if_empty=False)
    if not exp_ids:
        console.print("[yellow]no graphids/* experiments found[/yellow]")
        raise typer.Exit(0)

    cutoff_ms = int((time.time() - days * 86400) * 1000)
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
        entry = seen.setdefault(
            pid,
            {
                "plan_id": pid,
                "first_seen": r.info.start_time,
                "n_runs": 0,
                "dataset": r.data.tags.get("graphids.dataset", ""),
                "group": r.data.tags.get("graphids.group", ""),
            },
        )
        entry["n_runs"] = int(entry["n_runs"]) + 1  # type: ignore[operator]

    if not seen:
        console.print(f"[yellow]no plans in last {days} days[/yellow]")
        raise typer.Exit(0)

    table = Table(title=f"plans (last {days}d)", show_lines=False)
    table.add_column("plan_id", style="cyan", no_wrap=True)
    table.add_column("dataset")
    table.add_column("group")
    table.add_column("# runs", justify="right")
    table.add_column("first seen (UTC)")
    for pid, e in sorted(seen.items(), key=lambda kv: -int(kv[1]["first_seen"])):  # type: ignore[arg-type]
        ts = datetime.fromtimestamp(int(e["first_seen"]) / 1000, UTC).isoformat(timespec="seconds")
        table.add_row(pid, str(e["dataset"]), str(e["group"]), str(e["n_runs"]), ts)
    console.print(table)


@plans_app.command("show")
def plan_show(
    plan_id: Annotated[str, typer.Argument(help="plan_id (uuid7) to inspect")],
    status: Annotated[
        str | None,
        typer.Option("--status", help="Filter by MLflow status (FINISHED|FAILED|RUNNING|...)"),
    ] = None,
    names_only: Annotated[
        bool,
        typer.Option("--names-only", help="Print row names only (one per line, machine-readable)."),
    ] = False,
) -> None:
    """Per-row state for one plan — single consolidated MLflow query.

    ``--names-only`` is the shell-composition hook for retry loops::

        graphids plans show $PID --status FAILED --names-only \\
          | xargs -I{} graphids submit --plan plan.json --row-name {} -C pitzer
    """
    from graphids._mlflow import build_search_filter

    client, exp_ids = _mlflow_ctx()

    runs = client.search_runs(
        experiment_ids=exp_ids,
        filter_string=build_search_filter(plan_id=plan_id, status=status),
        max_results=1000,
        order_by=["attributes.start_time ASC"],
    )
    if not runs:
        if names_only:
            raise typer.Exit(0)
        msg = f"no MLflow runs found for plan_id={plan_id}"
        if status:
            msg += f" with status={status}"
        console.print(f"[red]{msg}[/red]")
        raise typer.Exit(1)

    if names_only:
        for r in runs:
            name = r.data.tags.get("graphids.row_name") or (r.info.run_name or "")
            print(name)
        return

    head_tags = runs[0].data.tags
    console.print(
        f"[bold]plan_id[/bold]: [cyan]{plan_id}[/cyan]  "
        f"[bold]plan_module[/bold]: {head_tags.get('graphids.plan_module', '?')}  "
        f"[bold]git_sha[/bold]: {head_tags.get('graphids.git_sha', '?')}"
    )
    console.print(
        f"[bold]dataset[/bold]: {head_tags.get('graphids.dataset', '?')}  "
        f"[bold]seed[/bold]: {head_tags.get('graphids.seed', '?')}  "
        f"[bold]n_runs[/bold]: {len(runs)}"
        + (f"  [bold]filter[/bold]: status={status}" if status else "")
    )

    table = Table(show_lines=False)
    for col in ("row_name", "phase", "variant", "status", "started", "duration"):
        table.add_column(col)
    for r in runs:
        duration = (
            f"{(r.info.end_time - r.info.start_time) / 1000:.1f}s" if r.info.end_time else "—"
        )
        started = datetime.fromtimestamp(r.info.start_time / 1000, UTC).strftime("%m-%d %H:%M")
        row_name = r.data.tags.get("graphids.row_name") or (r.info.run_name or "")
        table.add_row(
            row_name,
            r.data.tags.get("graphids.phase", ""),
            r.data.tags.get("graphids.variant", ""),
            r.info.status,
            started,
            duration,
        )
    console.print(table)


@plans_app.command("where")
def plan_where(
    plan_id: Annotated[str, typer.Argument(help="plan_id (uuid7)")],
    row: Annotated[
        str | None,
        typer.Option("--row", "-r", help="Row name. Omit to print all rows of this plan."),
    ] = None,
) -> None:
    """Print on-disk locations + MLflow run_id for one (or all) rows of a plan."""
    from graphids._mlflow import build_search_filter

    client, exp_ids = _mlflow_ctx()

    runs = client.search_runs(
        experiment_ids=exp_ids,
        filter_string=build_search_filter(plan_id=plan_id, row_name=row),
        max_results=1000,
        order_by=["attributes.start_time ASC"],
    )
    if not runs:
        console.print(
            f"[red]no MLflow runs for plan_id={plan_id}" + (f" row={row}" if row else "") + "[/red]"
        )
        raise typer.Exit(1)

    for r in runs:
        run_dir = r.data.tags.get("graphids.run_dir", "")
        row_name = r.data.tags.get("graphids.row_name") or (r.info.run_name or "")
        ckpt = Path(run_dir) / "checkpoints" / "best_model.ckpt"
        scripts_dir = Path(run_dir) / ".parsl_scripts"
        stderr_files = (
            sorted(scripts_dir.glob("*.stderr"), reverse=True) if scripts_dir.exists() else []
        )
        stderr = stderr_files[0] if stderr_files else None

        console.print(
            f"[bold cyan]{row_name}[/bold cyan]  "
            f"[dim]({r.data.tags.get('graphids.phase', '?')} / {r.info.status})[/dim]"
        )
        console.print(f"  run_dir:  {run_dir or '—'}")
        console.print(
            f"  ckpt:     {ckpt} {'[green]✓[/green]' if ckpt.exists() else '[red]✗[/red]'}"
        )
        console.print(f"  stderr:   {stderr or '—'}")
        console.print(f"  mlflow:   run_id={r.info.run_id} status={r.info.status}")
        if r.data.metrics:
            best = {k: v for k, v in r.data.metrics.items() if "auroc" in k.lower()}
            if best:
                console.print("  metrics:  " + "  ".join(f"{k}={v:.4f}" for k, v in best.items()))


def _slurm_job_info() -> dict[str, dict[str, str]] | None:
    """Query squeue for all user jobs.

    Returns {job_id: {state, elapsed, left, partition}}, or None if squeue is
    unavailable (not on a SLURM system).  An empty dict means squeue succeeded
    but no jobs are queued — any MLflow RUNNING run is therefore a zombie.
    """
    import os
    import subprocess

    try:
        raw = subprocess.check_output(
            [
                "squeue",
                f"--user={os.environ['USER']}",
                "-h",
                "--format=%i|%T|%M|%L|%P",
                "-M",
                "all",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except FileNotFoundError:
        return None  # squeue not installed
    except Exception:
        return None

    result: dict[str, dict[str, str]] = {}
    for line in raw.splitlines():
        parts = line.strip().split("|")
        if len(parts) >= 5:
            jid, state, elapsed, left, partition = parts[:5]
            result[jid.strip()] = {
                "state": state.strip(),
                "elapsed": elapsed.strip(),
                "left": left.strip(),
                "partition": partition.strip(),
            }
    return result


def _best_metric(metrics: dict[str, float], action: str) -> str:
    """Most informative single-number summary from a run's metric dict."""
    if action == "fit":
        if "val_auroc" in metrics:
            return f"val_auroc={metrics['val_auroc']:.4f}"
    else:
        if "auroc_macro" in metrics:
            return f"auroc_macro={metrics['auroc_macro']:.4f}"
        if "auroc_weighted" in metrics:
            return f"auroc_w={metrics['auroc_weighted']:.4f}"
    # fallback: first top-level auroc key (not nested under a subtest path)
    for k, v in metrics.items():
        if "auroc" in k and not k.startswith("system/") and "/" not in k:
            return f"{k}={v:.4f}"
    return ""


def _last_error(stderr_path: str) -> str:
    """Last meaningful error line from a SLURM stderr file, truncated to 100 chars."""
    try:
        lines = [
            ln.rstrip()
            for ln in Path(stderr_path).read_text(errors="replace").splitlines()
            if ln.strip()
        ]
        for line in reversed(lines):
            if any(
                kw in line for kw in ("Error", "Exception", "CANCELLED", "slurmstepd", "Killed")
            ):
                return line.strip()[:100]
        return lines[-1][:100] if lines else "—"
    except Exception:
        return "—"


@plans_app.command("health")
def plan_health(
    plan: Annotated[Path, typer.Option("--plan", "-p", help="Rendered plan.json")],
    tail: Annotated[
        int,
        typer.Option(
            "--tail", "-t", help="Lines of stderr to tail for FAILED/ZOMBIE (0 = one-liner only)"
        ),
    ] = 0,
) -> None:
    """Cross-reference expected rows (rendered JSON) against MLflow + SLURM queue.

    Row health labels:

    \b
    FINISHED  MLflow FINISHED — best metric + checkpoint indicator
    FAILED    MLflow FAILED   — last error line from stderr
    RUNNING   SLURM job active — elapsed / time-left / partition
    ZOMBIE    MLflow RUNNING, SLURM job gone — last error line
    MISSING   no MLflow run yet — test rows show whether fit ckpt is ready

    Use --tail N to dump the last N stderr lines for FAILED/ZOMBIE rows.
    """
    import json
    from collections import Counter

    from graphids._mlflow import build_search_filter

    plan_data = json.loads(plan.read_text())
    plan_id = plan_data["plan_id"]
    first_meta = plan_data["rows"][0]["meta"] if plan_data["rows"] else {}
    dataset, seed = first_meta.get("dataset", "?"), first_meta.get("seed", "?")
    row_meta: dict[str, dict] = {r["name"]: r for r in plan_data["rows"]}
    expected = list(row_meta)

    # MLflow — most-recent run per row_name
    client, exp_ids = _mlflow_ctx()
    mlruns = client.search_runs(
        experiment_ids=exp_ids,
        filter_string=build_search_filter(plan_id=plan_id),
        max_results=1000,
        order_by=["attributes.start_time DESC"],
    )
    mlflow_by_row: dict[str, object] = {}
    for r in mlruns:
        name = r.data.tags.get("graphids.row_name", "")
        if name and name not in mlflow_by_row:
            mlflow_by_row[name] = r

    # SLURM job info: None = not on SLURM, {} = on SLURM / queue empty
    slurm_jobs = _slurm_job_info()

    _COLORS = {
        "FINISHED": "green",
        "FAILED": "red",
        "RUNNING": "cyan",
        "ZOMBIE": "yellow",
        "MISSING": "dim",
    }

    classified: list[tuple[str, str, str, str, str]] = []  # name,action,health,duration,detail
    for name in expected:
        action = row_meta[name]["action"]
        plan_run_dir = row_meta[name].get("identity", {}).get("run_dir", "")
        mr = mlflow_by_row.get(name)

        if mr is None:
            detail = ""
            if action == "test" and plan_run_dir:
                ckpt = Path(plan_run_dir, "checkpoints", "best_model.ckpt")
                detail = "ckpt [green]✓[/green]" if ckpt.exists() else "ckpt [red]✗[/red]"
            classified.append((name, action, "MISSING", "—", detail))
            continue

        job_id = mr.data.tags.get("slurm.job_id", "—")
        run_dir = mr.data.tags.get("graphids.run_dir", "")
        duration = (
            f"{(mr.info.end_time - mr.info.start_time) / 1000:.0f}s" if mr.info.end_time else "—"
        )

        if mr.info.status == "FINISHED":
            health = "FINISHED"
            metrics = {k: v for k, v in mr.data.metrics.items() if not k.startswith("system/")}
            ckpt = Path(run_dir, "checkpoints", "best_model.ckpt") if run_dir else None
            ckpt_flag = (
                "  ckpt [green]✓[/green]" if (ckpt and ckpt.exists()) else "  ckpt [red]✗[/red]"
            )
            detail = (_best_metric(metrics, action) or "") + ckpt_flag

        elif mr.info.status == "FAILED":
            health = "FAILED"
            stderrs = (
                sorted(Path(run_dir, ".parsl_scripts").glob("*.stderr"), reverse=True)
                if run_dir
                else []
            )
            detail = f"↳ {_last_error(str(stderrs[0]))}" if stderrs else "no log found"

        elif mr.info.status == "RUNNING":
            jinfo = slurm_jobs.get(job_id) if slurm_jobs is not None else None
            if slurm_jobs is None:
                health, detail = "RUNNING", f"job={job_id}  (squeue unavailable)"
            elif jinfo is not None:
                health = "RUNNING"
                detail = f"elapsed={jinfo['elapsed']}  left={jinfo['left']}  [{jinfo['partition']}]"
            else:
                health = "ZOMBIE"
                stderrs = (
                    sorted(Path(run_dir, ".parsl_scripts").glob("*.stderr"), reverse=True)
                    if run_dir
                    else []
                )
                err = f"↳ {_last_error(str(stderrs[0]))}" if stderrs else "no log"
                detail = f"job={job_id}  {err}"
        else:
            health, detail = mr.info.status, ""

        classified.append((name, action, health, duration, detail))

    console.print(
        f"[bold]plan_id:[/bold] [cyan]{plan_id}[/cyan]  "
        f"dataset={dataset}  seed={seed}  plan={plan.name}"
    )
    table = Table(show_lines=False)
    for col in ("row_name", "action", "health", "duration", "detail"):
        table.add_column(col, no_wrap=(col == "row_name"))
    for name, action, health, duration, detail in classified:
        color = _COLORS.get(health, "white")
        table.add_row(name, action, f"[{color}]{health}[/{color}]", duration, detail)
    console.print(table)

    counts = Counter(c[2] for c in classified)
    console.print("  " + "  ·  ".join(f"{v} {k.lower()}" for k, v in sorted(counts.items())))

    needs_action = [
        (n, a, h, det) for n, a, h, _, det in classified if h in ("FAILED", "ZOMBIE", "MISSING")
    ]
    if not needs_action:
        return

    console.print("\n[bold]action needed:[/bold]")
    for name, action, health, detail in needs_action:
        color = _COLORS[health]
        console.print(f"  [{color}]{health:8s}[/{color}]  {name}/{action}  {detail}")
        if tail > 0 and health in ("FAILED", "ZOMBIE"):
            mr = mlflow_by_row.get(name)
            run_dir = mr.data.tags.get("graphids.run_dir", "") if mr else ""
            if run_dir:
                stderrs = sorted(Path(run_dir, ".parsl_scripts").glob("*.stderr"), reverse=True)
                if stderrs and stderrs[0].exists():
                    for line in stderrs[0].read_text(errors="replace").splitlines()[-tail:]:
                        console.print(f"    [dim]{line}[/dim]")
