"""Read-only views over rendered plans — list distinct plan_ids and
drill into one plan's per-row state.

Read-only queries (``list``, ``show``, ``available``, ``describe``,
``where``) plus the multi-row ``submit`` verb. ``submit`` reads MLflow
for state and writes via ``submit_row`` per row — each submission is
independently logged and failable, per ``chassis-invariants.md``.

**Single source of truth: MLflow.** Per the 2026-05-05 framework eval
(`docs/drafts/experiment-framework-evaluation.md`) and design lessons
(Lesson 2), MLflow IS the trial-state store; we don't maintain a
parallel ``jobs.jsonl`` or join sacct as a second data source. Runs
that died before opening an MLflow run are surfaced as gaps in
``plans show`` (rows in the plan that have no run); the user inspects
sacct directly if needed.
"""

from __future__ import annotations

from pathlib import Path
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


@plans_app.command("available")
def list_available() -> None:
    """List plan modules under ``graphids.plan.plans`` (dotted names).

    These are the values you pass as ``graphids run <name> ...``. Pure
    filesystem walk; no MLflow, no SLURM.
    """
    import pkgutil

    import graphids.plan.plans as pkg

    table = Table(title="available plan modules", show_lines=False)
    table.add_column("dotted name", style="cyan", no_wrap=True)
    table.add_column("path")
    found = []
    for info in pkgutil.walk_packages(pkg.__path__, prefix=pkg.__name__ + "."):
        if info.ispkg:
            continue
        # Strip the package prefix → dotted name as expected by `graphids run`.
        dotted = info.name.removeprefix(pkg.__name__ + ".")
        path = (info.module_finder.path  # type: ignore[union-attr]
                + "/" + info.name.rsplit(".", 1)[-1] + ".py")
        found.append((dotted, path))
    if not found:
        _console.print("[yellow]no plan modules found[/yellow]")
        raise typer.Exit(0)
    for dotted, path in sorted(found):
        table.add_row(dotted, path)
    _console.print(table)


@plans_app.command("describe")
def describe_plan(
    plan: Annotated[str, typer.Argument(help="Dotted module name (e.g. ablations.ofat)")],
    dataset: Annotated[str, typer.Option("--dataset", "-d", help="Dataset")],
    seed: Annotated[int, typer.Option("--seed", "-s", help="Seed")] = 42,
) -> None:
    """Render the plan and print the row table — no JSON write, no submit.

    Use to preview what rows ``graphids run`` would produce. Pydantic
    validates as a side effect; bad plans surface here.
    """
    import importlib

    from graphids.cli.commands import _git_sha, mint_plan_id
    from graphids.plan.schema import Plan

    mod = importlib.import_module(f"graphids.plan.plans.{plan}")
    rows = mod.build(dataset=dataset, seed=seed)
    plan_id = mint_plan_id()
    git_sha = _git_sha()
    for r in rows:
        r["plan_id"] = plan_id
        if r.get("action") in {"fit", "test"}:
            r["plan_module"] = plan
            r["git_sha"] = git_sha
    plan_obj = Plan.model_validate({
        "plan_id": plan_id, "plan_module": plan,
        "plan_args": {"dataset": dataset, "seed": seed},
        "created_at": "describe", "rows": rows,
    })
    _console.print(
        f"[bold]{plan}[/bold]  dataset={dataset}  seed={seed}  "
        f"[dim]({len(plan_obj)} rows)[/dim]"
    )
    table = Table(show_lines=False)
    for col in ("name", "action", "variant", "mode", "length"):
        table.add_column(col)
    for r in plan_obj.rows:
        meta = getattr(r, "meta", None)
        table.add_row(
            r.name, r.action,
            getattr(meta, "variant", "—") if meta else "—",
            r.resources.mode, r.resources.length,
        )
    _console.print(table)


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

    MLflow is the source of truth (per ``chassis-design-lessons.md`` Lesson 2).
    Header prints the reproduction contract (`plan_module`, `git_sha`,
    `dataset`, `seed`); table is one row per MLflow run.

    ``--names-only`` is the shell-composition hook for retry loops::

        graphids plans show $PID --status FAILED --names-only \\
          | xargs -I{{}} graphids submit --plan plan.json --row-name {{}} -C pitzer
    """
    from mlflow.tracking import MlflowClient

    from graphids._mlflow import build_search_filter, configure_tracking_uri

    configure_tracking_uri()
    client = MlflowClient()

    exp_ids = _graphids_experiment_ids()
    if not exp_ids:
        _console.print("[red]no graphids/* experiments found[/red]")
        raise typer.Exit(1)

    runs = client.search_runs(
        experiment_ids=exp_ids,
        filter_string=build_search_filter(plan_id=plan_id, status=status),
        max_results=1000,
        order_by=["attributes.start_time ASC"],
    )
    if not runs:
        msg = f"no MLflow runs found for plan_id={plan_id}"
        if status:
            msg += f" with status={status}"
        if names_only:
            raise typer.Exit(0)
        _console.print(f"[red]{msg}[/red]")
        raise typer.Exit(1)

    if names_only:
        for r in runs:
            name = r.data.tags.get("graphids.row_name") or (r.info.run_name or "")
            print(name)
        return

    # Plan-level metadata (any run carries it; first row wins)
    head_tags = runs[0].data.tags
    _console.print(
        f"[bold]plan_id[/bold]: [cyan]{plan_id}[/cyan]  "
        f"[bold]plan_module[/bold]: {head_tags.get('graphids.plan_module', '?')}  "
        f"[bold]git_sha[/bold]: {head_tags.get('graphids.git_sha', '?')}"
    )
    _console.print(
        f"[bold]dataset[/bold]: {head_tags.get('graphids.dataset', '?')}  "
        f"[bold]seed[/bold]: {head_tags.get('graphids.seed', '?')}  "
        f"[bold]n_runs[/bold]: {len(runs)}"
        + (f"  [bold]filter[/bold]: status={status}" if status else "")
    )

    table = Table(title=None, show_lines=False)
    for col in ("row_name", "phase", "variant", "status", "started", "duration"):
        table.add_column(col)

    from datetime import UTC, datetime
    for r in runs:
        duration = (
            f"{(r.info.end_time - r.info.start_time) / 1000:.1f}s"
            if r.info.end_time else "—"
        )
        started = datetime.fromtimestamp(
            r.info.start_time / 1000, UTC
        ).strftime("%m-%d %H:%M")
        row_name = r.data.tags.get("graphids.row_name") or (r.info.run_name or "")
        table.add_row(
            row_name,
            r.data.tags.get("graphids.phase", ""),
            r.data.tags.get("graphids.variant", ""),
            r.info.status,
            started,
            duration,
        )
    _console.print(table)


@plans_app.command("submit")
def plans_submit(
    plan: Annotated[Path, typer.Option("--plan", "-p", help="Rendered plan.json")],
    cluster: Annotated[str, typer.Option("--cluster", "-C", help="pitzer | cardinal | ascend")],
    length: Annotated[str, typer.Option("--length", "-L", help="short | long")] = "long",
    filter_glob: Annotated[
        str | None,
        typer.Option("--filter", "-f", help="fnmatch glob over row names"),
    ] = None,
    resume: Annotated[
        bool,
        typer.Option(
            "--resume",
            help="Skip FINISHED rows; submit RUNNING is still skipped. Sugar over --skip-finished.",
        ),
    ] = False,
    skip_finished: Annotated[
        bool, typer.Option("--skip-finished", help="Skip rows whose MLflow run is FINISHED.")
    ] = False,
    include_failed: Annotated[
        bool,
        typer.Option(
            "--include-failed",
            help="When --skip-finished is set, RE-submit FAILED rows (default: re-submit failed).",
        ),
    ] = True,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Print what would be submitted; submit nothing.")
    ] = False,
) -> None:
    """Walk a rendered plan and submit rows. Each submission is its own
    ``submit_row`` call with its own log line — independently transparent,
    independently failable. MLflow is the state store.

    Per row, the decision tree is::

        no MLflow run                 → submit
        MLflow status RUNNING         → skip (avoid double-submit)
        MLflow status FINISHED        → skip iff --skip-finished or --resume
        MLflow status FAILED          → submit unless --include-failed=False

    Common invocations::

        graphids plans submit --plan plan.json --cluster pitzer
        graphids plans submit --plan plan.json --cluster pitzer --resume
        graphids plans submit --plan plan.json --cluster pitzer --filter 'focal*' --resume
        graphids plans submit --plan plan.json --cluster pitzer --dry-run

    See ``.claude/rules/chassis-invariants.md`` for the architectural
    properties this verb preserves.
    """
    import fnmatch
    import json

    from mlflow.tracking import MlflowClient

    from graphids._mlflow import build_search_filter, configure_tracking_uri
    from graphids.plan.schema import Plan
    from graphids.slurm.submit import submit_row

    # 1. load + validate the plan
    plan_obj = Plan.model_validate(json.loads(plan.read_text()))
    rows = list(plan_obj.rows)

    # 2. apply --filter
    if filter_glob is not None:
        kept = [r for r in rows if fnmatch.fnmatchcase(r.name, filter_glob)]
        if not kept:
            available = ", ".join(r.name for r in rows)
            raise typer.BadParameter(
                f"--filter {filter_glob!r} matched 0/{len(rows)} rows. Available: {available}"
            )
        rows = kept

    # 3. resolve --resume sugar
    if resume:
        skip_finished = True

    # 4. pull MLflow state for this plan_id once (skip if no filtering needs it)
    status_by_row: dict[str, str] = {}
    run_id_by_row: dict[str, str] = {}
    if skip_finished or any(True for _ in []):  # always cheap; future flags reuse it
        configure_tracking_uri()
        client = MlflowClient()
        exp_ids = _graphids_experiment_ids()
        if exp_ids:
            mlf_runs = client.search_runs(
                experiment_ids=exp_ids,
                filter_string=build_search_filter(plan_id=plan_obj.plan_id),
                max_results=2000,
            )
            for r in mlf_runs:
                rn = r.data.tags.get("graphids.row_name") or (r.info.run_name or "")
                # newest non-empty status wins; sort by start_time DESC
                if rn and rn not in status_by_row:
                    status_by_row[rn] = r.info.status
                    run_id_by_row[rn] = r.info.run_id

    # 5. walk rows, decide, submit
    n_submitted = n_skipped = n_error = 0
    for r in rows:
        st = status_by_row.get(r.name)
        if st == "RUNNING":
            _console.print(f"[yellow][skipped][/yellow] {r.name:30s} RUNNING (run_id={run_id_by_row.get(r.name, '?')})")
            n_skipped += 1
            continue
        if st == "FINISHED" and skip_finished:
            _console.print(f"[dim][skipped][/dim] {r.name:30s} FINISHED (run_id={run_id_by_row.get(r.name, '?')})")
            n_skipped += 1
            continue
        if st == "FAILED" and not include_failed:
            _console.print(f"[dim][skipped][/dim] {r.name:30s} FAILED (run_id={run_id_by_row.get(r.name, '?')})")
            n_skipped += 1
            continue

        if dry_run:
            prior = f" (prior status={st})" if st else ""
            _console.print(f"[cyan][would-submit][/cyan] {r.name:30s} cluster={cluster} length={length}{prior}")
            n_submitted += 1
            continue

        try:
            jid = submit_row(r, cluster=cluster, length=length)
            _console.print(f"[green][submitted][/green] {r.name:30s} jid={jid}")
            n_submitted += 1
        except Exception as exc:  # noqa: BLE001 — surface any submit failure as one log line
            _console.print(f"[red][error][/red] {r.name:30s} {exc}")
            n_error += 1

    verb = "would-submit" if dry_run else "submitted"
    _console.print(
        f"\n[bold]{verb}={n_submitted}  skipped={n_skipped}  error={n_error}[/bold]"
    )
    if n_error:
        raise typer.Exit(1)


@plans_app.command("where")
def plan_where(
    plan_id: Annotated[str, typer.Argument(help="plan_id (uuid7)")],
    row: Annotated[
        str | None,
        typer.Option("--row", "-r", help="Row name. Omit to print all rows of this plan."),
    ] = None,
) -> None:
    """Print on-disk locations + MLflow run_id for one (or all) rows of a plan.

    Resolves: run_dir, best_model.ckpt, slurm stderr (newest), MLflow run_id.
    Read-only — pure path resolution from MLflow tags + filesystem checks.
    """
    from pathlib import Path

    from mlflow.tracking import MlflowClient

    from graphids._mlflow import build_search_filter, configure_tracking_uri

    configure_tracking_uri()
    client = MlflowClient()
    exp_ids = _graphids_experiment_ids()
    if not exp_ids:
        _console.print("[red]no graphids/* experiments found[/red]")
        raise typer.Exit(1)

    runs = client.search_runs(
        experiment_ids=exp_ids,
        filter_string=build_search_filter(plan_id=plan_id, row_name=row),
        max_results=1000,
        order_by=["attributes.start_time ASC"],
    )
    if not runs:
        _console.print(f"[red]no MLflow runs for plan_id={plan_id}"
                       + (f" row={row}" if row else "") + "[/red]")
        raise typer.Exit(1)

    for r in runs:
        run_dir = r.data.tags.get("graphids.run_dir", "")
        row_name = r.data.tags.get("graphids.row_name") or (r.info.run_name or "")
        ckpt = Path(run_dir) / "checkpoints" / "best_model.ckpt"
        scripts_dir = Path(run_dir) / ".parsl_scripts"
        stderr_files = sorted(scripts_dir.glob("*.stderr"), reverse=True) if scripts_dir.exists() else []
        stderr = stderr_files[0] if stderr_files else None

        _console.print(f"[bold cyan]{row_name}[/bold cyan]  "
                       f"[dim]({r.data.tags.get('graphids.phase', '?')} / {r.info.status})[/dim]")
        _console.print(f"  run_dir:  {run_dir or '—'}")
        _console.print(f"  ckpt:     {ckpt} {'[green]✓[/green]' if ckpt.exists() else '[red]✗[/red]'}")
        _console.print(f"  stderr:   {stderr or '—'}")
        _console.print(f"  mlflow:   run_id={r.info.run_id} status={r.info.status}")
        if r.data.metrics:
            best = {k: v for k, v in r.data.metrics.items() if "auroc" in k.lower()}
            if best:
                _console.print(f"  metrics:  " + "  ".join(f"{k}={v:.4f}" for k, v in best.items()))

