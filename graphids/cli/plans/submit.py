"""``graphids plans submit`` — multi-row submit. Each row is independently
logged and failable, per ``chassis-invariants.md``.

Decision tree per row:

    no MLflow run                 → submit
    MLflow status RUNNING         → skip (avoid double-submit)
    MLflow status FINISHED        → skip iff --skip-finished or --resume
    MLflow status FAILED          → submit unless --include-failed=False
"""

from __future__ import annotations

import fnmatch
import json
from pathlib import Path
from typing import Annotated

import typer

from graphids.cli.plans import _mlflow_ctx, console, plans_app


def _mlflow_status_by_row(plan_id: str) -> tuple[dict[str, str], dict[str, str]]:
    """Return ``(status_by_row, run_id_by_row)`` for one plan_id.

    Newest non-empty status per row name wins (search returns DESC by default
    via MLflow's order_by tiebreak; we explicitly preserve first-seen).
    """
    from graphids._mlflow import build_search_filter

    client, exp_ids = _mlflow_ctx(exit_if_empty=False)
    if not exp_ids:
        return {}, {}
    runs = client.search_runs(
        experiment_ids=exp_ids,
        filter_string=build_search_filter(plan_id=plan_id),
        max_results=2000,
    )
    status_by_row: dict[str, str] = {}
    run_id_by_row: dict[str, str] = {}
    for r in runs:
        rn = r.data.tags.get("graphids.row_name") or (r.info.run_name or "")
        if rn and rn not in status_by_row:
            status_by_row[rn] = r.info.status
            run_id_by_row[rn] = r.info.run_id
    return status_by_row, run_id_by_row


@plans_app.command("submit")
def plans_submit(
    plan: Annotated[Path, typer.Option("--plan", "-p", help="Rendered plan.json")],
    cluster: Annotated[str, typer.Option("--cluster", "-C", help="pitzer | cardinal | ascend")],
    length: Annotated[
        str | None,
        typer.Option("--length", "-L", help="short | long (default: row's own resources.length)"),
    ] = None,
    filter_glob: Annotated[
        str | None,
        typer.Option("--filter", "-f", help="fnmatch glob over row names"),
    ] = None,
    resume: Annotated[
        bool,
        typer.Option("--resume", help="Skip FINISHED rows. Sugar over --skip-finished."),
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
    ``submit_row`` call with its own log line.

    Common invocations::

        graphids plans submit --plan plan.json --cluster pitzer
        graphids plans submit --plan plan.json --cluster pitzer --resume
        graphids plans submit --plan plan.json --cluster pitzer --filter 'focal*' --resume
        graphids plans submit --plan plan.json --cluster pitzer --dry-run
    """
    from graphids.plan.schema import Plan
    from graphids.slurm.submit import submit_row

    plan_obj = Plan.model_validate(json.loads(plan.read_text()))
    rows = list(plan_obj.rows)

    if filter_glob is not None:
        kept = [r for r in rows if fnmatch.fnmatchcase(r.name, filter_glob)]
        if not kept:
            available = ", ".join(r.name for r in rows)
            raise typer.BadParameter(
                f"--filter {filter_glob!r} matched 0/{len(rows)} rows. Available: {available}"
            )
        rows = kept

    if resume:
        skip_finished = True

    status_by_row, run_id_by_row = (
        _mlflow_status_by_row(plan_obj.plan_id) if skip_finished else ({}, {})
    )

    n_submitted = n_skipped = n_error = 0
    fit_jid_by_name: dict[str, str] = {}  # name → jid for fit rows submitted this run
    extract_jid: str | None = None  # jid of most recent extract row this invocation
    for r in rows:
        st = status_by_row.get(r.name)
        rid = run_id_by_row.get(r.name, "?")
        if st == "RUNNING":
            console.print(f"[yellow][skipped][/yellow] {r.name:30s} RUNNING (run_id={rid})")
            n_skipped += 1
            continue
        if st == "FINISHED" and skip_finished:
            console.print(f"[dim][skipped][/dim] {r.name:30s} FINISHED (run_id={rid})")
            n_skipped += 1
            continue
        if st == "FAILED" and not include_failed:
            console.print(f"[dim][skipped][/dim] {r.name:30s} FAILED (run_id={rid})")
            n_skipped += 1
            continue

        action = getattr(r, "action", None)

        # Fit→test afterok chain (narrow, name-convention only — not a DAG runner).
        # Per `compose._emit`, every test row is named `{fit_name}-test`. If the
        # matching fit row was submitted in THIS invocation, chain on its jid so
        # the test waits for ckpt + LM to exist. Test rows with no fit pair, or
        # rows whose fit was --skip-finished, submit unchained.
        #
        # Extract→fit afterok chain: if an extract row was submitted this invocation,
        # gate all subsequent fit rows on it so fusion states exist before training starts.
        afterok = None
        if action == "test" and r.name.endswith("-test") and r.name[:-5] in fit_jid_by_name:
            afterok = fit_jid_by_name[r.name[:-5]]
        elif action == "fit" and extract_jid is not None:
            afterok = extract_jid

        if dry_run:
            prior = f" (prior status={st})" if st else ""
            chain = f" afterok={afterok}" if afterok else ""
            eff_length = length or r.resources.length
            console.print(
                f"[cyan][would-submit][/cyan] {r.name:30s} cluster={cluster} length={eff_length}{prior}{chain}"
            )
            n_submitted += 1
            if action == "fit":
                fit_jid_by_name[r.name] = "<dry-run>"
            elif action == "extract":
                extract_jid = "<dry-run>"
            continue

        try:
            jid = submit_row(
                r, cluster=cluster, length=length or r.resources.length, depends_on_afterok=afterok
            )
            chain = f" (afterok={afterok})" if afterok else ""
            console.print(f"[green][submitted][/green] {r.name:30s} jid={jid}{chain}")
            n_submitted += 1
            if action == "fit":
                fit_jid_by_name[r.name] = jid
            elif action == "extract":
                extract_jid = jid
        except Exception as exc:  # noqa: BLE001 — surface any submit failure as one log line
            console.print(f"[red][error][/red] {r.name:30s} {exc}")
            n_error += 1

    verb = "would-submit" if dry_run else "submitted"
    console.print(f"\n[bold]{verb}={n_submitted}  skipped={n_skipped}  error={n_error}[/bold]")
    if n_error:
        raise typer.Exit(1)
