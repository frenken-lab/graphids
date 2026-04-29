"""``graphids run`` and ``graphids status`` — plan-driven workflow.

A *plan* is a jsonnet file declaring ``{ nodes: [...] }``. ``graphids
run`` renders it to **JSONL** on stdout — one row per node, with a
``submit_command`` string for each. It does not submit anything; the
user (or an LLM walking the JSONL) iterates row by row and invokes
``graphids submit`` per node.

    # Render the blueprint (does NOT submit):
    graphids run    <plan.jsonnet> --dataset X --seed Y --cluster C

    # Inspect MLflow status across the plan (read-only):
    graphids status <plan.jsonnet> --dataset X --seed Y

Use ``graphids submit <preset.jsonnet>`` for atomic one-shot submissions
(no plan needed). See ``.claude/rules/single-submission-primitive.md``
for the architectural commitment behind this shape.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer

from graphids.cli.app import PlanDataset, PlanPath, PlanSeed, PlanVariants, app


def _load_plan(
    plan_path: Path,
    *,
    dataset: str,
    seed: int,
    variants: str | None = None,
):
    """Render the plan jsonnet, parse to ``tuple[Node, ...]``, optionally filter by variants."""
    from graphids.config.jsonnet import render
    from graphids.slurm.dag import filter_with_upstream, parse_plan

    try:
        rendered = render(plan_path, tla={"dataset": dataset, "seed": seed})
        nodes = parse_plan(rendered)
        if variants:
            nodes = filter_with_upstream(nodes, tuple(v.strip() for v in variants.split(",")))
        return nodes
    except (RuntimeError, ValueError) as exc:
        raise typer.BadParameter(f"failed to load plan {plan_path}: {exc}") from exc


@app.command("run", rich_help_panel="Plan", no_args_is_help=True)
def run_cli(
    plan_path: PlanPath,
    dataset: PlanDataset,
    seed: PlanSeed,
    cluster: Annotated[
        str, typer.Option("--cluster", help="Target cluster (e.g. cardinal, pitzer)")
    ] = "pitzer",
    variants: PlanVariants = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Write JSONL blueprint to file instead of stdout."),
    ] = None,
) -> None:
    """Render a plan to a JSONL blueprint of ``graphids submit`` invocations.

    Does not submit. By default writes to stdout; use --output to save as a
    persistent artifact. Each line: one JSON object per topo-sorted node with
    a ``submit_command`` string. Pipe to ``jq`` or iterate manually.
    """
    from graphids.slurm.run import render_plan_jsonl

    nodes = _load_plan(plan_path, dataset=dataset, seed=seed, variants=variants)
    jsonl = render_plan_jsonl(nodes, dataset=dataset, seed=seed, cluster=cluster)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(jsonl)
        typer.echo(f"blueprint → {output}", err=True)
    else:
        sys.stdout.write(jsonl)
        sys.stdout.flush()


@app.command("status", rich_help_panel="Plan")
def status_cli(
    plan_path: PlanPath,
    dataset: PlanDataset,
    seed: PlanSeed,
    cluster: Annotated[
        str, typer.Option("--cluster", help="Target cluster (e.g. cardinal, pitzer)")
    ] = "pitzer",
    variants: PlanVariants = None,
    fmt: Annotated[
        str,
        typer.Option("--format", "-f", help="table | json"),
    ] = "table",
) -> None:
    """Query MLflow per plan node, report status.

    PENDING/FAILED/KILLED rows include the submit_command for that cluster
    so you can copy-paste without opening the blueprint file.
    """
    from graphids.slurm.status import format_json, format_table, query_all

    nodes = _load_plan(plan_path, dataset=dataset, seed=seed, variants=variants)
    statuses = query_all(nodes, dataset=dataset, seed=seed)
    if fmt == "table":
        sys.stdout.write(format_table(statuses, dataset=dataset, seed=seed, cluster=cluster))
    elif fmt == "json":
        sys.stdout.write(format_json(statuses, dataset=dataset, seed=seed, cluster=cluster))
    else:
        raise typer.BadParameter(f"--format must be table|json (got {fmt!r})")
    sys.stdout.flush()
