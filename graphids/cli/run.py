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

from graphids.cli.app import app


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
    plan_path: Annotated[
        Path,
        typer.Argument(
            help="Plan jsonnet (e.g. configs/plans/ofat.jsonnet).",
            exists=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ],
    dataset: Annotated[str, typer.Option("--dataset", help="Dataset TLA")],
    seed: Annotated[int, typer.Option("--seed", help="Seed TLA")],
    cluster: Annotated[
        str, typer.Option("--cluster", help="Target cluster (e.g. cardinal, pitzer)")
    ] = "pitzer",
    variants: Annotated[
        str | None,
        typer.Option(
            "--variants",
            metavar="A,B,...",
            help="Subset of node names (transitive upstream deps auto-included).",
        ),
    ] = None,
) -> None:
    """Render a plan to a JSONL blueprint of ``graphids submit`` invocations.

    Does not submit. Stdout. Each line: one JSON object per topo-sorted node
    with a ``submit_command`` string. Pipe to ``jq`` or iterate manually.
    """
    from graphids.slurm.run import render_plan_jsonl

    nodes = _load_plan(plan_path, dataset=dataset, seed=seed, variants=variants)
    sys.stdout.write(render_plan_jsonl(nodes, dataset=dataset, seed=seed, cluster=cluster))
    sys.stdout.flush()


@app.command("status", rich_help_panel="Plan")
def status_cli(
    plan_path: Annotated[
        Path,
        typer.Argument(
            help="Plan jsonnet.",
            exists=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ],
    dataset: Annotated[str, typer.Option("--dataset", help="Dataset TLA")],
    seed: Annotated[int, typer.Option("--seed", help="Seed TLA")],
    variants: Annotated[
        str | None,
        typer.Option(
            "--variants",
            metavar="A,B,...",
            help="Subset of node names (transitive upstream deps auto-included).",
        ),
    ] = None,
    fmt: Annotated[
        str,
        typer.Option("--format", "-f", help="table | json"),
    ] = "table",
) -> None:
    """Query MLflow per plan node, report status."""
    from graphids.slurm.status import format_json, format_table, query_all

    nodes = _load_plan(plan_path, dataset=dataset, seed=seed, variants=variants)
    statuses = query_all(nodes, dataset=dataset, seed=seed)
    if fmt == "table":
        sys.stdout.write(format_table(statuses, dataset=dataset, seed=seed))
    elif fmt == "json":
        sys.stdout.write(format_json(statuses, dataset=dataset, seed=seed))
    else:
        raise typer.BadParameter(f"--format must be table|json (got {fmt!r})")
    sys.stdout.flush()
