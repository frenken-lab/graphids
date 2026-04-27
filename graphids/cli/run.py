"""``graphids run`` and ``graphids status`` — plan-driven workflow.

A *plan* is a jsonnet file declaring ``{ nodes: [...] }``. Each entry
becomes one ``graphids submit`` invocation in the rendered bash artifact.

    # Render the artifact (does NOT submit):
    graphids run    <plan.jsonnet> --dataset X --seed Y --cluster C > runme.sh
    bash runme.sh   # actually submits

    # Inspect MLflow status across the plan (read-only):
    graphids status <plan.jsonnet> --dataset X --seed Y

Use ``graphids submit <preset.jsonnet>`` for atomic one-shot submissions
(no plan needed).
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
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Emit without --skip-if-finished on each line. Default: skip FINISHED nodes.",
        ),
    ] = False,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Write the bash artifact to FILE (default: stdout).",
        ),
    ] = None,
) -> None:
    """Render a plan to an executable bash artifact composed of `graphids submit` calls.

    Does not submit. Pipe to ``bash`` or save with ``--output FILE``.
    """
    from graphids.slurm.run import render_plan_script

    nodes = _load_plan(plan_path, dataset=dataset, seed=seed, variants=variants)
    invocation = "graphids run " + " ".join(sys.argv[2:])  # everything after `graphids run`
    script = render_plan_script(
        nodes,
        dataset=dataset,
        seed=seed,
        cluster=cluster,
        skip_finished=not force,
        invocation=invocation,
    )
    if output:
        output.write_text(script)
        # Make it executable for the user's convenience.
        output.chmod(output.stat().st_mode | 0o111)
    else:
        sys.stdout.write(script)
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
