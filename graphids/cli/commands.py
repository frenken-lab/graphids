"""The four-step chassis: ``run`` (render plan) → ``exec`` (run row) → ``submit`` (SLURM).

Each command is a thin Typer wrapper over one library call. Pipelines walk
the rendered JSON array externally; per
``.claude/rules/single-submission-primitive.md`` no Python pipeline driver
exists.

Usage:
    graphids run plan.jsonnet --dataset hcrl_sa --seed 42 -o plan.json
    jq -c '.[]' plan.json | while read row; do
        graphids submit --row "$row" --cluster pitzer
    done
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated

import typer

from graphids.cli.app import app


def _load_row(raw: str):
    """Parse a single-row JSON string (or ``-`` for stdin) into a validated ``Row``.

    Wraps in a singleton list so ``BlueprintArray``'s discriminated union picks
    the row subclass by ``action``.
    """
    from graphids.blueprint import BlueprintArray

    text = sys.stdin.read() if raw == "-" else raw
    return BlueprintArray.model_validate([json.loads(text)])[0]


@app.command("run", rich_help_panel="Plans", no_args_is_help=True)
def run_cli(
    plan: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            resolve_path=True,
            help="Plan jsonnet path (e.g. configs/plans/ofat.jsonnet)",
        ),
    ],
    dataset: Annotated[str, typer.Option("--dataset", help="Dataset TLA (e.g. hcrl_sa)")],
    seed: Annotated[int, typer.Option("--seed", help="Seed TLA")],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Write JSON array to this file (default: stdout)."),
    ] = None,
) -> None:
    """Render a plan, validate, write the row array as JSON."""
    from graphids.blueprint import BlueprintArray
    from graphids.config.jsonnet import render

    rendered = render(plan, tla={"dataset": dataset, "seed": seed})
    blueprint = BlueprintArray.model_validate(rendered)
    out = blueprint.model_dump_json(indent=2)
    if output is None:
        sys.stdout.write(out + "\n")
    else:
        output.write_text(out + "\n")
        print(f"wrote {len(blueprint)} rows to {output}", file=sys.stderr)


@app.command("exec", rich_help_panel="Execution", no_args_is_help=True)
def exec_cli(
    row: Annotated[
        str,
        typer.Option("--row", help="One row JSON object (from `graphids run`). '-' for stdin."),
    ],
    ckpt_path: Annotated[
        str | None,
        typer.Option("--ckpt-path", help="Resume fit / load test weights. Filesystem path."),
    ] = None,
) -> None:
    """Execute one row in-process. Dispatches on ``row.action`` (fit | test | extract | cmd)."""
    from graphids.orchestrate import run_row

    run_row(_load_row(row), ckpt_path=ckpt_path)


@app.command("submit", rich_help_panel="SLURM", no_args_is_help=True)
def submit_cli(
    row: Annotated[
        str,
        typer.Option("--row", help="One row JSON object (from `graphids run`). '-' for stdin."),
    ],
    cluster: Annotated[
        str, typer.Option("--cluster", help="Target cluster: pitzer | cardinal | ascend")
    ],
    length: Annotated[
        str, typer.Option("--length", help="short | long (per submit_profiles.json)")
    ] = "long",
    ckpt_path: Annotated[
        str | None,
        typer.Option("--ckpt-path", help="Filesystem path passed through to `graphids exec`."),
    ] = None,
    depends_on_afterok: Annotated[
        str | None,
        typer.Option(
            "--depends-on-afterok",
            help="Add SBATCH --dependency=afterok:<jid> (data dep on a SLURM job).",
        ),
    ] = None,
    depends_on_afterany: Annotated[
        str | None,
        typer.Option(
            "--depends-on-afterany",
            help="Add SBATCH --dependency=afterany:<jid> (preempt-resume chain).",
        ),
    ] = None,
) -> None:
    """Submit one row to SLURM via Parsl SlurmProvider. Prints jid on stdout."""
    from graphids.slurm import submit_row

    jid = submit_row(
        _load_row(row),
        cluster=cluster,
        length=length,
        ckpt_path=ckpt_path,
        depends_on_afterok=depends_on_afterok,
        depends_on_afterany=depends_on_afterany,
    )
    print(jid)
    sys.stdout.flush()
