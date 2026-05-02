"""`graphids run` — render a plan.jsonnet, validate, emit JSON array.

Pure read path: no submission, no MLflow query, no side effects beyond
writing to stdout or `--output`. Walking the rows is the caller's job:

    graphids run plan.jsonnet --dataset hcrl_sa --seed 42 -o plan.json
    jq -c '.[]' plan.json | while read row; do
        graphids submit --row "$row" --skip-if-finished
    done

(The `submit --row` half is the next step — not yet implemented.)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer

from graphids.cli.app import app


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
        typer.Option(
            "--output",
            "-o",
            help="Write JSON array to this file (default: stdout).",
        ),
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
