"""`graphids exec` — execute one row in-process. Login-node smoke / non-SLURM path.

The compute-node submitit body imports :func:`graphids.orchestrate.run_row`
directly (it's pickled into the sbatch job by submitit, no CLI involved).
This command exists for direct testing, debugging, and one-off runs:

    graphids run plan.jsonnet --dataset hcrl_sa --seed 42 \\
        | jq -c '.[0]' \\
        | xargs -I{} graphids exec --row {}

Or via stdin:

    echo '<row-json>' | graphids exec --row -
"""

from __future__ import annotations

import json
import sys
from typing import Annotated

import typer

from graphids.cli.app import app


@app.command("exec", rich_help_panel="Execution", no_args_is_help=True)
def exec_cli(
    row: Annotated[
        str,
        typer.Option(
            "--row",
            help="One row JSON object (from `graphids run`). Use '-' for stdin.",
        ),
    ],
    ckpt_path: Annotated[
        str | None,
        typer.Option(
            "--ckpt-path",
            help="Resume fit weights / load test weights. Filesystem path.",
        ),
    ] = None,
) -> None:
    """Execute one row in-process. Dispatches on `row.action` (fit | test | extract | cmd)."""
    from graphids.blueprint import BlueprintArray
    from graphids.orchestrate import run_row

    raw = sys.stdin.read() if row == "-" else row
    # Wrap in a singleton list so the discriminated union picks the right
    # row type by ``action``. BlueprintArray is the canonical Row validator.
    row_obj = BlueprintArray.model_validate([json.loads(raw)])[0]
    run_row(row_obj, ckpt_path=ckpt_path)
