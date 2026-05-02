"""``graphids submit`` — submit one row as a SLURM job. Prints jid on stdout.

Pipelines walk a plan's JSON array externally and invoke this per row:

    graphids run plan.jsonnet --dataset hcrl_sa --seed 42 -o plan.json
    jq -c '.[]' plan.json | while read row; do
        graphids submit --row "$row" --cluster pitzer --length long
    done

This is the ONLY caller of ``graphids.slurm.submit_row``; per
``.claude/rules/single-submission-primitive.md`` no Python pipeline driver
exists.
"""

from __future__ import annotations

import json
import sys
from typing import Annotated

import typer

from graphids.cli.app import app


@app.command("submit", rich_help_panel="SLURM", no_args_is_help=True)
def submit_cli(
    row: Annotated[
        str,
        typer.Option(
            "--row",
            help="One row JSON object (from `graphids run`). Use '-' to read stdin.",
        ),
    ],
    cluster: Annotated[
        str,
        typer.Option("--cluster", help="Target cluster: pitzer | cardinal | ascend"),
    ],
    length: Annotated[
        str, typer.Option("--length", help="short | long (per submit_profiles.json)")
    ] = "long",
    ckpt_path: Annotated[
        str | None,
        typer.Option(
            "--ckpt-path",
            help="Resume fit / load test weights. Filesystem path passed to `graphids exec`.",
        ),
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
    from graphids.blueprint import TrainRow
    from graphids.slurm import submit_row

    raw = sys.stdin.read() if row == "-" else row
    train_row = TrainRow.model_validate(json.loads(raw))
    jid = submit_row(
        train_row,
        cluster=cluster,
        length=length,
        ckpt_path=ckpt_path,
        depends_on_afterok=depends_on_afterok,
        depends_on_afterany=depends_on_afterany,
    )
    print(jid)
    sys.stdout.flush()
