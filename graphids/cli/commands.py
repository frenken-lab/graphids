"""Four-step chassis: ``run`` → ``exec`` → ``submit`` (+ stdin / stdout pipe).

Each command is a thin Typer wrapper over one library call. Pipelines walk
the rendered JSON externally (``jq -c '.[]' plan.json | while read row``) —
no Python pipeline driver, per ``single-submission-primitive.md``.

Usage:
    graphids run supervised --dataset hcrl_sa --seed 42 -o plan.json
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

# ── Typer option aliases (collapse the Annotated[T, typer.Option(...)] ladder) ──
_RowOpt = Annotated[str, typer.Option("--row", help="One row JSON. '-' for stdin.")]
_CkptOpt = Annotated[
    str | None, typer.Option("--ckpt-path", help="Resume fit / load test weights.")
]
_ClusterOpt = Annotated[
    str, typer.Option("--cluster", help="Target cluster: pitzer | cardinal | ascend")
]
_LengthOpt = Annotated[
    str, typer.Option("--length", help="short | long (per submit_profiles.json)")
]
_DepOk = Annotated[
    str | None, typer.Option("--depends-on-afterok", help="SBATCH afterok dependency.")
]
_DepAny = Annotated[
    str | None, typer.Option("--depends-on-afterany", help="SBATCH afterany dependency.")
]


def _load_row(raw: str):
    """Parse single-row JSON (or ``-`` stdin) → validated ``Row`` via discriminated union."""
    from graphids.graphids.config.blueprint import BlueprintArray

    text = sys.stdin.read() if raw == "-" else raw
    return BlueprintArray.model_validate([json.loads(text)])[0]


@app.command("run", rich_help_panel="Plans", no_args_is_help=True)
def run_cli(
    plan: Annotated[
        str,
        typer.Argument(
            help="Dotted module name under graphids.configs.plans "
            "(e.g. 'supervised', 'ofat', 'ops.gat_taunorm_smoke'). "
            "Calls module.build(dataset=..., seed=...).",
        ),
    ],
    dataset: Annotated[str, typer.Option("--dataset", help="Dataset (e.g. hcrl_sa)")],
    seed: Annotated[int, typer.Option("--seed", help="Random seed")],
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Write JSON to file (default: stdout).")
    ] = None,
) -> None:
    """Render a plan, validate, write the row array as JSON.

    Imports ``graphids.configs.plans.<plan>`` and calls
    ``build(dataset=..., seed=...)``. The blueprint is validated with
    Pydantic before serialization; render bugs surface here, not at
    SLURM submission.
    """
    import importlib

    from graphids.graphids.config.blueprint import BlueprintArray

    mod = importlib.import_module(f"graphids.configs.plans.{plan}")
    blueprint = BlueprintArray.model_validate(mod.build(dataset=dataset, seed=seed))
    out = blueprint.model_dump_json(indent=2) + "\n"
    if output is None:
        sys.stdout.write(out)
    else:
        output.write_text(out)
        print(f"wrote {len(blueprint)} rows to {output}", file=sys.stderr)


@app.command("exec", rich_help_panel="Execution", no_args_is_help=True)
def exec_cli(row: _RowOpt, ckpt_path: _CkptOpt = None) -> None:
    """Execute one row in-process. Dispatches on ``row.action`` (fit | test | extract | analyze)."""
    from graphids.orchestrate import run_row

    run_row(_load_row(row), ckpt_path=ckpt_path)


@app.command("submit", rich_help_panel="SLURM", no_args_is_help=True)
def submit_cli(
    row: _RowOpt,
    cluster: _ClusterOpt,
    length: _LengthOpt = "long",
    ckpt_path: _CkptOpt = None,
    depends_on_afterok: _DepOk = None,
    depends_on_afterany: _DepAny = None,
) -> None:
    """Submit one row to SLURM via Parsl SlurmProvider. Prints jid on stdout."""
    from graphids.slurm.submit import submit_row

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
