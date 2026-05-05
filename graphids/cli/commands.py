"""Four-step chassis: ``run`` → ``exec`` → ``submit`` (+ stdin / stdout pipe).

Each command is a thin Typer wrapper over one library call. Pipelines walk
the rendered JSON externally (``jq -c '.rows[]' plan.json | while read row``) —
renders are pure JSON per ``chassis-invariants.md``; the multi-row
verb ``plans submit`` exists in ``cli/plans.py``.

Usage:
    graphids run supervised --dataset hcrl_sa --seed 42 -o plan.json
    jq -c '.rows[]' plan.json | while read row; do
        graphids submit --row "$row" --cluster pitzer
    done
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer

from graphids.cli.app import app


def _git_sha() -> str:
    """Short git SHA of the working tree, or ``"unknown"`` outside a repo / on
    a corrupt checkout. Stamped onto fit/test rows at render time so MLflow
    runs carry the reproduction-contract SHA (`git checkout <sha> && graphids
    run <module> --filter <name>` regenerates the exact rendered config).
    """
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            capture_output=True, text=True, check=True, cwd=Path(__file__).resolve().parent,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def mint_plan_id() -> str:
    """RFC 9562 UUIDv7 — 48-bit ms timestamp + 4-bit version + 74 bits random.

    Lex-sortable == temporally-sortable, so ``ls plan_*.json | sort`` and
    MLflow tag ranges over ``graphids.plan_id`` are temporally ordered.
    """
    ts_ms = int(time.time() * 1000)
    rand = int.from_bytes(os.urandom(10), "big")
    rand_a = (rand >> 64) & 0xFFF
    rand_b = rand & ((1 << 62) - 1)
    val = (ts_ms << 80) | (0x7 << 76) | (rand_a << 64) | (0b10 << 62) | rand_b
    h = f"{val:032x}"
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"

# ── Typer option aliases (collapse the Annotated[T, typer.Option(...)] ladder) ──
_RowOpt = Annotated[
    str | None,
    typer.Option(
        "--row", "-r",
        help="One row JSON. '-' for stdin. Mutually exclusive with --plan/--row-name.",
    ),
]
_PlanFileOpt = Annotated[
    Path | None,
    typer.Option(
        "--plan", "-p",
        help="Path to a rendered plan.json. Use with --row-name to pick one row.",
    ),
]
_RowNameOpt = Annotated[
    str | None,
    typer.Option(
        "--row-name", "-n",
        help="Row name within --plan to operate on (single row, exact match).",
    ),
]
_CkptOpt = Annotated[
    str | None, typer.Option("--ckpt-path", help="Resume fit / load test weights.")
]
_ClusterOpt = Annotated[
    str, typer.Option("--cluster", "-C", help="Target cluster: pitzer | cardinal | ascend")
]
_LengthOpt = Annotated[
    str, typer.Option("--length", "-L", help="short | long (per submit_profiles.json)")
]
_DepOk = Annotated[
    str | None, typer.Option("--depends-on-afterok", help="SBATCH afterok dependency.")
]
_DepAny = Annotated[
    str | None, typer.Option("--depends-on-afterany", help="SBATCH afterany dependency.")
]


def _resolve_row(row: str | None, plan_file: Path | None, row_name: str | None):
    """Pick one row from either inline JSON (``--row``) or a plan file
    (``--plan`` + ``--row-name``). Validates as discriminated-union ``Row``.

    Single-row contract — exactly one path; submit/exec still operates on
    one row, the rule is not violated.
    """
    from pydantic import TypeAdapter

    from graphids.plan.schema import Row

    inline = row is not None
    by_name = plan_file is not None or row_name is not None

    if inline and by_name:
        raise typer.BadParameter("--row is mutually exclusive with --plan/--row-name")
    if not inline and not (plan_file is not None and row_name is not None):
        raise typer.BadParameter(
            "specify either --row '<json>' or --plan FILE --row-name NAME"
        )

    if inline:
        text = sys.stdin.read() if row == "-" else row
        return TypeAdapter(Row).validate_python(json.loads(text))

    plan_obj = json.loads(plan_file.read_text())  # type: ignore[union-attr]
    matches = [r for r in plan_obj["rows"] if r.get("name") == row_name]
    if not matches:
        names = ", ".join(r["name"] for r in plan_obj["rows"])
        raise typer.BadParameter(
            f"--row-name {row_name!r} not in {plan_file}. Available: {names}"
        )
    if len(matches) > 1:
        raise typer.BadParameter(f"--row-name {row_name!r} matched {len(matches)} rows (must be unique)")
    return TypeAdapter(Row).validate_python(matches[0])


@app.command("run", rich_help_panel="Plans", no_args_is_help=True)
def run_cli(
    plan: Annotated[
        str,
        typer.Argument(
            help="Dotted module name under graphids.plan.plans "
            "(e.g. 'supervised', 'ofat', 'smoke.gat_taunorm'). "
            "Calls module.build(dataset=..., seed=...).",
        ),
    ],
    dataset: Annotated[str, typer.Option("--dataset", help="Dataset (e.g. hcrl_sa)")],
    seed: Annotated[int, typer.Option("--seed", help="Random seed")],
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Write JSON to file (default: stdout).")
    ] = None,
    filter_glob: Annotated[
        str | None,
        typer.Option(
            "--filter",
            help="Render only rows whose name matches this fnmatch glob "
            "(e.g. 'gat_focal*'). Single-row retry = single-name filter; "
            "user/LLM iterates (or use `graphids plans submit` for bulk).",
        ),
    ] = None,
) -> None:
    """Render a plan, validate, write the ``Plan`` JSON object.

    Imports ``graphids.plan.plans.<plan>`` and calls
    ``build(dataset=..., seed=...)``. Mints a fresh ``plan_id`` (uuid7)
    and threads it onto every row. Output JSON shape:
    ``{plan_id, plan_module, plan_args, created_at, rows: [...]}``.

    Render bugs surface here, not at SLURM submission.
    """
    import fnmatch
    import importlib

    from graphids.plan.schema import Plan

    mod = importlib.import_module(f"graphids.plan.plans.{plan}")
    rows = mod.build(dataset=dataset, seed=seed)
    if filter_glob is not None:
        kept = [r for r in rows if fnmatch.fnmatchcase(r["name"], filter_glob)]
        if not kept:
            names = ", ".join(r["name"] for r in rows)
            raise typer.BadParameter(
                f"--filter {filter_glob!r} matched 0/{len(rows)} rows. Available: {names}"
            )
        rows = kept
    plan_id = mint_plan_id()
    git_sha = _git_sha()
    for r in rows:
        r["plan_id"] = plan_id
        if r.get("action") in {"fit", "test"}:
            r["plan_module"] = plan
            r["git_sha"] = git_sha
    plan_obj = Plan.model_validate({
        "plan_id": plan_id,
        "plan_module": plan,
        "plan_args": {"dataset": dataset, "seed": seed},
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "rows": rows,
    })
    out = plan_obj.model_dump_json(indent=2) + "\n"
    if output is None:
        sys.stdout.write(out)
    else:
        output.write_text(out)
        print(
            f"wrote {len(plan_obj)} rows (plan_id={plan_id}) to {output}",
            file=sys.stderr,
        )


@app.command("exec", rich_help_panel="Execution", no_args_is_help=True)
def exec_cli(
    row: _RowOpt = None,
    plan: _PlanFileOpt = None,
    row_name: _RowNameOpt = None,
    ckpt_path: _CkptOpt = None,
) -> None:
    """Execute one row in-process. Dispatches on ``row.action`` (fit | test | extract | analyze | cache).

    Specify the row via either ``--row '<json>'`` (or ``--row -`` for stdin)
    or ``--plan plan.json --row-name NAME``. Single-row only.
    """
    from graphids.orchestrate import run_row

    run_row(_resolve_row(row, plan, row_name), ckpt_path=ckpt_path)


@app.command("cache", rich_help_panel="Execution", no_args_is_help=True)
def cache_cli(
    dataset: Annotated[str, typer.Option("--dataset", help="Dataset name (catalog key, e.g. hcrl_sa)")],
    vocab_scope: Annotated[
        str, typer.Option("--vocab-scope", help="train | all (cache partition)")
    ] = "train",
) -> None:
    """Build the dataset cache for ``(dataset, vocab_scope)``. Idempotent.

    Same body as a ``cache``-action row, runnable directly on a login node
    (no SLURM ingest) for small datasets — or via ``graphids submit`` for
    HCRL-scale builds.
    """
    from graphids.plan.schema import CacheRow, Resources
    from graphids.orchestrate import cache

    cache(
        CacheRow(
            name=f"cache_{dataset}_{vocab_scope}",
            action="cache",
            plan_id=mint_plan_id(),
            dataset=dataset,
            vocab_scope=vocab_scope,  # type: ignore[arg-type]
            resources=Resources(mode="cpu", length="short"),
        )
    )


@app.command("submit", rich_help_panel="SLURM", no_args_is_help=True)
def submit_cli(
    cluster: _ClusterOpt,
    row: _RowOpt = None,
    plan: _PlanFileOpt = None,
    row_name: _RowNameOpt = None,
    length: _LengthOpt = "long",
    ckpt_path: _CkptOpt = None,
    depends_on_afterok: _DepOk = None,
    depends_on_afterany: _DepAny = None,
) -> None:
    """Submit one row to SLURM via Parsl SlurmProvider. Prints jid on stdout.

    Specify the row via either ``--row '<json>'`` or
    ``--plan plan.json --row-name NAME``. For bulk submission, see
    ``graphids plans submit --plan FILE --cluster X``.
    """
    from graphids.slurm.submit import submit_row

    jid = submit_row(
        _resolve_row(row, plan, row_name),
        cluster=cluster,
        length=length,
        ckpt_path=ckpt_path,
        depends_on_afterok=depends_on_afterok,
        depends_on_afterany=depends_on_afterany,
    )
    print(jid)
    sys.stdout.flush()
