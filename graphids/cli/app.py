"""Typer CLI app for GraphIDS.

No torch / model imports at module level — safe on login nodes.
Heavy imports are deferred to inside command functions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

app = typer.Typer(
    name="graphids",
    help="GraphIDS: CAN Bus Intrusion Detection via Knowledge Distillation",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)


# ---------------------------------------------------------------------------
# Root callback — runs once per CLI invocation before any subcommand.
# Scoped to cheap setup only (logging level + OTel providers). Spawn-method
# + CPU-thread setup imports torch and so stays inside command bodies
# (training.py:_ensure_spawn / _configure_cpu_threads) so ``<cmd> --help``
# keeps its fast path on login nodes.
# ---------------------------------------------------------------------------


@app.callback()
def _main(
    ctx: typer.Context,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Debug-level logging on the graphids logger"),
    ] = False,
) -> None:
    """GraphIDS CLI — shared setup for every subcommand."""
    import logging

    from graphids.runtime import _configure_logging

    logging.getLogger("graphids").setLevel(logging.DEBUG if verbose else logging.INFO)
    _configure_logging()
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose


# ---------------------------------------------------------------------------
# Per-element parsers (Typer option `parser=` callbacks)
# ---------------------------------------------------------------------------


def _parse_kv_pair(raw: str) -> tuple[str, Any]:
    """Parse one ``key=value`` flag into a typed ``(key, value)`` pair.

    JSON-decodes the value; bare unquoted identifiers fall through as strings.
    Shared by ``--tla`` (key is a jsonnet TLA name) and ``--set`` (key is a
    dotted path into the rendered dict).
    """
    key, eq, val = raw.partition("=")
    if not eq:
        raise typer.BadParameter(f"expected key=value, got {raw!r}")
    try:
        return key, json.loads(val)
    except json.JSONDecodeError:
        return key, val


# ---------------------------------------------------------------------------
# Shared option types
# ---------------------------------------------------------------------------

ConfigPath = Annotated[
    Path,
    typer.Option(
        "--config",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Path to jsonnet stage config",
    ),
]
# The ``parser=`` callback returns ``(key, value)`` tuples at runtime, but
# Typer's annotation inspector can't handle ``list[tuple[...]]`` — so the
# annotation lies with ``list[str] | None`` and consumers do ``dict(tla or [])``
# to recover the mapping. This keeps validation + metavar inside the Option
# decl while staying within what Typer's type system supports.
TlaList = Annotated[
    list[str] | None,
    typer.Option(
        "--tla",
        parser=_parse_kv_pair,
        metavar="KEY=JSON",
        help="key=value TLA for jsonnet (repeatable)",
    ),
]
SetList = Annotated[
    list[str] | None,
    typer.Option(
        "--set",
        parser=_parse_kv_pair,
        metavar="DOTTED.PATH=JSON",
        help="dotted.path=value override on rendered dict (repeatable)",
    ),
]
CkptPath = Annotated[
    str | None, typer.Option("--ckpt-path", help="Checkpoint path for trainer method")
]

# Plan-workflow shared options (used by ``run`` + ``status`` so the two
# sibling commands present an identical surface — change here, both update).
PlanPath = Annotated[
    Path,
    typer.Argument(
        help="Plan jsonnet (e.g. configs/plans/ofat.jsonnet).",
        exists=True,
        dir_okay=False,
        resolve_path=True,
    ),
]
PlanDataset = Annotated[str, typer.Option("--dataset", help="Dataset TLA")]
PlanSeed = Annotated[int, typer.Option("--seed", help="Seed TLA")]
PlanVariants = Annotated[
    str | None,
    typer.Option(
        "--variants",
        metavar="A,B,...",
        help="Subset of node names (transitive upstream deps auto-included).",
    ),
]


# ---------------------------------------------------------------------------
# Shell completion helpers
# ---------------------------------------------------------------------------
#
# Each ``_complete_*`` takes an ``incomplete`` prefix (typer passes whatever the
# user has typed so far after ``<TAB>``) and returns the matching values. Values
# come from the authoritative source (dataset catalog, axes.json frozenset) — no
# hardcoded lists that can drift. Each helper defers its imports so ``--help``
# stays fast.


def _complete_dataset(incomplete: str) -> list[str]:
    from graphids.config.catalog import dataset_names

    return [n for n in dataset_names() if n.startswith(incomplete)]
