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
    import os

    from graphids._otel import init_providers

    logging.getLogger("graphids").setLevel(logging.DEBUG if verbose else logging.INFO)
    init_providers(
        "graphids",
        wandb_entity=os.environ.get("WANDB_ENTITY", ""),
        wandb_project=os.environ.get("WANDB_PROJECT", "graphids"),
    )
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
    from graphids.config.topology import dataset_names

    return [n for n in dataset_names() if n.startswith(incomplete)]


def dotted_to_nested(overrides: list[tuple[str, Any]] | None) -> dict[str, Any]:
    """Expand ``[(dotted.path, value), ...]`` into a nested dict.

    Output is fed to ``render(set_pairs=...)`` which passes it as the
    ``overrides`` ``std.extVar`` consumed by every ablation preset's
    ``std.mergePatch(...)`` apex. Single entry point for ``--set`` flag
    handling — replaces the prior in-place ``apply_overrides`` mutator.
    """
    out: dict[str, Any] = {}
    for key, typed_val in overrides or []:
        parts = key.split(".")
        cur = out
        for part in parts[:-1]:
            nxt = cur.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                cur[part] = nxt
            cur = nxt
        cur[parts[-1]] = typed_val
    return out
