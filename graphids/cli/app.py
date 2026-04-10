"""Typer CLI app for GraphIDS.

No torch / model imports at module level — safe on login nodes.
Heavy imports are deferred to inside command functions.
"""

from __future__ import annotations

import json
from typing import Annotated, Any

import typer

app = typer.Typer(
    name="graphids",
    help="GraphIDS: CAN Bus Intrusion Detection via Knowledge Distillation",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)

# ---------------------------------------------------------------------------
# Shared option types
# ---------------------------------------------------------------------------

ConfigPath = Annotated[str, typer.Option("--config", help="Path to jsonnet stage config")]
TlaList = Annotated[
    list[str] | None,
    typer.Option("--tla", help="key=value TLA for jsonnet (repeatable)"),
]
SetList = Annotated[
    list[str] | None,
    typer.Option("--set", help="dotted.path=value override on rendered dict (repeatable)"),
]
CkptPath = Annotated[
    str | None, typer.Option("--ckpt-path", help="Checkpoint path for trainer method")
]


# ---------------------------------------------------------------------------
# Shared parsers
# ---------------------------------------------------------------------------


def parse_tla(values: list[str] | None) -> dict[str, Any]:
    """Parse ``--tla key=json_value`` flags into a typed dict.

    Bare unquoted identifiers (``--tla dataset=hcrl_sa``) are tolerated as a
    shell convenience and treated as strings.
    """
    tla: dict[str, Any] = {}
    for raw in values or []:
        key, eq, val = raw.partition("=")
        if not eq:
            raise typer.BadParameter(f"--tla expects key=value, got {raw!r}")
        try:
            tla[key] = json.loads(val)
        except json.JSONDecodeError:
            tla[key] = val
    return tla


def apply_overrides(rendered: dict[str, Any], overrides: list[str] | None) -> None:
    """Apply ``--set dotted.path=value`` overrides in-place on a rendered dict."""
    for raw in overrides or []:
        key, eq, val = raw.partition("=")
        if not eq:
            raise typer.BadParameter(f"--set expects dotted.path=value, got {raw!r}")
        try:
            typed_val: Any = json.loads(val)
        except json.JSONDecodeError:
            typed_val = val

        parts = key.split(".")
        cur: Any = rendered
        for part in parts[:-1]:
            nxt = cur.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                cur[part] = nxt
            cur = nxt
        cur[parts[-1]] = typed_val
