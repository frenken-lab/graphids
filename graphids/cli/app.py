"""Typer app + root callback for GraphIDS.

No torch / model imports at module level — safe on login nodes. Heavy
imports are deferred to inside command bodies.
"""

from __future__ import annotations

from typing import Annotated

import typer

app = typer.Typer(
    name="graphids",
    help="GraphIDS: CAN Bus Intrusion Detection via Knowledge Distillation",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)


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

    from graphids.orchestrate import setup

    logging.getLogger("graphids").setLevel(logging.DEBUG if verbose else logging.INFO)
    setup(mode="render")
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
