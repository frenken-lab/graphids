"""Typer app + root callback. Login-node safe: no torch/model imports here."""

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
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging")] = False,
) -> None:
    """Shared setup for every subcommand."""
    import logging

    from graphids.runtime import setup

    logging.getLogger("graphids").setLevel(logging.DEBUG if verbose else logging.INFO)
    setup(mode="render")
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
