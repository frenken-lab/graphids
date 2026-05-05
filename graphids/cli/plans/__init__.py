"""Read-only views over rendered plans + the multi-row ``submit`` verb.

MLflow IS the trial-state store (per ``chassis-invariants.md`` #3); we
do not maintain a parallel ``jobs.jsonl`` or join sacct. Runs that died
before opening an MLflow run surface as gaps in ``plans show``.

Module layout:
- ``__init__.py`` — Typer subapp + shared MLflow bootstrap (``_mlflow_ctx``).
- ``views.py``   — read-only commands: ``available``, ``describe``, ``list``,
                   ``show``, ``where``.
- ``submit.py``  — multi-row submit verb.
"""

from __future__ import annotations

import typer
from rich.console import Console

from graphids.cli.app import app

plans_app = typer.Typer(
    name="plans",
    help="Read-only views over rendered plans (list / status) + multi-row submit.",
    no_args_is_help=True,
)
app.add_typer(plans_app, name="plans")

console = Console()


def _mlflow_ctx(*, exit_if_empty: bool = True):
    """Return ``(client, exp_ids)`` for graphids/* experiments.

    Centralizes the ``configure_tracking_uri()`` + ``MlflowClient()`` +
    ``search_experiments(filter_string="name LIKE 'graphids/%'")`` ritual
    that every plans command needs. If no experiments match and
    ``exit_if_empty`` is set, prints and exits 1 — callers that want to
    continue (e.g. ``plans submit`` falling back to "submit everything")
    pass ``exit_if_empty=False``.
    """
    from mlflow.tracking import MlflowClient

    from graphids._mlflow import configure_tracking_uri

    configure_tracking_uri()
    client = MlflowClient()
    exp_ids = [
        e.experiment_id
        for e in client.search_experiments(filter_string="name LIKE 'graphids/%'")
    ]
    if not exp_ids and exit_if_empty:
        console.print("[red]no graphids/* experiments found[/red]")
        raise typer.Exit(1)
    return client, exp_ids


# Register commands by importing for side effects.
from graphids.cli.plans import submit, views  # noqa: E402,F401
