"""Typer app + root callback. Login-node safe: no torch/model imports here.

Owns the structlog configuration since this module is the login-node
entry point that runs first; ``orchestrate`` imports
:func:`configure_logging` from here for compute-side use too.
"""

from __future__ import annotations

import functools
import logging
import os
from typing import Annotated, Any

import structlog
import typer

app = typer.Typer(
    name="graphids",
    help="GraphIDS: CAN Bus Intrusion Detection via Knowledge Distillation",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)


_SLURM_KEYS: dict[str, str] = {
    "SLURM_JOB_ID": "slurm.job_id",
    "SLURM_CLUSTER_NAME": "slurm.cluster_name",
    "SLURM_JOB_PARTITION": "slurm.partition",
    "SLURM_NODELIST": "slurm.nodelist",
    "SLURM_CPUS_PER_TASK": "slurm.cpus_per_task",
    "SLURM_GPUS_ON_NODE": "slurm.gpus_on_node",
    "CUDA_VISIBLE_DEVICES": "slurm.cuda_visible_devices",
}


def _slurm_context(_logger: Any, _method: str, event_dict: dict) -> dict:  # noqa: ARG001
    for env, key in _SLURM_KEYS.items():
        if (v := os.environ.get(env)) and key not in event_dict:
            event_dict[key] = v
    return event_dict


@functools.cache
def configure_logging() -> None:
    """structlog → JSON sync stderr with SLURM env auto-attached. Idempotent."""
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.format_exc_info,
            _slurm_context,
            structlog.processors.JSONRenderer(),
        ],
        cache_logger_on_first_use=True,
    )


@app.callback()
def _main(
    ctx: typer.Context,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging")] = False,
) -> None:
    """Shared setup for every subcommand."""
    logging.getLogger("graphids").setLevel(logging.DEBUG if verbose else logging.INFO)
    configure_logging()
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
