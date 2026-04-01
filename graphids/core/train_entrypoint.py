"""Programmatic training entrypoint from canonical TrainingSpec."""

from __future__ import annotations

from typing import Any

from graphids.cli import CLI_KWARGS, GraphIDSCLI
from graphids.core.contracts import TrainingContract, TrainingSpec


def run_training_from_spec(spec: TrainingSpec) -> None:
    """Execute one training run from canonical spec through GraphIDSCLI."""
    cli_args: list[str] = ["fit"]
    for cfg in spec.config_files:
        cli_args += ["--config", cfg]
    cli_args += TrainingContract.to_cli_overrides(spec)
    GraphIDSCLI(**CLI_KWARGS, args=cli_args)


def run_training_from_payload(payload: dict[str, Any]) -> None:
    run_training_from_spec(TrainingContract.from_envelope(payload))
