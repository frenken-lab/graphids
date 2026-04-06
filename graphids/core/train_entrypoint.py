"""Shared train/test entry points for both CLI and pipeline paths.

The render -> validate -> instantiate -> trainer.<method> chain lives here
so both ``graphids.cli._training`` (dev path) and
``graphids.orchestrate.entrypoint`` (pipeline path) call the same code.
"""

from __future__ import annotations

from typing import Any


def _execute(
    rendered: dict[str, Any],
    *,
    method: str = "fit",
    ckpt_path: str | None = None,
) -> None:
    """Core chain: validate -> instantiate -> trainer.<method>."""
    from graphids.config.schemas import validate_config
    from graphids.instantiate import instantiate

    validated = validate_config(rendered)
    run = instantiate(rendered, validated=validated)
    getattr(run.trainer, method)(run.model, datamodule=run.datamodule, ckpt_path=ckpt_path)


def run_training(
    *,
    config_path: str,
    tla: dict[str, Any] | None = None,
    overrides: list[str] | None = None,
    ckpt_path: str | None = None,
    method: str = "fit",
) -> None:
    """Dev-path: render -> validate -> instantiate -> trainer.<method>."""
    from graphids.cli.app import apply_overrides
    from graphids.config.jsonnet import render_config

    rendered = render_config(config_path, tla=tla or None)
    apply_overrides(rendered, overrides)
    if ckpt_path and "ckpt_path" not in rendered:
        rendered["ckpt_path"] = ckpt_path
    _execute(rendered, method=method, ckpt_path=ckpt_path)


def run_training_from_spec(spec: Any, method: str = "fit") -> None:
    """Pipeline-path: spec -> render -> validate -> instantiate -> trainer.<method>.

    ``spec`` is a ``TrainingSpec`` (imported lazily to avoid circular deps).
    """
    from graphids.config.jsonnet import render_config

    rendered = render_config(spec.jsonnet_path, tla=spec.jsonnet_tla or None)
    _execute(rendered, method=method)


def run_test_from_spec(spec: Any) -> None:
    """Pipeline-path shortcut for test."""
    run_training_from_spec(spec, method="test")
