"""Training commands: fit, test.

Both share the same prelude (render → validate → build) with the
pipeline driver (``orchestrate/run.py``), then dispatch through
``orchestrate.stage.train`` / ``orchestrate.stage.evaluate`` so the
CLI and the pipeline loop produce identical markers, OTel wiring,
and GPU-reset semantics.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from graphids.cli.app import CkptPath, ConfigPath, SetList, TlaList, app
from graphids.orchestrate.config import ResolvedConfig


def _prepare(
    config: Path,
    tla: list[Any] | None,
    overrides: list[Any] | None,
) -> tuple[ResolvedConfig, object]:
    """Shared prelude: render → apply_overrides → resolve → wire OTel → build.

    Returns ``(resolved, artifacts)``. Heavy imports live inside the
    function so the app stays login-node-safe.
    """
    from graphids._cpu import configure_cpu_threads
    from graphids._otel import wire_file_exporters
    from graphids._spawn import ensure_spawn
    from graphids.cli.app import apply_overrides
    from graphids.config.jsonnet import render
    from graphids.orchestrate.stage import build

    ensure_spawn()
    configure_cpu_threads()

    rendered = render(config, tla=dict(tla or []) or None)
    apply_overrides(rendered, overrides)
    resolved = ResolvedConfig.from_rendered(rendered, stage_name=config.stem)
    if resolved.run_dir is not None:
        wire_file_exporters(resolved.run_dir)
    artifacts = build(resolved)
    return resolved, artifacts


@app.command(rich_help_panel="Training")
def fit(
    config: ConfigPath,
    tla: TlaList = None,
    set_: SetList = None,
    ckpt_path: CkptPath = None,
) -> None:
    """Train a model from a jsonnet stage config."""
    from graphids.orchestrate.stage import train

    resolved, artifacts = _prepare(config, tla, set_)
    train(artifacts, resolved, resume_from=ckpt_path)


@app.command(rich_help_panel="Training")
def test(
    config: ConfigPath,
    tla: TlaList = None,
    set_: SetList = None,
    ckpt_path: CkptPath = None,
) -> None:
    """Evaluate a trained model on the test set."""
    from graphids.orchestrate.stage import evaluate

    resolved, artifacts = _prepare(config, tla, set_)
    # When --ckpt-path is explicit, it overrides the resolved ckpt_file.
    if ckpt_path:
        resolved = replace(resolved, ckpt_file=Path(ckpt_path))
    evaluate(artifacts, resolved)
