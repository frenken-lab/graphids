"""Training commands: fit, test, validate, predict."""

from __future__ import annotations

from graphids.cli.app import CkptPath, ConfigPath, SetList, TlaList, app, parse_tla


def _run_trainer_method(
    method: str,
    config: str,
    tla: list[str] | None,
    overrides: list[str] | None,
    ckpt_path: str | None,
) -> None:
    """Render -> validate -> instantiate -> wire OTel -> trainer.<method>."""
    from pathlib import Path

    from graphids.config.jsonnet import render_config
    from graphids.config.schemas import validate_config
    from graphids.core.otel import wire_file_exporters
    from graphids.instantiate import instantiate
    from graphids.orchestrate._setup import ensure_spawn

    ensure_spawn()

    rendered = render_config(config, tla=parse_tla(tla))
    from graphids.cli.app import apply_overrides

    apply_overrides(rendered, overrides)
    if ckpt_path and "ckpt_path" not in rendered:
        rendered["ckpt_path"] = ckpt_path

    validated = validate_config(rendered)
    run = instantiate(rendered, validated=validated)

    if run.trainer.default_root_dir:
        wire_file_exporters(Path(run.trainer.default_root_dir))

    getattr(run.trainer, method)(run.model, datamodule=run.datamodule, ckpt_path=ckpt_path)


@app.command(rich_help_panel="Training")
def fit(
    config: ConfigPath,
    tla: TlaList = None,
    set_: SetList = None,
    ckpt_path: CkptPath = None,
) -> None:
    """Train a model from a jsonnet stage config."""
    _run_trainer_method("fit", config, tla, set_, ckpt_path)


@app.command(rich_help_panel="Training")
def test(
    config: ConfigPath,
    tla: TlaList = None,
    set_: SetList = None,
    ckpt_path: CkptPath = None,
) -> None:
    """Evaluate a trained model on the test set."""
    _run_trainer_method("test", config, tla, set_, ckpt_path)


@app.command(rich_help_panel="Training")
def validate(
    config: ConfigPath,
    tla: TlaList = None,
    set_: SetList = None,
    ckpt_path: CkptPath = None,
) -> None:
    """Run the validation loop."""
    _run_trainer_method("validate", config, tla, set_, ckpt_path)


@app.command(rich_help_panel="Training")
def predict(
    config: ConfigPath,
    tla: TlaList = None,
    set_: SetList = None,
    ckpt_path: CkptPath = None,
) -> None:
    """Run the prediction loop."""
    _run_trainer_method("predict", config, tla, set_, ckpt_path)
