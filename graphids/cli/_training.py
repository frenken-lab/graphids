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
    """Shared render -> validate -> instantiate -> trainer.<method> chain."""
    import torch.multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    mp.set_sharing_strategy("file_system")

    from graphids.core.train_entrypoint import run_training

    run_training(
        config_path=config,
        tla=parse_tla(tla),
        overrides=overrides,
        ckpt_path=ckpt_path,
        method=method,
    )


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
