"""Dict→objects→Trainer.fit bridge. importlib-instantiate from ``rendered_config``,
open/close the MLflow run, dispatch on ``row.action``. Loop lives in
:mod:`graphids.core.trainer`."""

from __future__ import annotations

import importlib
import time
from typing import Any

from mlflow.entities import Metric
from mlflow.tracking import MlflowClient

from graphids import runtime
from graphids._mlflow import (
    MLflowTrainingCallback,
    _scalar_metrics,
    end_training_run,
    start_training_run,
)
from graphids.blueprint import ExtractRow, Row, TrainRow
import torch_geometric

from graphids.core.trainer import Trainer, TrainerConfig


def _instantiate(spec: dict[str, Any]) -> Any:
    """Build ``{class_path, init_args}``; recurses on nested class_path blocks
    in init_args (e.g. GAT's ``loss_fn``)."""
    rec = lambda v: _instantiate(v) if isinstance(v, dict) and "class_path" in v else v  # noqa: E731
    ia = {k: rec(v) for k, v in spec.get("init_args", {}).items()}
    mod, _, attr = spec["class_path"].rpartition(".")
    return getattr(importlib.import_module(mod), attr)(**ia)


def _build(row: TrainRow, run_id: str) -> tuple[Any, Any, list, TrainerConfig]:
    """Instantiate model + datamodule + callbacks + TrainerConfig. The
    ``mlflow`` callback dict-key is replaced by :class:`MLflowTrainingCallback`
    bound to the live ``run_id`` (the bare class has no run_id to instantiate)."""
    rc = row.rendered_config
    model = _instantiate(rc["model"])
    datamodule = _instantiate(rc["data"])
    callbacks: list = []
    for name, spec in rc.get("callbacks", {}).items():
        if name == "mlflow":
            callbacks.append(MLflowTrainingCallback(run_id=run_id))
        else:
            callbacks.append(_instantiate(spec))
    trainer_cfg = TrainerConfig(**{k: v for k, v in rc["trainer"].items() if k != "callbacks"})
    return model, datamodule, callbacks, trainer_cfg


def train(row: TrainRow, *, ckpt_path: str | None = None) -> None:
    """Open MLflow run, instantiate, fit, close. FAILED on any exception."""
    torch_geometric.seed_everything(row.meta.seed)
    run_id = start_training_run(row, phase="fit")
    try:
        model, dm, callbacks, cfg = _build(row, run_id)
        Trainer(cfg, callbacks=callbacks).fit(model, dm, ckpt_path=ckpt_path)
    except BaseException:
        end_training_run(run_id, status="FAILED")
        raise
    end_training_run(run_id, status="FINISHED")


def evaluate(row: TrainRow, *, ckpt_path: str | None = None) -> dict[str, float]:
    """Open MLflow run, instantiate, test, log returned metrics, close."""
    torch_geometric.seed_everything(row.meta.seed)
    run_id = start_training_run(row, phase="test")
    try:
        model, dm, callbacks, cfg = _build(row, run_id)
        metrics = Trainer(cfg, callbacks=callbacks).test(model, dm, ckpt_path=ckpt_path)
        ts = int(time.time() * 1000)
        ms = [Metric(k, float(v), ts, 0) for k, v in _scalar_metrics(metrics).items()]
        MlflowClient().log_batch(run_id, metrics=ms)
    except BaseException:
        end_training_run(run_id, status="FAILED")
        raise
    end_training_run(run_id, status="FINISHED")
    return metrics


def extract(row: ExtractRow) -> None:
    """One-shot fusion-feature extraction. Idempotent on ``row.output_dir``."""
    from graphids.core.data.fusion_states import extract_fusion_states

    extract_fusion_states(
        checkpoints=row.extractor_ckpts,
        dataset=row.dataset,
        output_dir=row.output_dir,
        max_samples=row.max_samples,
        max_val_samples=row.max_val_samples,
        batch_size=row.batch_size,
        seed=row.seed,
        window_size=row.window_size,
        stride=row.stride,
        val_fraction=row.val_fraction,
    )


def run_row(row: Row, *, ckpt_path: str | None = None) -> None:
    runtime.setup()
    if isinstance(row, ExtractRow):
        extract(row)
        return
    runtime.register_preempt_handler(row)
    {"fit": train, "test": evaluate}[row.action](row, ckpt_path=ckpt_path)
