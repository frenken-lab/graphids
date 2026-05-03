"""Dict→objects→Lightning bridge. importlib-instantiates from
``rendered_config``, opens/closes the MLflow run, and dispatches on
``row.action``.

Lightning owns the train/val loop, AMP autocast, gradient clipping,
optimizer state, scheduler stepping, and the callback lifecycle.
graphids only owns:

- DM bind + setup BEFORE Lightning constructs dataloaders (DMs are not
  ``LightningDataModule``s — they're our own protocol).
- ``model.prepare_from_datamodule(dm)`` so lazy ``_build()`` runs with
  DM-supplied vocab/channel sizes before the budget probe needs them.
- VGAE/DGI calibration via ``model.on_test_setup(dm, device)`` after
  ckpt load, before the test loop.
"""

from __future__ import annotations

import importlib
import os
import time
from typing import Any

import lightning.pytorch as pl
import torch
import torch_geometric
from mlflow.entities import Metric
from mlflow.tracking import MlflowClient

from graphids import runtime
from graphids._fs import atomic_load
from graphids._mlflow import (
    _scalar_metrics,
    end_training_run,
    start_training_run,
)
from graphids.blueprint import ExtractRow, Row, TrainRow
from graphids.core.models.base import strip_orig_mod_prefix


def _instantiate(spec: dict[str, Any]) -> Any:
    """Build ``{class_path, init_args}``; recurses on nested class_path blocks
    in init_args (e.g. GAT's ``loss_fn``)."""
    rec = lambda v: _instantiate(v) if isinstance(v, dict) and "class_path" in v else v  # noqa: E731
    ia = {k: rec(v) for k, v in spec.get("init_args", {}).items()}
    mod, _, attr = spec["class_path"].rpartition(".")
    return getattr(importlib.import_module(mod), attr)(**ia)


def _build(row: TrainRow) -> tuple[Any, Any, list, dict]:
    """Instantiate model + datamodule + callbacks + trainer kwargs.

    All callbacks go through ``_instantiate``; ``MLflowTrainingCallback``
    reads its run_id from ``$GRAPHIDS_MLFLOW_RUN_ID`` (set by
    ``train``/``evaluate`` before this call).
    """
    rc = row.rendered_config
    model = _instantiate(rc["model"])
    datamodule = _instantiate(rc["data"])
    callbacks = [_instantiate(spec) for spec in rc.get("callbacks", {}).values()]
    trainer_kwargs = {k: v for k, v in rc["trainer"].items() if k != "callbacks"}
    return model, datamodule, callbacks, trainer_kwargs


def _device_from_kwargs(trainer_kwargs: dict) -> torch.device:
    """Resolve the device pl.Trainer will pick, so calibration / ckpt-load
    can run on the same one before fit."""
    if trainer_kwargs.get("accelerator") == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_trainer(callbacks: list, trainer_kwargs: dict) -> pl.Trainer:
    """``pl.Trainer`` with graphids defaults that don't belong in jsonnet.

    - ``logger=False``: MLflow is wired via ``MLflowTrainingCallback``;
      Lightning's default TensorBoard logger would write a parallel
      ``lightning_logs/`` we never read.
    - ``enable_progress_bar=False``: SLURM logs go to ``*_log.err``, the
      tqdm bar would smear stderr.
    - ``num_sanity_val_steps=0``: our val path runs full epochs and our
      DM constructs val loaders only after ``setup``; the default sanity
      pass races that.
    """
    return pl.Trainer(
        callbacks=callbacks,
        logger=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        **trainer_kwargs,
    )


def _load_state_into_model(ckpt_path: str, model: torch.nn.Module) -> dict:
    """Read a graphids/Lightning ckpt, restore weights into ``model``,
    fire ``on_load_checkpoint``. Returns the raw ckpt dict.

    ``strict=False`` tolerates removed buffers (e.g. DGI ``svdd_calibrated``,
    dropped when centroid fit moved from state_dict to test-start).
    """
    ckpt = atomic_load(ckpt_path, map_location="cpu", weights_only=True)
    state = strip_orig_mod_prefix(ckpt.get("state_dict", ckpt))
    # Align ckpt to target's compile-prefix convention.
    remap = {k.replace("_orig_mod.", ""): k for k in model.state_dict().keys()}
    state = {remap.get(k, k): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    if hasattr(model, "on_load_checkpoint"):
        model.on_load_checkpoint(ckpt)
    return ckpt


def train(row: TrainRow, *, ckpt_path: str | None = None) -> None:
    """Open MLflow run, instantiate, fit, close. FAILED on any exception."""
    torch_geometric.seed_everything(row.meta.seed)
    run_id = start_training_run(row, phase="fit")
    os.environ["GRAPHIDS_MLFLOW_RUN_ID"] = run_id
    try:
        model, dm, callbacks, trainer_kwargs = _build(row)
        device = _device_from_kwargs(trainer_kwargs)

        # Wire the DM and model BEFORE Lightning takes over. Every DM
        # implements ``bind(*, model, device)``; ``setup("fit")`` populates
        # ``num_ids``/``in_channels``/``num_classes``/``test_datasets`` so
        # ``prepare_from_datamodule`` can lazy-build the model.
        dm.bind(model=model, device=device)
        dm.setup("fit")
        model.prepare_from_datamodule(dm)
        # Lightning's ``trainer.datamodule`` is None when we pass dataloaders
        # directly (our DMs aren't ``LightningDataModule`` subclasses); stash
        # the DM on the model so callbacks (MLflow / curriculum) can find it.
        model._graphids_dm = dm

        trainer = _make_trainer(callbacks, trainer_kwargs)
        trainer.fit(
            model,
            train_dataloaders=dm.train_dataloader(),
            val_dataloaders=dm.val_dataloader(),
            ckpt_path=ckpt_path,
        )
    except BaseException:
        end_training_run(run_id, status="FAILED")
        raise
    end_training_run(run_id, status="FINISHED")


def evaluate(row: TrainRow, *, ckpt_path: str | None = None) -> dict[str, float]:
    """Open MLflow run, instantiate, test, log returned metrics, close."""
    torch_geometric.seed_everything(row.meta.seed)
    run_id = start_training_run(row, phase="test")
    os.environ["GRAPHIDS_MLFLOW_RUN_ID"] = run_id
    try:
        model, dm, callbacks, trainer_kwargs = _build(row)
        device = _device_from_kwargs(trainer_kwargs)

        dm.bind(model=model, device=device)
        dm.setup("test")
        model.prepare_from_datamodule(dm)
        model._graphids_dm = dm

        # Restore weights BEFORE on_test_setup — score-based detectors
        # (VGAE/DGI) need the trained encoder to fit calibration buffers.
        if ckpt_path:
            _load_state_into_model(ckpt_path, model)

        # VGAE/DGI calibration buffers (z-norm stats, SVDD center) are
        # deterministic functions of (trained encoder, fit-phase data) and
        # are NOT persisted through state_dict — refit at test-start.
        dm.setup("fit")
        model.to(device)
        model.on_test_setup(dm, device)

        trainer = _make_trainer(callbacks, trainer_kwargs)
        # ckpt_path NOT passed — we already restored above so the calibration
        # hook saw trained weights. Lightning's ckpt-load would happen too late.
        trainer.test(model, dataloaders=dm.test_dataloader())

        metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}
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
