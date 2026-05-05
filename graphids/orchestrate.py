"""Dict → objects → Lightning bridge (v3).

Process-level setup lives in ``runtime_v3``. Preempt-resume delegated to
``pl.plugins.environments.SLURMEnvironment(auto_requeue=True,
requeue_signal=signal.SIGUSR2)`` — Lightning calls ``scontrol requeue``
natively, same job ID, ``afterok`` deps stay valid. No custom signal
handler.

Lightning owns: train/val loop, AMP, gradient clipping, optimizer state,
scheduler, callback lifecycle, MLflow run lifecycle (via ``MLFlowLogger``),
SLURM preempt-resume (via ``SLURMEnvironment`` plugin).

graphids owns:
- ``dm.setup("fit")`` BEFORE ``trainer.fit`` so ``model.prepare_from_datamodule``
  reads vocab/channel sizes.
- ``model.to(device)`` BEFORE dataloader build so the VRAM probe sees the
  right device.
- VGAE/DGI calibration via ``model.on_test_setup(dm, device)`` after ckpt
  load, before ``trainer.test``.
- Upstream LM lineage via ``client.log_inputs`` for fusion fits.
"""

from __future__ import annotations

import importlib
import os
import signal
from typing import Any

import lightning.pytorch as pl
import torch
import torch_geometric
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.plugins.environments import SLURMEnvironment
from mlflow.entities import LoggedModelInput

from graphids._fs import atomic_load
from graphids._mlflow import _find_logged_model_by_ckpt, identity_tags
from graphids.blueprint import AnalyzeRow, ExtractRow, Row, TrainRow
from graphids.core.models.base import strip_orig_mod_prefix
from graphids.runtime import setup


def _instantiate(spec: dict[str, Any]) -> Any:
    """Build ``{class_path, init_args}``; recurses on nested ``class_path`` blocks."""
    rec = lambda v: _instantiate(v) if isinstance(v, dict) and "class_path" in v else v  # noqa: E731
    init_args = {k: rec(v) for k, v in spec.get("init_args", {}).items()}
    mod, _, attr = spec["class_path"].rpartition(".")
    return getattr(importlib.import_module(mod), attr)(**init_args)


def _build(row: TrainRow) -> tuple[Any, Any, list, dict]:
    rc = row.rendered_config
    callbacks = [_instantiate(spec) for spec in rc.get("callbacks", {}).values()]
    return (
        _instantiate(rc["model"]),
        _instantiate(rc["data"]),
        callbacks,
        {k: v for k, v in rc["trainer"].items() if k != "callbacks"},
    )


def _device_from_kwargs(kw: dict) -> torch.device:
    if kw.get("accelerator") == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_trainer(callbacks: list, kw: dict, logger: MLFlowLogger) -> pl.Trainer:
    """``pl.Trainer`` with graphids defaults + SLURMEnvironment when in a job.

    SLURMEnvironment(auto_requeue=True, requeue_signal=USR2):
    - traps SIGUSR2 (USR1 conflicts with NCCL)
    - calls ``scontrol requeue $SLURM_JOB_ID`` — same job ID
    - downstream ``afterok`` chains stay valid across preemption
    Pairs with ``submit_row``'s ``--signal=USR2@N`` directive.
    """
    plugins: list[Any] = []
    if os.environ.get("SLURM_JOB_ID"):
        plugins.append(SLURMEnvironment(auto_requeue=True, requeue_signal=signal.SIGUSR2))
    return pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        plugins=plugins or None,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        **kw,
    )


def _logger_for(row: TrainRow, phase: str) -> MLFlowLogger:
    """``MLFlowLogger`` carrying graphids identity tags. Lifecycle is
    Lightning's: lazy-open on first ``run_id`` access, FINISHED/FAILED
    finalize on teardown. Tracking URI from ``MLFLOW_TRACKING_URI`` env.
    """
    return MLFlowLogger(
        experiment_name=f"graphids/{row.meta.dataset}/{row.meta.group}",
        run_name=row.identity.run_name,
        tags=identity_tags(row, phase),
    )


def _load_state_into_model(ckpt_path: str, model: torch.nn.Module) -> dict:
    """Restore weights from ckpt; align keys across compile prefix variants."""
    ckpt = atomic_load(ckpt_path, map_location="cpu", weights_only=True)
    state = strip_orig_mod_prefix(ckpt.get("state_dict", ckpt))
    remap = {k.replace("_orig_mod.", ""): k for k in model.state_dict()}
    state = {remap.get(k, k): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    if hasattr(model, "on_load_checkpoint"):
        model.on_load_checkpoint(ckpt)
    return ckpt


def _log_upstream_inputs(logger: MLFlowLogger, row: TrainRow) -> None:
    """``client.log_inputs`` for fusion upstream LM lineage. Triggers lazy
    run-open as a side effect (we need ``run_id`` before fit).
    """
    if not row.upstreams:
        return
    client = logger.experiment  # MlflowClient
    inputs, missing = [], []
    for u in row.upstreams:
        lm = _find_logged_model_by_ckpt(client, row.meta.dataset, u.ckpt_path)
        if lm is None:
            missing.append(f"{u.role}={u.ckpt_path}")
        else:
            inputs.append(LoggedModelInput(model_id=lm.model_id))
    if inputs:
        client.log_inputs(run_id=logger.run_id, models=inputs)
    if missing:
        import structlog

        structlog.get_logger(__name__).warning(
            "upstream_lm_missing",
            run_id=logger.run_id,
            dataset=row.meta.dataset,
            missing=missing,
        )


def _prepare(row: TrainRow, *, setup_stage: str) -> tuple[Any, Any, list, dict, torch.device]:
    """Shared bootstrap: seed, build, dm.setup, prepare_from_datamodule."""
    torch_geometric.seed_everything(row.meta.seed)
    model, dm, callbacks, kw = _build(row)
    device = _device_from_kwargs(kw)
    dm.setup(setup_stage)
    model.prepare_from_datamodule(dm)
    return model, dm, callbacks, kw, device


def train(row: TrainRow, *, ckpt_path: str | None = None) -> None:
    model, dm, callbacks, kw, device = _prepare(row, setup_stage="fit")
    model.to(device)  # before dataloader build — VRAM probe reads model.device

    logger = _logger_for(row, phase="fit")
    _log_upstream_inputs(logger, row)  # side-effect: lazy-opens the run
    trainer = _make_trainer(callbacks, kw, logger)
    trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)


def evaluate(row: TrainRow, *, ckpt_path: str | None = None) -> dict[str, float]:
    model, dm, callbacks, kw, device = _prepare(row, setup_stage="test")

    if ckpt_path:
        _load_state_into_model(ckpt_path, model)

    # VGAE/DGI calibration buffers (z-norm stats, SVDD center) refit at
    # test-start from fit-phase data — not persisted in state_dict.
    dm.setup("fit")
    model.to(device)
    model.on_test_setup(dm, device)

    logger = _logger_for(row, phase="test")
    trainer = _make_trainer(callbacks, kw, logger)
    # ckpt_path NOT passed — already restored above so calibration saw
    # trained weights. Lightning's ckpt-load would happen too late.
    trainer.test(model, datamodule=dm)
    return {k: float(v) for k, v in trainer.callback_metrics.items()}


def extract(row: ExtractRow) -> None:
    """One-shot fusion-feature extraction. Pure data transform — no MLflow run."""
    from graphids.core.data.extract import extract_states

    extract_states(
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


def analyze(row: AnalyzeRow) -> None:
    """Run the per-checkpoint artifact pipeline on a single ckpt."""
    from graphids.core.artifacts import Analyzer

    Analyzer(row).run()


def run_row(row: Row, *, ckpt_path: str | None = None) -> None:
    setup()
    if isinstance(row, ExtractRow):
        extract(row)
        return
    if isinstance(row, AnalyzeRow):
        analyze(row)
        return
    {"fit": train, "test": evaluate}[row.action](row, ckpt_path=ckpt_path)
