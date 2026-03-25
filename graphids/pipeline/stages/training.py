"""Training stages: generic runner for all stages (autoencoder, curriculum, normal, fusion, temporal)."""

from __future__ import annotations

import gc
import os
import structlog
from pathlib import Path

import pytorch_lightning as pl
import torch

from .trainer_factory import build_datamodule, build_module, make_trainer

log = structlog.get_logger()


def _resume_ckpt_path(cfg, stage: str) -> str | None:
    """Find a checkpoint to resume from.

    Resolution order:
    1. ``KD_GAT_CKPT_PATH`` env var — explicit override from orchestrator
       (set by orchestrator when retrying a timed-out stage).
    2. Lightning auto-save — ``.pl_auto_save.ckpt`` in persistent_root,
       written by ``SLURMEnvironment(auto_requeue=True)`` on SIGUSR1.
    """
    # 1. Explicit override from orchestrator
    path = os.environ.get("KD_GAT_CKPT_PATH")
    try:
        del os.environ["KD_GAT_CKPT_PATH"]
    except KeyError:
        pass
    if path and Path(path).exists():
        log.info("resume_from_orchestrator_checkpoint", path=path)
        return path
    if path:
        log.warning("checkpoint_path_not_found", path=path)

    # 2. Lightning auto-save from SLURMEnvironment (timeout requeue)
    auto_save = Path.cwd() / ".pl_auto_save.ckpt"
    if auto_save.exists():
        log.info("resume_from_auto_save", path=str(auto_save))
        return str(auto_save)

    return None


def _save_and_cleanup(module, trainer, cfg, stage: str) -> dict:
    """Extract results after training.

    All modules save via Lightning's ModelCheckpoint (.ckpt).
    RL agents persist extra state via on_save_checkpoint hooks.
    """
    ckpt = getattr(trainer.checkpoint_callback, "best_model_path", "")

    metrics = {}
    if trainer.callback_metrics:
        metrics = {k: v.item() if hasattr(v, "item") else v
                   for k, v in trainer.callback_metrics.items()}

    if stage == "fusion":
        metrics["fusion_method"] = cfg.fusion.method

    log.info("training_complete", stage=stage, checkpoint=ckpt)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"checkpoint": ckpt, "metrics": metrics}


def train_stage(cfg) -> dict:
    """Generic training for all stages: autoencoder, curriculum, normal, fusion, temporal."""
    stage = cfg.stage

    # Early exit for disabled temporal
    if stage == "temporal" and not cfg.temporal.enabled:
        log.warning("temporal.enabled=False, skipping")
        return {"status": "skipped", "reason": "temporal.enabled=False"}

    pl.seed_everything(cfg.seed)
    dm, device = build_datamodule(cfg, stage)
    module = build_module(cfg, stage, device, dm=dm)

    overrides = module.trainer_overrides(cfg, dm) if hasattr(module, "trainer_overrides") else {}
    trainer = make_trainer(cfg, stage, **overrides)

    trainer.fit(module, datamodule=dm, ckpt_path=_resume_ckpt_path(cfg, stage))
    return _save_and_cleanup(module, trainer, cfg, stage)
