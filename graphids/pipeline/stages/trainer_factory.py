"""Trainer factory, KD preparation, and model loading."""

from __future__ import annotations

import os
from pathlib import Path

import pytorch_lightning as pl
import structlog
import torch
import torch.nn as nn

from graphids.config import STAGE_MODEL_MAP

log = structlog.get_logger()

# model_type → canonical stage that produces the teacher checkpoint.
_TEACHER_STAGE: dict[str, str] = {}
for _stage, _model in STAGE_MODEL_MAP.items():
    _TEACHER_STAGE.setdefault(_model, _stage)


def make_trainer(cfg, stage: str, **overrides) -> pl.Trainer:
    """Create a Lightning Trainer from config with optional overrides.

    Stages like fusion pass overrides for callbacks, max_epochs, logger, etc.
    while inheriting precision, gradient_clip, deterministic, etc. from cfg.training.
    """
    if "callbacks" not in overrides:
        from hydra.utils import instantiate
        overrides["callbacks"] = [cb for cb in instantiate(cfg.callbacks).values() if cb is not None]

    kwargs = dict(
        max_epochs=cfg.training.max_epochs,
        accelerator="gpu" if cfg.device == "cuda" and torch.cuda.is_available() else "cpu",
        devices=1,
        gradient_clip_val=cfg.training.gradient_clip,
        precision=cfg.training.precision,
        log_every_n_steps=cfg.training.log_every_n_steps,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        deterministic=cfg.training.deterministic,
        benchmark=cfg.training.cudnn_benchmark,
        enable_progress_bar=not bool(os.environ.get("SLURM_JOB_ID")),
    )
    kwargs.update(overrides)
    return pl.Trainer(**kwargs)


def build_datamodule(cfg, stage: str) -> tuple[pl.LightningDataModule, torch.device]:
    """Build DataModule + resolve device for any training stage."""
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    if stage == "fusion":
        from graphids.core.preprocessing import FusionDataModule
        dm = FusionDataModule(cfg, load_model_fn=load_model)
        dm.setup("fit")
        return dm, dm.device

    if stage == "temporal":
        from graphids.core.preprocessing import TemporalDataModule
        dm = TemporalDataModule(cfg, load_model_fn=load_model)
        dm.setup("fit")
        return dm, dm.device

    # All graph-based stages start from CANBusDataModule
    from graphids.core.preprocessing import CANBusDataModule
    raw_dm = CANBusDataModule.from_cfg(cfg)
    raw_dm.setup("fit")
    raw_dm.populate_config(cfg)

    if stage == "curriculum":
        dm = _build_curriculum_dm(raw_dm, cfg, device)
        return dm, device

    return raw_dm, device


def _build_curriculum_dm(raw_dm, cfg, device):
    """Score difficulty with VGAE, build CurriculumDataModule."""
    import gc

    from graphids.core.preprocessing.curriculum import CurriculumDataModule

    vgae = load_model(cfg, "vgae", "autoencoder", device)
    normals = [g for g in raw_dm.train_dataset if int(g.y[0]) == 0]
    attacks = [g for g in raw_dm.train_dataset if int(g.y[0]) == 1]

    scores = vgae.score_difficulty(normals, canid_weight=cfg.vgae.canid_weight)

    del vgae
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return CurriculumDataModule(normals, attacks, scores, list(raw_dm.val_dataset), cfg)


def build_module(cfg, stage: str, device: torch.device, dm=None) -> pl.LightningModule:
    """Build the Lightning module for any training stage."""
    if stage == "fusion":
        from graphids.core.models.fusion_baselines import build_fusion_module
        return build_fusion_module(cfg, device)
    if stage == "temporal":
        from graphids.core.models.temporal import TemporalLightningModule
        return TemporalLightningModule.from_datamodule(cfg, dm)

    # Graph-based stages (autoencoder, normal, curriculum): generic via registry
    from graphids.core.models.registry import get_module_cls
    module_cls = get_module_cls(cfg.model_type)
    teacher, projection = prepare_kd(cfg, cfg.model_type, device)
    return module_cls(cfg, teacher=teacher, projection=projection)


def prepare_kd(
    cfg, model_type: str, device: torch.device,
) -> tuple[nn.Module | None, nn.Linear | None]:
    """Resolve teacher path, load & freeze teacher, create projection if needed.

    Reads num_ids and in_channels from cfg (populated by DataModule.populate_config).
    Returns (teacher, projection) when KD is active, or (None, None) otherwise.
    """
    if not any(a.type == "kd" for a in cfg.get("auxiliaries", [])):
        return None, None

    num_ids, in_channels = cfg.num_ids, cfg.in_channels

    # --- Resolve teacher checkpoint path ---
    kd = next(a for a in cfg.get("auxiliaries", []) if a.type == "kd")
    if kd.get("model_path"):
        teacher_path = Path(kd.model_path)
    else:
        teacher_scale = kd.get("teacher_scale", "large")
        stage = _TEACHER_STAGE.get(model_type)
        if stage is None:
            raise ValueError(f"No teacher stage mapping for model_type '{model_type}'")
        from graphids.config import resolve
        teacher_cfg = resolve(f"model_type={model_type}", f"scale={teacher_scale}",
                              f"dataset={cfg.dataset}", f"seed={cfg.seed}")
        teacher_path = Path(teacher_cfg.checkpoints[model_type])
        if not teacher_path.exists():
            raise FileNotFoundError(
                f"Teacher checkpoint not found: {teacher_path}. "
                f"Train {model_type}/{teacher_scale} first, or set model_path explicitly."
            )

    # --- Load and freeze teacher ---
    teacher, tcfg = _load_checkpoint(teacher_path, model_type, device)
    for p in teacher.parameters():
        p.requires_grad = False

    # --- Projection layer (VGAE latent space alignment) ---
    projection = None
    if model_type == "vgae":
        s_dim = cfg.vgae.latent_dim
        t_dim = tcfg.vgae.latent_dim
        if s_dim != t_dim:
            log.info("projection_layer", student_dim=s_dim, teacher_dim=t_dim)
            projection = nn.Linear(s_dim, t_dim).to(device)

    return teacher, projection


def _load_checkpoint(
    ckpt_path: Path, model_type: str, device: torch.device,
) -> tuple[nn.Module, object]:
    """Load a checkpoint, return (inner model on device in eval mode, hparams cfg)."""
    from graphids.core.models.registry import get_module_cls

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    module = get_module_cls(model_type).load_from_checkpoint(
        str(ckpt_path), map_location="cpu", weights_only=False,
    )
    model = module.model
    model.to(device).eval()
    tcfg = module.hparams.get("cfg", {})
    if isinstance(tcfg, dict):
        from omegaconf import OmegaConf
        tcfg = OmegaConf.create(tcfg)
    return model, tcfg


def load_model(
    cfg, model_type: str, stage: str, device: torch.device,
) -> nn.Module:
    """Load a trained model's inner nn.Module. Returns frozen model on device."""
    ckpt_path = Path(cfg.checkpoints[model_type])
    model, _ = _load_checkpoint(ckpt_path, model_type, device)
    return model


