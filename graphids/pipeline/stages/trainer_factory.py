"""Trainer factory, KD preparation, and model loading."""

from __future__ import annotations

import os
from pathlib import Path

import pytorch_lightning as pl
import structlog
import torch
import torch.nn as nn

from graphids.config import STAGE_MODEL_MAP
from .callbacks import RunMetadataCallback

log = structlog.get_logger()

# model_type → canonical stage that produces the teacher checkpoint.
_TEACHER_STAGE: dict[str, str] = {}
for _stage, _model in STAGE_MODEL_MAP.items():
    _TEACHER_STAGE.setdefault(_model, _stage)


def make_trainer(cfg, stage: str) -> pl.Trainer:
    """Create a Lightning Trainer from config."""
    from hydra.utils import instantiate

    cbs = [cb for cb in instantiate(cfg.callbacks).values() if cb is not None]
    cbs.append(RunMetadataCallback())

    return pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator="gpu" if cfg.device == "cuda" and torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=cbs,
        gradient_clip_val=cfg.training.gradient_clip,
        precision=cfg.training.precision,
        log_every_n_steps=cfg.training.log_every_n_steps,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        deterministic=cfg.training.deterministic,
        benchmark=cfg.training.cudnn_benchmark,
        enable_progress_bar=bool(os.environ.get("SLURM_JOB_ID")),
    )


def prepare_kd(
    cfg, model_type: str, num_ids: int, in_channels: int, device: torch.device,
) -> tuple[nn.Module | None, nn.Linear | None]:
    """Resolve teacher path, load & freeze teacher, create projection if needed.

    Returns (teacher, projection) when KD is active, or (None, None) otherwise.
    """
    if not any(a.type == "kd" for a in cfg.get("auxiliaries", [])):
        return None, None

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
    from graphids.core.models.registry import get as registry_get
    from omegaconf import OmegaConf

    checkpoint = torch.load(str(teacher_path), map_location="cpu", weights_only=True)
    sd = checkpoint
    if isinstance(sd, dict) and "state_dict" in sd:
        raw = sd["state_dict"]
        sd = {k.replace("model.", ""): v for k, v in raw.items() if k.startswith("model.")} or raw

    tcfg_path = teacher_path.parent / "config.yaml"
    if not tcfg_path.exists():
        raise FileNotFoundError(f"Teacher config not found: {tcfg_path}")
    tcfg = OmegaConf.load(tcfg_path)

    t_num_ids = num_ids
    for key in sd:
        if key.endswith("id_embedding.weight"):
            t_num_ids = sd[key].shape[0]
            break

    teacher = registry_get(model_type)(tcfg, t_num_ids, in_channels)

    if model_type == "dqn":
        teacher.load_state_dict(sd.get("q_network") or sd.get("q_network_state_dict") or sd)
    else:
        teacher.load_state_dict(sd)

    log.info("loaded_teacher", model_type=model_type, path=str(teacher_path), num_ids=t_num_ids)
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # --- Projection layer (VGAE latent space alignment) ---
    projection = None
    if model_type == "vgae":
        tmp = registry_get("vgae")(cfg, num_ids, in_channels)
        projection = make_projection(tmp, teacher, "vgae", device)
        del tmp

    return teacher, projection


def make_projection(
    student: nn.Module, teacher: nn.Module, model_type: str, device: torch.device,
) -> nn.Linear | None:
    """Create projection layer if teacher/student latent dims differ."""
    if model_type == "vgae":
        s_dim = getattr(student, "latent_dim", getattr(student, "_latent_dim", 16))
        t_dim = getattr(teacher, "latent_dim", getattr(teacher, "_latent_dim", 96))
    elif model_type == "gat":
        s_dim = getattr(student, "hidden_channels", getattr(student, "out_channels", 2))
        t_dim = getattr(teacher, "hidden_channels", getattr(teacher, "out_channels", 2))
    else:
        return None
    if s_dim != t_dim:
        log.info("projection_layer", student_dim=s_dim, teacher_dim=t_dim)
        return nn.Linear(s_dim, t_dim).to(device)
    return None


def load_frozen_cfg(cfg, stage: str, model_type: str | None = None):
    """Load the frozen config.yaml saved by a prior training stage."""
    mt = model_type or STAGE_MODEL_MAP.get(stage, stage)
    p = Path(cfg.checkpoints.get(mt, "")).parent / "config.yaml"
    if not p.exists():
        raise FileNotFoundError(
            f"Frozen config not found for stage '{stage}' (model_type={mt}). "
            f"The '{stage}' stage must be trained first."
        )
    from omegaconf import OmegaConf
    return OmegaConf.load(p)


def load_model(
    cfg, model_type: str, stage: str, num_ids: int, in_channels: int, device: torch.device,
) -> nn.Module:
    """Load a trained model using its frozen config and the registry."""
    from graphids.core.models.registry import get as registry_get

    frozen_cfg = load_frozen_cfg(cfg, stage, model_type=model_type)
    model = registry_get(model_type)(frozen_cfg, num_ids, in_channels)
    model.load_state_dict(torch.load(cfg.checkpoints[model_type], map_location="cpu", weights_only=True))
    model.to(device).eval()
    return model


def build_optimizer_dict(optimizer, cfg):
    """Return optimizer or {optimizer, lr_scheduler} dict for Lightning."""
    if not cfg.training.use_scheduler or cfg.training.scheduler is None:
        return optimizer

    from hydra.utils import instantiate
    sched = instantiate(cfg.training.scheduler, optimizer=optimizer)
    return {"optimizer": optimizer, "lr_scheduler": {"scheduler": sched, "monitor": cfg.training.monitor_metric}}
