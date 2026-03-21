"""Trainer factory and optimizer/scheduler helpers."""

from __future__ import annotations

import structlog
import os
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn as nn
from .callbacks import RunMetadataCallback

from graphids.config import STAGE_MODEL_MAP

log = structlog.get_logger()


def _instantiate_callbacks(cfg) -> list:
    """Instantiate callbacks from config, add non-configurable ones."""
    from hydra.utils import instantiate

    cbs = [cb for cb in instantiate(cfg.callbacks).values() if cb is not None]
    cbs.append(RunMetadataCallback())
    return cbs


# model_type → the canonical stage that produces the teacher checkpoint.
# "curriculum" is preferred over "normal" for GAT since the teacher (large) always
# trains via curriculum. Derived from STAGE_MODEL_MAP with first-wins semantics.

_TEACHER_STAGE: dict[str, str] = {}
for _stage, _model in STAGE_MODEL_MAP.items():
    _TEACHER_STAGE.setdefault(_model, _stage)


def resolve_teacher_path(cfg, model_type: str) -> Path:
    """Auto-resolve teacher checkpoint path for KD.

    Resolution order:
    1. Explicit ``kd.model_path`` (manual override)
    2. Auto-resolve from ``kd.teacher_scale`` via the artifact resolver

    The ``teacher_scale`` field (default ``"large"``) makes the teacher
    reference scale-agnostic — today it's "large", but could be any
    variant that produces a checkpoint for the given model_type.
    """
    kd = next((a for a in cfg.get("auxiliaries", []) if a.type == "kd"), None)
    if kd and kd.model_path:
        return Path(kd.model_path)

    teacher_scale = kd.teacher_scale if kd else "large"
    stage = _TEACHER_STAGE.get(model_type)
    if stage is None:
        raise ValueError(f"No teacher stage mapping for model_type '{model_type}'")

    from graphids.config import resolve
    teacher_cfg = resolve(f"model_type={model_type}", f"scale={teacher_scale}", f"dataset={cfg.dataset}", f"seed={cfg.seed}")
    path = Path(teacher_cfg.checkpoints[model_type])
    if not path.exists():
        raise FileNotFoundError(
            f"Teacher checkpoint not found: {path}. "
            f"Train {model_type}/{teacher_scale} first, or set model_path explicitly."
        )
    log.info("auto_resolved_teacher", model_type=model_type, scale=teacher_scale, path=str(path))
    return path


def prepare_kd(
    cfg,
    model_type: str,
    num_ids: int,
    in_channels: int,
    device: torch.device,
) -> tuple[nn.Module | None, nn.Linear | None]:
    """Resolve, load, and prepare all KD components for a training stage.

    Returns ``(teacher, projection)`` when KD is active, or
    ``(None, None)`` when KD is disabled.  Centralizes the entire
    teacher lifecycle so training functions need only::

        teacher, projection = prepare_kd(cfg, "vgae", num_ids, in_ch, device)

    No if/else branching required in calling code.
    """
    if not any(a.type == "kd" for a in cfg.get("auxiliaries", [])):
        return None, None

    teacher_path = resolve_teacher_path(cfg, model_type)
    teacher = _load_teacher(str(teacher_path), model_type, cfg, num_ids, in_channels, device)

    # Projection layer only needed for VGAE (latent space alignment)
    projection = None
    if model_type == "vgae":
        from graphids.core.models.registry import get as registry_get

        tmp_student = registry_get("vgae").factory(cfg, num_ids, in_channels)
        projection = make_projection(tmp_student, teacher, "vgae", device)
        del tmp_student

    return teacher, projection


def _load_teacher(
    teacher_path: str,
    model_type: str,
    cfg,
    num_ids: int,
    in_channels: int,
    device: torch.device,
) -> nn.Module:
    """Load and freeze a teacher model from checkpoint.  Internal to prepare_kd()."""
    from graphids.core.models.registry import get as registry_get

    checkpoint = torch.load(teacher_path, map_location="cpu", weights_only=True)
    sd = _extract_state_dict(checkpoint)

    teacher_cfg_path = Path(teacher_path).parent / "config.yaml"
    if not teacher_cfg_path.exists():
        raise FileNotFoundError(
            f"Teacher config not found: {teacher_cfg_path}. "
            f"Cannot load teacher without its frozen config (risk of dimension mismatch)."
        )
    from omegaconf import OmegaConf
    tcfg = OmegaConf.load(teacher_cfg_path)

    # Infer num_ids from checkpoint embedding if present
    t_num_ids = num_ids
    for key in sd:
        if key.endswith("id_embedding.weight"):
            t_num_ids = sd[key].shape[0]
            break

    teacher = registry_get(model_type).factory(tcfg, t_num_ids, in_channels)

    # DQN checkpoints have nested state dict
    if model_type == "dqn":
        if "q_network" in sd:
            teacher.load_state_dict(sd["q_network"])
        elif "q_network_state_dict" in sd:
            teacher.load_state_dict(sd["q_network_state_dict"])
        else:
            teacher.load_state_dict(sd)
    else:
        teacher.load_state_dict(sd)

    log.info("loaded_teacher", model_type=model_type, path=teacher_path, num_ids=t_num_ids)

    teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


def make_projection(
    student_model: nn.Module,
    teacher: nn.Module,
    model_type: str,
    device: torch.device,
) -> nn.Linear | None:
    """Create projection layer if teacher/student latent dims differ."""
    if model_type == "vgae":
        s_dim = getattr(student_model, "latent_dim", getattr(student_model, "_latent_dim", 16))
        t_dim = getattr(teacher, "latent_dim", getattr(teacher, "_latent_dim", 96))
    elif model_type == "gat":
        s_dim = getattr(student_model, "hidden_channels", getattr(student_model, "out_channels", 2))
        t_dim = getattr(teacher, "hidden_channels", getattr(teacher, "out_channels", 2))
    else:
        return None

    if s_dim != t_dim:
        proj = nn.Linear(s_dim, t_dim).to(device)
        log.info("projection_layer_created", student_dim=s_dim, teacher_dim=t_dim)
        return proj
    return None


def _extract_state_dict(checkpoint) -> dict:
    """Handle Lightning checkpoint format, return clean state dict."""
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        sd = checkpoint["state_dict"]
        return {k.replace("model.", ""): v for k, v in sd.items() if k.startswith("model.")} or sd
    return checkpoint


def load_frozen_cfg(
    cfg, stage: str, model_type: str | None = None
):
    """Load the frozen config.yaml saved during training for *stage*.

    model_type defaults to the canonical owner of the stage (e.g. "autoencoder" → "vgae").

    Raises FileNotFoundError if the frozen config doesn't exist.
    """
    mt = model_type or STAGE_MODEL_MAP.get(stage, cfg.model_type)
    p = Path(cfg.checkpoints[mt]).parent / "config.yaml"
    if not p.exists():
        raise FileNotFoundError(
            f"Frozen config not found for stage '{stage}' (model_type={mt}). "
            f"The '{stage}' stage must be trained first (with config saved) "
            f"before dependent stages can load it."
        )
    try:
        from omegaconf import OmegaConf
        return OmegaConf.load(p)
    except Exception as e:
        raise RuntimeError(f"Could not load frozen config {p}: {e}") from e


def load_model(
    cfg,
    model_type: str,
    stage: str,
    num_ids: int,
    in_channels: int,
    device: torch.device,
) -> nn.Module:
    """Load a trained model using its frozen config and the registry."""
    from graphids.core.models.registry import get as registry_get

    frozen_cfg = load_frozen_cfg(cfg, stage, model_type=model_type)
    model = registry_get(model_type).factory(frozen_cfg, num_ids, in_channels)
    model.load_state_dict(torch.load(cfg.checkpoints[model_type], map_location="cpu", weights_only=True))
    model.to(device)
    model.eval()
    return model


def build_optimizer_dict(optimizer, cfg):
    """Return optimizer or {optimizer, lr_scheduler} dict for Lightning."""
    t = cfg.training
    if not t.use_scheduler or not t.scheduler:
        return optimizer

    from hydra.utils import instantiate

    sched = instantiate(t.scheduler, optimizer=optimizer)

    if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": sched, "monitor": t.monitor_metric},
        }
    return {"optimizer": optimizer, "lr_scheduler": sched}


def make_trainer(
    cfg,
    stage: str,
    extra_callbacks: list | None = None,
) -> pl.Trainer:
    """Create a Lightning Trainer with standard callbacks."""
    t = cfg.training
    torch.backends.cudnn.benchmark = t.cudnn_benchmark

    csv_logger = pl.loggers.CSVLogger(save_dir=".", name="", version="")

    callbacks = _instantiate_callbacks(cfg)

    if extra_callbacks:
        callbacks.extend(extra_callbacks)

    # On SLURM: enable auto-requeue so Lightning catches SIGUSR1,
    # saves .pl_auto_save.ckpt, and calls scontrol requeue automatically.
    # The bash wrapper (_preamble.sh) forwards USR1 from SLURM to Python.
    plugins = []
    if os.environ.get("SLURM_JOB_ID"):
        from pytorch_lightning.plugins.environments import SLURMEnvironment

        plugins.append(SLURMEnvironment(auto_requeue=True))

    return pl.Trainer(
        default_root_dir=".",
        max_epochs=t.max_epochs,
        accelerator="auto",
        devices="auto",
        precision=t.precision,
        gradient_clip_val=t.gradient_clip,
        accumulate_grad_batches=t.accumulate_grad_batches,
        callbacks=callbacks,
        logger=csv_logger,
        plugins=plugins or None,
        log_every_n_steps=t.log_every_n_steps,
        enable_progress_bar=True,
        deterministic=t.deterministic,
    )
