"""Trainer factory and optimizer/scheduler helpers."""

from __future__ import annotations

import logging
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import DeviceStatsMonitor, EarlyStopping, ModelCheckpoint

from graphids.config import MLFLOW_TRACKING_URI, PipelineConfig, stage_dir

log = logging.getLogger(__name__)


def load_teacher(
    teacher_path: str,
    model_type: str,
    cfg: PipelineConfig,
    num_ids: int,
    in_channels: int,
    device: torch.device,
) -> nn.Module:
    """Load a teacher model from its checkpoint for knowledge distillation.

    Uses the model registry (``registry.get(model_type).factory()``) to
    construct the architecture, then loads weights from *teacher_path*.
    Dimensions come from the **frozen config.json** saved alongside the
    checkpoint — never from the student config — preventing shape mismatches
    when teacher and student have different hidden sizes.

    The returned model is moved to *device*, set to eval mode, and has all
    parameters frozen (``requires_grad=False``).
    """
    from graphids.core.models.registry import get as registry_get

    checkpoint = torch.load(teacher_path, map_location="cpu", weights_only=True)
    sd = _extract_state_dict(checkpoint)

    teacher_cfg_path = Path(teacher_path).parent / "config.json"
    if not teacher_cfg_path.exists():
        raise FileNotFoundError(
            f"Teacher config not found: {teacher_cfg_path}. "
            f"Cannot load teacher without its frozen config (risk of dimension mismatch)."
        )
    tcfg = PipelineConfig.load(teacher_cfg_path)

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

    log.info("Loaded %s teacher from %s (num_ids=%d)", model_type, teacher_path, t_num_ids)

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
        log.info("Projection layer: %d -> %d", s_dim, t_dim)
        return proj
    return None


def _extract_state_dict(checkpoint) -> dict:
    """Handle Lightning checkpoint format, return clean state dict."""
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        sd = checkpoint["state_dict"]
        return {k.replace("model.", ""): v for k, v in sd.items() if k.startswith("model.")} or sd
    return checkpoint


def _cross_model_path(cfg: PipelineConfig, model_type: str, stage: str, filename: str) -> Path:
    """Build a path for a specific model_type (may differ from cfg.model_type).

    Used when loading another model's artifacts (e.g. loading VGAE checkpoint from GAT config).
    """
    aux_suffix = f"_{cfg.auxiliaries[0].type}" if cfg.auxiliaries else ""
    return (
        Path(cfg.experiment_root)
        / cfg.dataset
        / f"{model_type}_{cfg.scale}_{stage}{aux_suffix}"
        / filename
    )


from graphids.config.constants import STAGE_MODEL_MAP as _STAGE_MODEL_TYPE


def load_frozen_cfg(
    cfg: PipelineConfig, stage: str, model_type: str | None = None
) -> PipelineConfig:
    """Load the frozen config.json saved during training for *stage*.

    model_type defaults to the canonical owner of the stage (e.g. "autoencoder" → "vgae").
    When cfg.model_type already matches the stage owner, this is equivalent to config_path(cfg, stage).

    Raises FileNotFoundError if the frozen config doesn't exist.
    """
    from graphids.config import config_path

    mt = model_type or _STAGE_MODEL_TYPE.get(stage, cfg.model_type)
    if mt == cfg.model_type:
        p = config_path(cfg, stage)
    else:
        p = _cross_model_path(cfg, mt, stage, "config.json")
    if not p.exists():
        raise FileNotFoundError(
            f"Frozen config not found: {p}. "
            f"The '{stage}' stage must be trained first (with config saved) "
            f"before dependent stages can load it."
        )
    try:
        return PipelineConfig.load(p)
    except Exception as e:
        raise RuntimeError(f"Could not load frozen config {p}: {e}") from e


def load_model(
    cfg: PipelineConfig,
    model_type: str,
    stage: str,
    num_ids: int,
    in_channels: int,
    device: torch.device,
) -> nn.Module:
    """Load a trained model using its frozen config and the registry.

    Replaces the old ``load_vgae`` / ``load_gat`` helpers with a single
    generic loader that works for any registered model type.
    """
    from graphids.core.models.registry import get as registry_get

    frozen_cfg = load_frozen_cfg(cfg, stage, model_type=model_type)
    ckpt = _cross_model_path(cfg, model_type, stage, "best_model.pt")
    model = registry_get(model_type).factory(frozen_cfg, num_ids, in_channels)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.to(device)
    model.eval()
    return model


def build_optimizer_dict(optimizer, cfg: PipelineConfig):
    """Return optimizer or {optimizer, lr_scheduler} dict for Lightning."""
    t = cfg.training
    if not t.use_scheduler:
        return optimizer

    t_max = t.scheduler_t_max if t.scheduler_t_max > 0 else t.max_epochs

    if t.scheduler_type == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
    elif t.scheduler_type == "step":
        sched = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=t.scheduler_step_size,
            gamma=t.scheduler_gamma,
        )
    elif t.scheduler_type == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=t.monitor_mode,
            patience=t.patience // 2,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": sched, "monitor": t.monitor_metric},
        }
    else:
        log.warning("Unknown scheduler_type=%s, skipping", t.scheduler_type)
        return optimizer

    return {"optimizer": optimizer, "lr_scheduler": sched}


def _setup_mlflow_autolog() -> None:
    """Enable MLflow autolog for PyTorch Lightning.

    Called once per training process. Sets tracking URI and enables
    automatic logging of metrics, params, and checkpoints.
    """
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.pytorch.autolog(
        checkpoint=True,
        log_every_n_epoch=1,
        log_models=False,  # We save checkpoints via ModelCheckpoint callback
    )


def make_trainer(
    cfg: PipelineConfig,
    stage: str,
    extra_callbacks: list | None = None,
) -> pl.Trainer:
    """Create a Lightning Trainer with standard callbacks."""
    t = cfg.training
    out = stage_dir(cfg, stage)
    out.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.benchmark = t.cudnn_benchmark

    # Enable MLflow autolog (idempotent — safe to call multiple times)
    _setup_mlflow_autolog()

    callbacks = [
        ModelCheckpoint(
            dirpath=str(out),
            filename="best_model",
            monitor=t.monitor_metric,
            mode=t.monitor_mode,
            save_top_k=t.save_top_k,
            save_on_train_epoch_end=False,
        ),
        EarlyStopping(
            monitor=t.monitor_metric,
            patience=t.patience,
            mode=t.monitor_mode,
            check_on_train_epoch_end=False,
        ),
        DeviceStatsMonitor(cpu_stats=False),
    ]

    if extra_callbacks:
        callbacks.extend(extra_callbacks)

    return pl.Trainer(
        default_root_dir=str(out),
        max_epochs=t.max_epochs,
        accelerator="auto",
        devices="auto",
        precision=t.precision,
        gradient_clip_val=t.gradient_clip,
        accumulate_grad_batches=t.accumulate_grad_batches,
        callbacks=callbacks,
        log_every_n_steps=t.log_every_n_steps,
        enable_progress_bar=True,
        deterministic=t.deterministic,
    )
