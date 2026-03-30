"""Lightning training helpers shared across VGAE, GAT, and DGI modules."""

import contextlib
from typing import TypedDict

import structlog
import torch
import torch.nn.functional as F
from torch import Tensor

_log = structlog.get_logger()


class KDAuxiliary(TypedDict, total=False):
    """Schema for KD auxiliary config items — validated by jsonargparse at parse time."""
    type: str
    alpha: float
    # VGAE KD
    vgae_latent_weight: float
    vgae_recon_weight: float
    # GAT KD
    temperature: float
    # Teacher resolution
    teacher_scale: str
    model_path: str


class OOMSkipMixin:
    """Skip batch on CUDA OOM. Lightning natively handles training_step returning None."""

    def _oom_safe_step(self, batch, batch_idx, step_fn):
        try:
            return step_fn(batch, batch_idx)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            _log.warning("oom_batch_skipped", batch_idx=batch_idx,
                         num_graphs=batch.num_graphs, num_nodes=batch.num_nodes)
            return None


def soft_label_kd_loss(student_logits, teacher_logits, temperature: float):
    """Hinton soft-label KD loss: KL(student/T || teacher/T) * T^2."""
    return F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction="batchmean",
    ) * (temperature ** 2)


def focal_loss(logits, targets, gamma: float = 2.0):
    """Focal loss (Lin et al. 2017) for class-imbalanced classification."""
    ce = F.cross_entropy(logits, targets, reduction="none")
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()


@contextlib.contextmanager
def teacher_on_device(module, device):
    """Move KD teacher to *device* for inference, return to CPU after.

    Teacher is stored outside ``nn.Module._modules`` (via ``__dict__``) so
    Lightning never auto-transfers it to GPU.  This context manager is the
    only code path that moves it onto the accelerator.
    """
    teacher = module.teacher
    if teacher is None:
        yield
        return
    teacher.to(device)
    try:
        yield
    finally:
        teacher.to("cpu")


def binary_test_metrics():
    """Standard binary classification MetricCollection shared by all Lightning modules."""
    from torchmetrics import MetricCollection
    from torchmetrics.classification import (
        BinaryAccuracy, BinaryAUROC, BinaryF1Score,
        BinaryPrecision, BinaryRecall, BinarySpecificity,
    )
    return MetricCollection({
        "accuracy": BinaryAccuracy(), "f1": BinaryF1Score(),
        "precision": BinaryPrecision(), "recall": BinaryRecall(),
        "specificity": BinarySpecificity(), "auc": BinaryAUROC(),
    })


# ---------------------------------------------------------------------------
# Model loading + KD preparation (used by module __init__)
# ---------------------------------------------------------------------------


_MODULE_PATHS: dict[str, str] = {
    "vgae": "graphids.core.models.vgae.VGAEModule",
    "gat": "graphids.core.models.gat.GATModule",
    "dgi": "graphids.core.models.dgi.DGIModule",
    "fusion": "graphids.core.models.bandit.BanditFusionModule",
    "dqn": "graphids.core.models.dqn.DQNFusionModule",
}


def safe_load_checkpoint(model_type: str, ckpt_path, *, map_location="cpu"):
    """load_from_checkpoint with migration guard for pre-flatten checkpoints."""
    import importlib
    from pathlib import Path

    dotted = _MODULE_PATHS.get(model_type)
    if dotted is None:
        raise KeyError(
            f"No module class for '{model_type}'. "
            f"Available: {list(_MODULE_PATHS)}"
        )
    module_path, cls_name = dotted.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_path), cls_name)

    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    try:
        return cls.load_from_checkpoint(
            str(ckpt_path), map_location=map_location, weights_only=True,
        )
    except TypeError as exc:
        if any(k in str(exc) for k in ("vgae", "gat", "dgi", "training", "fusion")):
            raise RuntimeError(
                f"Checkpoint {ckpt_path} has nested hparams (pre-flatten format). "
                "Run: python scripts/migrate_checkpoints.py <checkpoint_dir>"
            ) from exc
        raise


def load_inner_model(
    model_type: str, ckpt_path, device,
) -> tuple[torch.nn.Module, object]:
    """Load a Lightning checkpoint, return (inner nn.Module on device in eval, hparams cfg)."""
    module = safe_load_checkpoint(model_type, ckpt_path)
    model = module.model
    model.to(device).eval()
    return model, module.hparams


def prepare_kd(
    cfg, model_type: str, device,
) -> tuple[torch.nn.Module | None, torch.nn.Linear | None]:
    """Resolve teacher checkpoint, load + freeze, create projection if needed.

    Returns (teacher, projection) when KD is active, (None, None) otherwise.
    Called by module __init__ or pipeline build_module.
    """
    from pathlib import Path

    if not any(getattr(a, "type", None) == "kd" for a in (getattr(cfg, "auxiliaries", None) or [])):
        return None, None

    kd = next(a for a in getattr(cfg, "auxiliaries", []) if a.type == "kd")
    if getattr(kd, "model_path", None):
        teacher_path = Path(kd.model_path)
    else:
        import copy

        from graphids.config import checkpoint_path
        teacher_scale = getattr(kd, "teacher_scale", "large")
        teacher_cfg = copy.copy(cfg)
        teacher_cfg.scale = teacher_scale
        teacher_path = checkpoint_path(
            cfg.lake_root, cfg.dataset, model_type, teacher_scale, cfg.seed, teacher_cfg,
            gat_stage=getattr(cfg, "gat_stage", "curriculum"),
        )
        if not teacher_path.exists():
            raise FileNotFoundError(
                f"Teacher checkpoint not found: {teacher_path}. "
                f"Train {model_type}/{teacher_scale} first, or set model_path explicitly."
            )

    teacher, tcfg = load_inner_model(model_type, teacher_path, device)
    teacher.requires_grad_(False)

    # Projection layer for VGAE latent space alignment
    projection = None
    if model_type == "vgae":
        s_dim = cfg.latent_dim
        t_dim = tcfg.latent_dim
        if s_dim != t_dim:
            _log.info("projection_layer", student_dim=s_dim, teacher_dim=t_dim)
            projection = torch.nn.Linear(s_dim, t_dim).to(device)

    return teacher, projection
