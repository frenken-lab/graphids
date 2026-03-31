"""Lightning training helpers shared across VGAE, GAT, and DGI modules."""

import contextlib
from typing import TypedDict

import pytorch_lightning as pl
import structlog
import torch

_log = structlog.get_logger()


# ---------------------------------------------------------------------------
# GraphModuleBase — shared base for VGAE, GAT, DGI Lightning modules
# ---------------------------------------------------------------------------


class GraphModuleBase(pl.LightningModule):
    """Shared base for VGAE, GAT, DGI — lazy setup, OOM guard, threshold metrics.

    Subclasses must implement ``_build()`` which constructs ``self.model`` and any
    other architecture components using ``self.hparams`` (populated by ``setup``).

    Threshold support (VGAE, DGI): call ``_init_threshold_metrics()`` in your
    ``__init__`` to enable ``BinaryROC`` accumulation and ``_find_threshold()``.
    GAT (supervised) does not need this.
    """

    # -- Lazy model construction ------------------------------------------------

    def setup(self, stage=None):
        if self.model is None:
            dm = self.trainer.datamodule
            self.hparams.num_ids = dm.num_ids
            self.hparams.in_channels = dm.in_channels
            self.hparams.num_classes = dm.num_classes
            self._build()

    def _build(self):
        raise NotImplementedError

    # -- Optimizer ----------------------------------------------------------------

    def configure_optimizers(self):
        """Adam + CosineAnnealingLR. Reads lr/weight_decay from hparams,
        T_max from trainer.max_epochs."""
        lr = getattr(self.hparams, "lr", 1e-3)
        wd = getattr(self.hparams, "weight_decay", 0.0)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)
        max_epochs = getattr(self.trainer, "max_epochs", None)
        if max_epochs and max_epochs > 1:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max_epochs,
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        return optimizer

    # -- OOM guard --------------------------------------------------------------

    def _oom_safe_step(self, batch, batch_idx, step_fn):
        try:
            return step_fn(batch, batch_idx)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            _log.warning(
                "oom_batch_skipped",
                batch_idx=batch_idx,
                num_graphs=batch.num_graphs,
                num_nodes=batch.num_nodes,
            )
            return None

    # -- BinaryROC threshold support (VGAE, DGI) -------------------------------

    def _init_threshold_metrics(self):
        """Call in ``__init__`` for modules that need Youden-J threshold."""
        from torchmetrics.classification import BinaryROC

        self.roc_metric = BinaryROC()
        self.test_threshold: float | None = None

    def _find_threshold(self) -> float | None:
        """Compute optimal threshold via Youden's J statistic from accumulated ROC data."""
        fpr, tpr, thresholds = self.roc_metric.compute()
        if thresholds.numel() < 2:
            return None
        j = tpr - fpr
        best = torch.argmax(j)
        return float(thresholds[best]) if best < len(thresholds) else None


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
