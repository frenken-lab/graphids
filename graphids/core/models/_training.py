"""Lightning training helpers shared across VGAE, GAT, and DGI modules.

KD is no longer a cross-cutting concern on these modules — it lives in
``graphids.core.losses.distillation`` as composable loss wrappers. This
file used to own ``_install_kd_teacher``, ``prepare_kd``, ``KDAuxiliary``
and ``teacher_on_device``; all five were deleted when KD collapsed to
"just a loss function" (Option B refactor).
"""

import contextlib

import pytorch_lightning as pl
import torch

from graphids.log import get_logger

_log = get_logger(__name__)


def try_compile(
    model: torch.nn.Module, *, conv_type: str | None = None, **kwargs
) -> torch.nn.Module:
    """Attempt ``torch.compile``; fall back to eager on inductor failure.

    Skips compile entirely for conv types that use ``to_dense_batch()``
    (e.g. GPS) — the ``Tensor.item()`` call causes graph breaks, repeated
    recompilation, and eventual CUDA illegal memory access.

    The inductor backend can also fail on unusual FX graph patterns (e.g.
    DGI's dual-encoder structure on torch 2.8). Rather than crash the
    entire job, log a warning and continue uncompiled.
    """
    # GPS and other quadratic conv types use to_dense_batch() which calls
    # Tensor.item() → graph break → repeated recompilation → CUDA crash.
    _INCOMPATIBLE_CONV_TYPES = frozenset({"gps"})
    if conv_type in _INCOMPATIBLE_CONV_TYPES:
        _log.warning(
            "compile_skipped",
            conv_type=conv_type,
            reason="to_dense_batch Tensor.item() causes graph breaks and CUDA crash",
        )
        return model
    if not hasattr(torch, "compile"):
        return model
    try:
        return torch.compile(model, **kwargs)
    except Exception:
        _log.warning("torch_compile_failed", model=type(model).__name__, fallback="eager")
        torch._dynamo.reset()
        return model


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

    KD is not a base-class concern anymore — it lives entirely in
    ``graphids.core.losses.distillation`` as composable loss wrappers.
    """

    # -- Lazy model construction ------------------------------------------------

    def setup(self, stage=None):
        if self.model is None:
            dm = self.trainer.datamodule
            if dm is not None:
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
                optimizer,
                T_max=max_epochs,
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

    # -- Shared test-epoch hooks -----------------------------------------------
    # Two patterns:
    #   (A) Simple: subclass sets ``self.test_metrics = binary_test_metrics()`` and
    #       the default hooks reset/compute/log. Used by GAT.
    #   (B) Thresholded: subclass also calls ``_init_threshold_metrics()``, overrides
    #       ``on_test_epoch_end`` to call ``_log_thresholded_metrics()``. Used by VGAE, DGI.

    def on_test_epoch_start(self):
        self.test_metrics.reset()
        if hasattr(self, "roc_metric"):
            self.roc_metric.reset()

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute())

    def _log_thresholded_metrics(self):
        """Derive Youden-J threshold from accumulated ROC, update test_metrics, log."""
        if not self.roc_metric.preds:
            return
        scores = torch.cat(self.roc_metric.preds).cpu()
        labels = torch.cat(self.roc_metric.target).cpu().long()
        if self.test_threshold is None:
            self.test_threshold = self._find_threshold() or float(scores.median())
        preds = (scores >= self.test_threshold).long()
        self.test_metrics.update(preds, labels)
        metrics = self.test_metrics.compute()
        metrics["threshold"] = self.test_threshold
        self.log_dict(metrics)

    def on_save_checkpoint(self, checkpoint):
        if getattr(self, "test_threshold", None) is not None:
            checkpoint["test_threshold"] = self.test_threshold

    def on_load_checkpoint(self, checkpoint):
        if "test_threshold" in checkpoint:
            self.test_threshold = checkpoint["test_threshold"]


@contextlib.contextmanager
def eval_mode(model):
    """Context manager: set model.eval(), restore original training state on exit.

    Enforces the critical constraint: never leak eval mode to callers.
    """
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(was_training)


def binary_test_metrics():
    """Standard binary classification MetricCollection shared by all Lightning modules."""
    from torchmetrics import MetricCollection
    from torchmetrics.classification import (
        BinaryAccuracy,
        BinaryAUROC,
        BinaryF1Score,
        BinaryPrecision,
        BinaryRecall,
        BinarySpecificity,
    )

    return MetricCollection(
        {
            "accuracy": BinaryAccuracy(),
            "f1": BinaryF1Score(),
            "precision": BinaryPrecision(),
            "recall": BinaryRecall(),
            "specificity": BinarySpecificity(),
            "auc": BinaryAUROC(),
        }
    )


# ---------------------------------------------------------------------------
# Model loading — used by the dev-path train CLI, fusion stages, and
# ``graphids.instantiate._build_loss`` when it needs to load a KD teacher
# checkpoint into a ``SoftLabelDistillation`` / ``FeatureDistillation``.
# ---------------------------------------------------------------------------


_MODULE_PATHS: dict[str, str] = {
    "vgae": "graphids.core.models.autoencoder.vgae.VGAEModule",
    "gat": "graphids.core.models.supervised.gat.GATModule",
    "dgi": "graphids.core.models.autoencoder.dgi.DGIModule",
    "fusion": "graphids.core.models.fusion.bandit.BanditFusionModule",
    "dqn": "graphids.core.models.fusion.dqn.DQNFusionModule",
}


def safe_load_checkpoint(model_type: str, ckpt_path, *, map_location="cpu"):
    """Load a Lightning checkpoint by model type, raising on missing files."""
    import importlib
    from pathlib import Path

    dotted = _MODULE_PATHS.get(model_type)
    if dotted is None:
        raise KeyError(f"No module class for '{model_type}'. Available: {list(_MODULE_PATHS)}")
    module_path, cls_name = dotted.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_path), cls_name)

    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    return cls.load_from_checkpoint(
        str(ckpt_path),
        map_location=map_location,
        weights_only=True,
    )


def load_inner_model(
    model_type: str,
    ckpt_path,
    device,
) -> tuple[torch.nn.Module, object]:
    """Load a Lightning checkpoint, return (inner nn.Module on device in eval, hparams cfg)."""
    module = safe_load_checkpoint(model_type, ckpt_path)
    model = module.model
    model.to(device).eval()
    return model, module.hparams
