"""Shared model infrastructure — base classes, utilities, contracts.

Graph family:
- ``GraphModuleBase`` — base for VGAE, GAT, DGI
- ``try_compile`` — safe torch.compile with conv-type gating
- ``eval_mode`` — context manager that restores training state

Shared:
- ``binary_test_metrics`` — standard MetricCollection for all families
- ``safe_load_checkpoint`` / ``load_inner_model`` — checkpoint loading registry
- ``LAYOUT`` / ``STATE_DIM`` / ``FeatureLayout`` — fusion state vector contract
- ``schema_for`` auto-generated Pydantic model configs
"""

from __future__ import annotations

import contextlib
import importlib
from pathlib import Path
from typing import NamedTuple

import pytorch_lightning as pl
import torch

from graphids._otel import get_logger

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# torch.compile helper
# ---------------------------------------------------------------------------


def try_compile(
    model: torch.nn.Module, *, conv_type: str | None = None, **kwargs
) -> torch.nn.Module:
    """Attempt ``torch.compile``; fall back to eager on inductor failure.

    Skips compile entirely for conv types that use ``to_dense_batch()``
    (e.g. GPS) — the ``Tensor.item()`` call causes graph breaks, repeated
    recompilation, and eventual CUDA illegal memory access.
    """
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
# Graph model base class (VGAE, GAT, DGI)
# ---------------------------------------------------------------------------


class GraphModuleBase(pl.LightningModule):
    """Shared base for VGAE, GAT, DGI — lazy setup, OOM guard, threshold metrics.

    Subclasses must implement ``_build()`` which constructs ``self.model`` and any
    other architecture components using ``self.hparams`` (populated by ``setup``).
    """

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

    def configure_optimizers(self):
        lr = getattr(self.hparams, "lr", 1e-3)
        wd = getattr(self.hparams, "weight_decay", 0.0)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)
        max_epochs = getattr(self.trainer, "max_epochs", None)
        if max_epochs and max_epochs > 1:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        return optimizer

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

    def _init_threshold_metrics(self):
        """Call in ``__init__`` for modules that need Youden-J threshold."""
        from torchmetrics.classification import BinaryROC

        self.roc_metric = BinaryROC()
        self.test_threshold: float | None = None

    def _find_threshold(self) -> float | None:
        fpr, tpr, thresholds = self.roc_metric.compute()
        if thresholds.numel() < 2:
            return None
        j = tpr - fpr
        best = torch.argmax(j)
        return float(thresholds[best]) if best < len(thresholds) else None

    def on_test_epoch_start(self):
        self.test_metrics.reset()
        if hasattr(self, "roc_metric"):
            self.roc_metric.reset()

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute())

    def _log_thresholded_metrics(self):
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


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def eval_mode(model):
    """Context manager: set model.eval(), restore original training state on exit."""
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
# Checkpoint loading registry
# ---------------------------------------------------------------------------

_MODULE_PATHS: dict[str, str] = {
    "vgae": "graphids.core.models.autoencoder.vgae_module.VGAEModule",
    "gat": "graphids.core.models.supervised.gat_module.GATModule",
    "dgi": "graphids.core.models.autoencoder.dgi_module.DGIModule",
    "fusion": "graphids.core.models.fusion.bandit.BanditFusionModule",
    "dqn": "graphids.core.models.fusion.dqn.DQNFusionModule",
}


def safe_load_checkpoint(model_type: str, ckpt_path, *, map_location="cpu"):
    """Load a Lightning checkpoint by model type, raising on missing files."""
    dotted = _MODULE_PATHS.get(model_type)
    if dotted is None:
        raise KeyError(f"No module class for '{model_type}'. Available: {list(_MODULE_PATHS)}")
    module_path, cls_name = dotted.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_path), cls_name)

    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # loss_fn is excluded from save_hyperparameters (it's an nn.Module) so
    # load_from_checkpoint can't reconstruct it. Rebuild from saved hparams.
    extra_kwargs: dict = {}
    if model_type in {"vgae", "gat"}:
        from graphids.core.losses.build import _VGAE_LOSS_KEYS, build_loss

        ckpt = torch.load(str(ckpt_path), map_location=map_location, weights_only=True)
        hp = ckpt.get("hyper_parameters", {})
        if model_type == "vgae":
            loss_cfg = {k: hp[k] for k in _VGAE_LOSS_KEYS if k in hp}
        else:
            loss_cfg = hp.get("loss_config")
        extra_kwargs["loss_fn"] = build_loss(model_type, loss_cfg, distillation_config=None)

    return cls.load_from_checkpoint(
        str(ckpt_path), map_location=map_location, weights_only=True, **extra_kwargs
    )


def load_inner_model(model_type: str, ckpt_path, device) -> tuple[torch.nn.Module, object]:
    """Load a Lightning checkpoint, return (inner nn.Module on device in eval, hparams cfg)."""
    module = safe_load_checkpoint(model_type, ckpt_path)
    model = module.model
    model.to(device).eval()
    return model, module.hparams


# ---------------------------------------------------------------------------
# Fusion state vector contract
# ---------------------------------------------------------------------------


class FeatureLayout(NamedTuple):
    """Offset, dim, and confidence index of one extractor inside the state vector."""

    offset: int
    dim: int
    confidence_idx: int


# Canonical ordering — must match extractor registry in core/data/fusion_states.py.
_EXTRACTOR_DIMS = [
    ("vgae", 8, 7),  # (name, feature_dim, confidence_index_within_block)
    ("gat", 7, 6),
]


def _build_layout() -> tuple[dict[str, FeatureLayout], int]:
    layout: dict[str, FeatureLayout] = {}
    offset = 0
    for name, dim, conf_idx in _EXTRACTOR_DIMS:
        layout[name] = FeatureLayout(offset, dim, offset + conf_idx)
        offset += dim
    return layout, offset


LAYOUT, STATE_DIM = _build_layout()


# ---------------------------------------------------------------------------
# Auto-generated Pydantic model schemas
# ---------------------------------------------------------------------------


def _build_schemas():
    """Lazy-build so model imports don't fire at package import time."""
    from graphids.core._schema_gen import schema_for
    from graphids.core.models.autoencoder.dgi_module import DGIModule
    from graphids.core.models.autoencoder.vgae_module import VGAEModule
    from graphids.core.models.fusion.bandit import BanditFusionModule
    from graphids.core.models.fusion.dqn import DQNFusionModule
    from graphids.core.models.fusion.mlp import MLPFusionModule
    from graphids.core.models.fusion.weighted_avg import WeightedAvgModule
    from graphids.core.models.supervised.gat_module import GATModule

    return {
        "VGAEConfig": schema_for(VGAEModule),
        "DGIConfig": schema_for(DGIModule),
        "GATConfig": schema_for(GATModule),
        "BanditConfig": schema_for(BanditFusionModule),
        "DQNConfig": schema_for(DQNFusionModule),
        "MLPFusionConfig": schema_for(MLPFusionModule),
        "WeightedAvgConfig": schema_for(WeightedAvgModule),
    }
