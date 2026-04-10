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
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, NamedTuple

import torch
import torch.nn as nn

from graphids._otel import get_logger
from graphids.core.trainer import MetricAccumulator

if TYPE_CHECKING:
    from graphids.core.trainer import Trainer

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# torch.compile helper
# ---------------------------------------------------------------------------


def try_compile(
    model: nn.Module, *, conv_type: str | None = None, **kwargs
) -> nn.Module:
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


class GraphModuleBase(nn.Module):
    """Shared base for VGAE, GAT, DGI — lazy setup, OOM guard, threshold metrics.

    Subclasses must implement ``_build()`` which constructs ``self.model`` and any
    other architecture components using ``self.hparams`` (populated by ``setup``).
    """

    automatic_optimization = True

    @staticmethod
    def _capture_hparams(local_vars: dict[str, Any], ignore: tuple[str, ...] = ()) -> SimpleNamespace:
        """Capture ``__init__`` kwargs as a ``SimpleNamespace`` for ``self.hparams``."""
        skip = {"self", "__class__", *ignore}
        return SimpleNamespace(**{k: v for k, v in local_vars.items() if k not in skip})

    def __init__(self) -> None:
        super().__init__()
        self._metric_acc = MetricAccumulator()
        self._trainer: Trainer | None = None
        # Non-persistent buffer that tracks device through .to()/.cuda()/.cpu()
        # — robust even for parameter-free modules (HF Transformers pattern).
        self.register_buffer("_device_tracker", torch.empty(0), persistent=False)

    @property
    def device(self) -> torch.device:
        return self._device_tracker.device

    # -- logging (replaces pl self.log / self.log_dict) ----------------------

    def log(self, name: str, value: Any, *, batch_size: int = 1, **_kwargs) -> None:
        """Store a metric for the trainer to read after this step."""
        v = float(value.detach()) if isinstance(value, torch.Tensor) else float(value)
        self._metric_acc.update(name, v, batch_size)

    def log_dict(self, metrics: dict[str, Any], **kwargs) -> None:
        for k, v in metrics.items():
            self.log(k, v, **kwargs)

    # -- setup + optimizers --------------------------------------------------

    def setup(self, datamodule=None):
        if self.model is None and datamodule is not None:
            self.hparams.num_ids = datamodule.num_ids
            self.hparams.in_channels = datamodule.in_channels
            self.hparams.num_classes = datamodule.num_classes
            self._build()

    def _build(self):
        raise NotImplementedError

    def build_optimizers(self, max_epochs: int) -> tuple[torch.optim.Optimizer | None, Any]:
        """Return ``(optimizer, scheduler_or_None)``. Called by Trainer."""
        lr = getattr(self.hparams, "lr", 1e-3)
        wd = getattr(self.hparams, "weight_decay", 0.0)
        opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)
        if max_epochs > 1:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
            return opt, scheduler
        return opt, None

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
    """Standard binary classification MetricCollection shared by all modules."""
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
    """Load a checkpoint by model type, raising on missing files.

    Supports both new raw-PyTorch format (``state_dict`` key) and legacy
    Lightning format (``hyper_parameters`` + wrapped state dict).
    """
    dotted = _MODULE_PATHS.get(model_type)
    if dotted is None:
        raise KeyError(f"No module class for '{model_type}'. Available: {list(_MODULE_PATHS)}")
    module_path, cls_name = dotted.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_path), cls_name)

    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location=map_location, weights_only=True)
    hp = ckpt.get("hyper_parameters", {})

    # Rebuild loss_fn for models that exclude it from hyperparameters
    extra_kwargs: dict = {}
    if model_type in {"vgae", "gat"}:
        from graphids.core.losses.build import _VGAE_LOSS_KEYS, build_loss

        if model_type == "vgae":
            loss_cfg = {k: hp[k] for k in _VGAE_LOSS_KEYS if k in hp}
        else:
            loss_cfg = hp.get("loss_config")
        extra_kwargs["loss_fn"] = build_loss(model_type, loss_cfg, distillation_config=None)

    # Reconstruct the module with saved hyperparameters
    init_kwargs = {**hp, **extra_kwargs}
    module = cls(**init_kwargs)
    module.load_state_dict(ckpt["state_dict"])

    if hasattr(module, "on_load_checkpoint"):
        module.on_load_checkpoint(ckpt)

    return module


def load_inner_model(model_type: str, ckpt_path, device) -> tuple[nn.Module, object]:
    """Load checkpoint, return (inner nn.Module on device in eval, hparams cfg)."""
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
