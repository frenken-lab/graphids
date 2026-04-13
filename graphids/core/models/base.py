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
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NamedTuple

import torch
import torch.nn as nn
from torchmetrics import Metric
from torchmetrics.functional.classification import binary_roc
from torchmetrics.utilities.data import dim_zero_cat

from graphids._otel import get_logger
from graphids.core.trainer import MetricAccumulator

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
# BinaryYoudenJThreshold — custom Metric
# ---------------------------------------------------------------------------


class BinaryYoudenJThreshold(Metric):
    """Buffer pooled scores/labels; compute() returns the Youden-J threshold.

    Replaces the prior ``BinaryROC() + _find_threshold()`` pair. The
    ``.preds`` / ``.target`` list states are read directly by
    ``_log_thresholded_metrics`` (same attribute names as the old
    ``BinaryROC`` to preserve that access pattern).
    """

    full_state_update = False

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        self.preds.append(preds)
        self.target.append(target)

    def compute(self) -> torch.Tensor:
        p = dim_zero_cat(self.preds)
        t = dim_zero_cat(self.target).long()
        fpr, tpr, thr = binary_roc(p, t)
        if thr.numel() < 2:
            return torch.tensor(float("nan"), device=thr.device)
        return thr[torch.argmax(tpr - fpr)]


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
        if datamodule is not None:
            # Capture test-set names for per-loader metric breakdowns (issue #26).
            # Falls back to a single "test" bucket for datamodules that don't
            # expose a dict of named test datasets.
            ds = getattr(datamodule, "test_datasets", None)
            self._test_set_names = list(ds.keys()) if ds else ["test"]

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
        """Call in ``__init__`` for modules that need a Youden-J threshold."""
        self.roc_metric = BinaryYoudenJThreshold()
        self.test_threshold: float | None = None

    # -- per-test-set evaluation (issue #26) ---------------------------------

    def on_test_epoch_start(self):
        self.test_metrics.reset()
        if hasattr(self, "roc_metric"):
            self.roc_metric.reset()
        names = getattr(self, "_test_set_names", None) or ["test"]
        # One MetricCollection per test-set, keys prefixed for structured logs.
        self._per_set_metrics = {
            n: self.test_metrics.clone(prefix=f"test/{n}/") for n in names
        }
        # Per-set buffers — scores/labels always; preds when the model emits
        # them directly (classifier flavor) or after thresholding (recon).
        self._test_buffers = {
            n: {"preds": [], "scores": [], "labels": []} for n in names
        }
        # Filled in on_test_epoch_end; stage.evaluate reads to persist to disk.
        self._test_predictions: dict[str, dict[str, torch.Tensor]] = {}

    def _record_test_batch(self, dataloader_idx: int, *, scores, labels, preds=None) -> None:
        """Buffer one batch's predictions under the right test-set bucket."""
        names = getattr(self, "_test_set_names", ["test"])
        name = names[dataloader_idx] if dataloader_idx < len(names) else names[-1]
        buf = self._test_buffers[name]
        buf["scores"].append(scores.detach().cpu())
        buf["labels"].append(labels.detach().cpu())
        if preds is not None:
            buf["preds"].append(preds.detach().cpu())

    def on_test_epoch_end(self):
        """Classifier flavor — compute per-set + aggregate metrics from buffers.

        Models that override with `_log_thresholded_metrics` (VGAE, DGI) take
        the threshold path instead.
        """
        self._log_classifier_metrics()
        self._finalize_test_predictions()

    def _log_classifier_metrics(self) -> None:
        """Per-set + aggregate metrics, assuming test_metrics takes raw scores."""
        if not getattr(self, "_per_set_metrics", None):
            return
        all_scores, all_labels = [], []
        for name, coll in self._per_set_metrics.items():
            buf = self._test_buffers[name]
            if not buf["scores"]:
                continue
            scores = torch.cat(buf["scores"])
            labels = torch.cat(buf["labels"]).long()
            coll.update(scores, labels)
            self.log_dict(coll.compute())
            self._log_operating_points(scores, labels, prefix=f"test/{name}/")
            all_scores.append(scores)
            all_labels.append(labels)
        if all_scores:
            pooled_s, pooled_l = torch.cat(all_scores), torch.cat(all_labels)
            self.test_metrics.update(pooled_s, pooled_l)
            self.log_dict(self.test_metrics.compute())
            self._log_operating_points(pooled_s, pooled_l, prefix="test/")

    def _log_thresholded_metrics(self):
        """Threshold flavor — one global threshold, per-set metrics at it."""
        if not self.roc_metric.preds:
            return
        pooled_scores = torch.cat(self.roc_metric.preds).cpu()
        pooled_labels = torch.cat(self.roc_metric.target).cpu().long()
        if self.test_threshold is None:
            thr = float(self.roc_metric.compute())
            self.test_threshold = thr if not math.isnan(thr) else float(pooled_scores.median())

        # Aggregate (preserves existing semantics).
        agg_preds = (pooled_scores >= self.test_threshold).long()
        self.test_metrics.update(agg_preds, pooled_labels)
        metrics = self.test_metrics.compute()
        metrics["threshold"] = self.test_threshold
        self.log_dict(metrics)
        # Operating points are score-based — valid on raw scores pre-threshold.
        self._log_operating_points(pooled_scores, pooled_labels, prefix="test/")

        # Per-set metrics at the same global threshold.
        if getattr(self, "_per_set_metrics", None):
            for name, coll in self._per_set_metrics.items():
                buf = self._test_buffers[name]
                if not buf["scores"]:
                    continue
                scores = torch.cat(buf["scores"])
                labels = torch.cat(buf["labels"]).long()
                preds = (scores >= self.test_threshold).long()
                coll.update(preds, labels)
                self.log_dict(coll.compute())
                self._log_operating_points(scores, labels, prefix=f"test/{name}/")
                # Materialize derived preds so _finalize_test_predictions persists them.
                buf["preds"] = [preds]
        self._finalize_test_predictions()

    def _log_operating_points(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        *,
        prefix: str = "",
        min_recall: float = 0.95,
        min_precision: float = 0.99,
    ) -> None:
        """Precision@recall and recall@precision — canonical IDS operating points.

        Skipped when labels are single-class (no positives or no negatives);
        the functional metrics raise in that regime.
        """
        if labels.unique().numel() < 2:
            return
        from torchmetrics.functional.classification import (
            binary_precision_at_fixed_recall,
            binary_recall_at_fixed_precision,
        )
        prec, thr_p = binary_precision_at_fixed_recall(scores, labels, min_recall=min_recall)
        rec, thr_r = binary_recall_at_fixed_precision(scores, labels, min_precision=min_precision)
        self.log_dict({
            f"{prefix}precision@{min_recall:g}recall": float(prec),
            f"{prefix}threshold@{min_recall:g}recall": float(thr_p),
            f"{prefix}recall@{min_precision:g}precision": float(rec),
            f"{prefix}threshold@{min_precision:g}precision": float(thr_r),
        })

    def _finalize_test_predictions(self) -> None:
        """Concatenate per-set buffers into a {name: {key: Tensor}} dict."""
        if not getattr(self, "_test_buffers", None):
            return
        self._test_predictions = {
            name: {
                k: torch.cat(v) if v else torch.empty(0)
                for k, v in buf.items()
                if v  # skip empty keys (e.g. preds on score-only models)
            }
            for name, buf in self._test_buffers.items()
            if buf["scores"]  # only sets we actually saw batches for
        }

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
    """Standard binary classification MetricCollection shared by all modules.

    MCC is the chance-corrected summary for imbalanced binary data (CAN
    intrusion is ~1% positive — F1 and accuracy can both look good while
    MCC exposes a dominant-class classifier). AP (area under PR curve) is
    the correct curve metric for imbalanced data. ECE measures probability
    calibration — only meaningful on classifier scores in [0, 1]; on
    threshold-flavor models (VGAE/DGI) it degenerates same as AUROC does,
    tracked as a pre-existing known issue.
    """
    from torchmetrics import MetricCollection
    from torchmetrics.classification import (
        BinaryAccuracy,
        BinaryAUROC,
        BinaryAveragePrecision,
        BinaryCalibrationError,
        BinaryF1Score,
        BinaryMatthewsCorrCoef,
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
            "ap": BinaryAveragePrecision(),
            "mcc": BinaryMatthewsCorrCoef(),
            "ece": BinaryCalibrationError(),
        }
    )


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def safe_load_checkpoint(model_type: str, ckpt_path, *, map_location="cpu"):
    """Load a checkpoint, dispatching on the ``class_path`` saved at write time.

    ``model_type`` is used only to know which loss_fn to rebuild for VGAE/GAT
    (loss is excluded from hyperparameters). Class lookup uses the
    self-describing ``class_path`` written by ``core.callbacks._build_checkpoint``.
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location=map_location, weights_only=True)
    dotted = ckpt.get("class_path")
    if not dotted:
        raise KeyError(
            f"Checkpoint {ckpt_path} missing 'class_path'. Re-train with the "
            "current callbacks.ModelCheckpoint to produce self-describing checkpoints."
        )
    module_path, cls_name = dotted.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_path), cls_name)

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
