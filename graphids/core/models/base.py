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
# Init-kwarg storage helper — shared by GraphModuleBase + FusionModuleBase
# ---------------------------------------------------------------------------


def store_init_kwargs(obj: nn.Module, locals_dict: dict) -> None:
    """Mirror every ``__init__`` kwarg from ``locals()`` onto ``obj``.

    Replaces the per-subclass ``self.X = X`` setattr block. Pair with the
    inspect-driven ``hparams`` property which reads the same signature
    back. Call from the subclass ``__init__`` immediately after
    ``super().__init__()``; computed defaults (e.g.
    ``self.id_encoder_kwargs = self.id_encoder_kwargs or {}``) and derived
    attrs (``self.model = None``, ``test_metrics``, ``_build()``) stay
    explicit afterward.
    """
    import inspect

    sig = inspect.signature(type(obj).__init__)
    for name in sig.parameters:
        if name == "self":
            continue
        if name in locals_dict:
            setattr(obj, name, locals_dict[name])


# ---------------------------------------------------------------------------
# torch.compile helper
# ---------------------------------------------------------------------------


def try_compile(model: nn.Module, *, conv_type: str | None = None, **kwargs) -> nn.Module:
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
    # torch.compile is lazy — errors surface at first forward, not here.
    # A broad except at wrap time masked zero real failures and swallowed
    # config bugs. Let exceptions propagate.
    return torch.compile(model, **kwargs)


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

    @property
    def hparams(self) -> SimpleNamespace:
        """Snapshot of the declared ``__init__`` params, read from ``self``.

        Subclasses assign each ``__init__`` argument to ``self`` explicitly
        (``self.conv_type = conv_type`` etc.). This property reads them back
        via ``inspect.signature(type(self).__init__)`` — standard signature
        introspection, not frame inspection. ``nn.Module`` values (e.g. a
        ``loss_fn`` argument) are skipped: they belong in the state_dict,
        not the hparams blob.
        """
        import inspect

        sig = inspect.signature(type(self).__init__)
        ns: dict[str, Any] = {}
        for name in sig.parameters:
            if name == "self":
                continue
            val = getattr(self, name, None)
            if isinstance(val, nn.Module):
                continue
            ns[name] = val
        return SimpleNamespace(**ns)

    def __init__(self) -> None:
        super().__init__()
        self._metric_acc = MetricAccumulator()
        # Non-persistent buffer that tracks device through .to()/.cuda()/.cpu()
        # — robust even for parameter-free modules (HF Transformers pattern).
        self.register_buffer("_device_tracker", torch.empty(0), persistent=False)

    def _store_init_kwargs(self, locals_dict: dict) -> None:
        """See :func:`store_init_kwargs`."""
        store_init_kwargs(self, locals_dict)

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
            # Write to self directly: ``self.hparams`` is a computed
            # property — assigning ``self.hparams.x = v`` would land on a
            # throwaway SimpleNamespace.
            self.num_ids = datamodule.num_ids
            self.in_channels = datamodule.in_channels
            self.num_classes = datamodule.num_classes
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

    def on_validation_epoch_end(self) -> None:
        """Override to flush epoch-level val metrics into _metric_acc before compute."""

    def on_test_epoch_start(self):
        self.test_metrics.reset()
        if hasattr(self, "roc_metric"):
            self.roc_metric.reset()
        names = getattr(self, "_test_set_names", None) or ["test"]
        # One MetricCollection per test-set, keys prefixed for structured logs.
        self._per_set_metrics = {n: self.test_metrics.clone(prefix=f"test/{n}/") for n in names}
        # ``scores`` holds 1-D scores for threshold-flavor (VGAE/DGI) and (N,K)
        # probabilities for classifier-flavor (GAT/fusion). ``preds`` is
        # optional hard predictions when the model emits them directly.
        self._test_buffers = {n: {"preds": [], "scores": [], "labels": []} for n in names}
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
        """Per-set + aggregate metrics from buffered ``(N, K)`` probabilities.

        Buffers on CPU — ``BinaryMCC``'s confusion matrix and friends raise on
        cross-device updates, so the collections get moved to match.
        """
        if not getattr(self, "_per_set_metrics", None):
            return
        self.test_metrics = self.test_metrics.cpu()
        all_probs, all_labels = [], []
        for name, coll in self._per_set_metrics.items():
            buf = self._test_buffers[name]
            if not buf["scores"]:
                continue
            probs = torch.cat(buf["scores"]).float()
            labels = torch.cat(buf["labels"]).long()
            coll = coll.cpu()
            self._per_set_metrics[name] = coll
            coll.update(probs, labels)
            self.log_dict(coll.compute())
            # Operating points are binary-specific — reconstruct class-1 score.
            if probs.ndim == 2 and probs.shape[1] == 2:
                self._log_operating_points(probs[:, 1], labels, prefix=f"test/{name}/")
            all_probs.append(probs)
            all_labels.append(labels)
        if all_probs:
            pooled_p, pooled_l = torch.cat(all_probs), torch.cat(all_labels)
            self.test_metrics.update(pooled_p, pooled_l)
            self.log_dict(self.test_metrics.compute())
            if pooled_p.ndim == 2 and pooled_p.shape[1] == 2:
                self._log_operating_points(pooled_p[:, 1], pooled_l, prefix="test/")

    def _log_thresholded_metrics(self):
        """Threshold flavor — one global threshold, per-set metrics at it."""
        if not self.roc_metric.preds:
            return
        pooled_scores = torch.cat(self.roc_metric.preds).cpu()
        pooled_labels = torch.cat(self.roc_metric.target).cpu().long()
        if self.test_threshold is None:
            thr = float(self.roc_metric.compute())
            self.test_threshold = thr if not math.isnan(thr) else float(pooled_scores.median())

        # Rebuild the aggregate + per-set collections at the discovered
        # threshold so hard-pred metrics binarize at Youden-J while curve
        # metrics (AUROC/AP/ECE) operate on raw scores per their contract.
        self.test_metrics = binary_test_metrics(threshold=self.test_threshold).to(
            pooled_scores.device
        )
        self.test_metrics.update(pooled_scores, pooled_labels)
        metrics = self.test_metrics.compute()
        metrics["threshold"] = self.test_threshold
        self.log_dict(metrics)
        # Operating points are score-based — valid on raw scores pre-threshold.
        self._log_operating_points(pooled_scores, pooled_labels, prefix="test/")

        # Per-set metrics at the same global threshold.
        if getattr(self, "_per_set_metrics", None):
            for name in list(self._per_set_metrics):
                buf = self._test_buffers[name]
                if not buf["scores"]:
                    continue
                scores = torch.cat(buf["scores"])
                labels = torch.cat(buf["labels"]).long()
                coll = binary_test_metrics(threshold=self.test_threshold).to(scores.device)
                coll.update(scores, labels)
                self._per_set_metrics[name] = coll
                # Prefix per-set keys so they don't collide with the aggregate
                # `accuracy`/`f1`/etc. above (MetricAccumulator batch-averages
                # collisions, silently corrupting the aggregate).
                prefix = f"test/{name}/"
                self.log_dict({f"{prefix}{k}": v for k, v in coll.compute().items()})
                self._log_operating_points(scores, labels, prefix=prefix)
                # Materialize derived preds so _finalize_test_predictions persists them.
                buf["preds"] = [(scores >= self.test_threshold).long()]
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
        # torchmetrics returns NaN when the target operating point is
        # unreachable (e.g. max precision < min_precision). That's a valid
        # "no such threshold" sentinel, not a metric value — skip it.
        candidates = {
            f"{prefix}precision@{min_recall:g}recall": float(prec),
            f"{prefix}threshold@{min_recall:g}recall": float(thr_p),
            f"{prefix}recall@{min_precision:g}precision": float(rec),
            f"{prefix}threshold@{min_precision:g}precision": float(thr_r),
        }
        self.log_dict({k: v for k, v in candidates.items() if not math.isnan(v)})

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


def classification_test_metrics(num_classes: int):
    """Unified classifier-flavor test metrics — aggregate, macro, weighted, per-class.

    Contract: ``update(probs, target)`` where ``probs`` is ``(N, K)`` float in the
    simplex and ``target`` is ``(N,)`` long. Decision metrics (accuracy/F1/
    precision/recall/specificity/MCC) argmax internally; curve metrics
    (AUROC/AP/ECE) consume the raw probabilities.

    Per-class decomposition uses ``ClasswiseWrapper`` — its dict return is
    merged flat into ``MetricCollection.compute()``, so ``log_dict(compute())``
    just works. Missing classes return 0 for their F1 (torchmetrics `#1494
    <https://github.com/Lightning-AI/torchmetrics/issues/1494>`_), which pulls
    the macro down; report ``weighted`` alongside ``macro``.
    """
    from torchmetrics import MetricCollection
    from torchmetrics.classification import (
        MulticlassAccuracy,
        MulticlassAUROC,
        MulticlassAveragePrecision,
        MulticlassCalibrationError,
        MulticlassF1Score,
        MulticlassMatthewsCorrCoef,
        MulticlassPrecision,
        MulticlassRecall,
        MulticlassSpecificity,
    )
    from torchmetrics.wrappers import ClasswiseWrapper

    labels = (
        ["benign", "attack"] if num_classes == 2 else [f"class_{i}" for i in range(num_classes)]
    )
    k = {"num_classes": num_classes}

    def cw(metric, prefix):
        return ClasswiseWrapper(metric, labels=labels, prefix=prefix)

    return MetricCollection(
        {
            # Aggregate scalars — no per-class ambiguity.
            "accuracy": MulticlassAccuracy(**k, average="micro"),
            "mcc": MulticlassMatthewsCorrCoef(**k),
            "ece": MulticlassCalibrationError(**k),
            # Macro + weighted averages.
            "f1_macro": MulticlassF1Score(**k, average="macro"),
            "f1_weighted": MulticlassF1Score(**k, average="weighted"),
            "precision_macro": MulticlassPrecision(**k, average="macro"),
            "precision_weighted": MulticlassPrecision(**k, average="weighted"),
            "recall_macro": MulticlassRecall(**k, average="macro"),
            "recall_weighted": MulticlassRecall(**k, average="weighted"),
            "specificity_macro": MulticlassSpecificity(**k, average="macro"),
            "specificity_weighted": MulticlassSpecificity(**k, average="weighted"),
            "auc_macro": MulticlassAUROC(**k, average="macro"),
            "auc_weighted": MulticlassAUROC(**k, average="weighted"),
            "ap_macro": MulticlassAveragePrecision(**k, average="macro", thresholds=None),
            "ap_weighted": MulticlassAveragePrecision(**k, average="weighted", thresholds=None),
            # Per-class — ClasswiseWrapper expands into class-named keys.
            "f1_pc": cw(MulticlassF1Score(**k, average=None), "f1_per_class/"),
            "precision_pc": cw(MulticlassPrecision(**k, average=None), "precision_per_class/"),
            "recall_pc": cw(MulticlassRecall(**k, average=None), "recall_per_class/"),
            "specificity_pc": cw(
                MulticlassSpecificity(**k, average=None), "specificity_per_class/"
            ),
            "auc_pc": cw(MulticlassAUROC(**k, average=None), "auc_per_class/"),
            "ap_pc": cw(
                MulticlassAveragePrecision(**k, average=None, thresholds=None),
                "pr_auc_per_class/",
            ),
        }
    )


def binary_test_metrics(threshold: float = 0.5):
    """Standard binary classification MetricCollection shared by all modules.

    Contract: ``update(preds, target)`` expects ``preds`` to be a **float**
    tensor of probabilities in [0, 1] (or logits) and ``target`` to be
    long/int. Curve metrics (AUROC, AP, ECE) validate this and raise on int
    preds. The hard-pred metrics (accuracy/f1/precision/recall/specificity/
    MCC) apply ``threshold`` internally. Passing already-thresholded int
    preds is wrong: it crashes the curve metrics.

    ``threshold`` is forwarded to every metric that accepts it. For
    threshold-flavor models (VGAE/DGI), rebuild the collection after
    Youden-J discovery; for fusion models, pass the agent's
    ``decision_threshold`` at construction time.

    MCC is the chance-corrected summary for imbalanced binary data (CAN
    intrusion is ~1% positive). AP (area under PR curve) is the correct
    curve metric for imbalanced data. ECE measures probability calibration
    — meaningful only on classifier scores in [0, 1].
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
            "accuracy": BinaryAccuracy(threshold=threshold),
            "f1": BinaryF1Score(threshold=threshold),
            "precision": BinaryPrecision(threshold=threshold),
            "recall": BinaryRecall(threshold=threshold),
            "specificity": BinarySpecificity(threshold=threshold),
            "mcc": BinaryMatthewsCorrCoef(threshold=threshold),
            "auc": BinaryAUROC(),
            "ap": BinaryAveragePrecision(),
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

    from graphids._fs import atomic_load

    ckpt = atomic_load(ckpt_path, map_location=map_location, weights_only=True)
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
