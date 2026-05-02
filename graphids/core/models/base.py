"""Shared model infrastructure — base classes, utilities, contracts.

Graph family:
- ``GraphModuleBase`` — base for VGAE, GAT, DGI
- ``try_compile`` — safe torch.compile with conv-type gating
- ``eval_mode`` — context manager that restores training state

Shared:
- ``_ModelBase`` — mixin shared by ``GraphModuleBase`` + ``FusionModuleBase``
- ``safe_load_checkpoint`` — checkpoint loading via class_path registry
- ``schema_for`` auto-generated Pydantic model configs
"""

from __future__ import annotations

import contextlib
import importlib
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
from structlog import get_logger

from graphids.core.trainer import MetricAccumulator

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# _ModelBase — shared mixin for every model module
# ---------------------------------------------------------------------------


class _ModelBase(nn.Module):
    """Shared infra for every model module:

    - ``MetricAccumulator`` — buffers per-step ``log()`` values for the trainer
      to flush at epoch boundaries (PL-style ``self.log("name", value)`` API).
    - ``_device_tracker`` — non-persistent buffer that tracks device through
      ``.to()``/``.cuda()``/``.cpu()`` even for parameter-free modules.
    - ``_store_init_kwargs(locals())`` — mirrors every declared ``__init__``
      kwarg onto ``self`` and caches the param-name list for cheap ``hparams``.
    - ``hparams`` — ``SimpleNamespace`` snapshot of init kwargs (skips
      ``nn.Module`` values; those belong in ``state_dict``).
    - ``log()`` / ``log_dict()`` — push to ``MetricAccumulator``.

    Subclasses call ``self._store_init_kwargs(locals())`` from their own
    ``__init__`` immediately after ``super().__init__()``.
    """

    def __init__(self) -> None:
        super().__init__()
        self._metric_acc = MetricAccumulator()
        # Non-persistent buffer that tracks device through .to()/.cuda()/.cpu()
        # — robust even for parameter-free modules (HF Transformers pattern).
        self.register_buffer("_device_tracker", torch.empty(0), persistent=False)
        self._hparam_names: tuple[str, ...] = ()

    def _store_init_kwargs(self, locals_dict: dict) -> None:
        import inspect

        sig = inspect.signature(type(self).__init__)
        names = tuple(n for n in sig.parameters if n != "self")
        for n in names:
            if n in locals_dict:
                setattr(self, n, locals_dict[n])
        self._hparam_names = names

    @property
    def hparams(self) -> SimpleNamespace:
        ns: dict[str, Any] = {}
        for name in self._hparam_names:
            val = getattr(self, name, None)
            if isinstance(val, nn.Module):
                continue
            ns[name] = val
        return SimpleNamespace(**ns)

    @property
    def device(self) -> torch.device:
        return self._device_tracker.device

    def log(self, name: str, value: Any, *, batch_size: int = 1, **_kwargs) -> None:
        """Store a metric for the trainer to read after this step."""
        v = float(value.detach()) if isinstance(value, torch.Tensor) else float(value)
        self._metric_acc.update(name, v, batch_size)

    def log_dict(self, metrics: dict[str, Any], **kwargs) -> None:
        for k, v in metrics.items():
            self.log(k, v, **kwargs)

    # -- per-test-set evaluation (issue #26) ---------------------------------
    #
    # Generic plumbing shared by every model family. Subclasses implement
    # ``test_step`` to call ``_record_test_batch(dataloader_idx, scores=...,
    # labels=..., preds=...)``; the default ``on_test_epoch_*`` lifecycle
    # buckets predictions per test loader, computes per-set + aggregate
    # ``self.test_metrics`` from the buffered ``(N, K)`` probabilities, and
    # exposes the concatenated buffers as ``self._test_predictions`` for
    # ``stage.evaluate`` to persist.

    def setup(self, datamodule=None) -> None:
        """Capture per-test-set names from the datamodule.

        Falls back to a single ``"test"`` bucket for datamodules that don't
        expose a dict of named test datasets. Subclasses that override
        ``setup`` (e.g. ``GraphModuleBase`` for lazy ``_build()``) should
        call ``super().setup(datamodule)``.
        """
        if datamodule is not None:
            ds = getattr(datamodule, "test_datasets", None)
            self._test_set_names = list(ds.keys()) if ds else ["test"]

    def on_test_setup(self, datamodule, device) -> None:
        """Fired after ``_prep`` (model on device, ckpt loaded) and before
        ``model.eval()`` + the test loop. Default no-op. Score-based
        detectors (VGAE/DGI) override to fit calibration buffers from a
        fit-phase loader."""

    def on_test_epoch_start(self) -> None:
        if hasattr(self, "test_metrics"):
            self.test_metrics.reset()
        if hasattr(self, "roc_metric"):
            self.roc_metric.reset()
        names = getattr(self, "_test_set_names", None) or ["test"]
        if hasattr(self, "test_metrics"):
            self._per_set_metrics = {
                n: self.test_metrics.clone(prefix=f"test/{n}/") for n in names
            }
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

    def on_test_epoch_end(self) -> None:
        """Classifier flavor — per-set + aggregate metrics from buffers.

        Models that override (VGAE, DGI) take the threshold path via
        ``_log_thresholded_metrics`` instead.
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
# Graph model base class (VGAE, GAT, DGI)
# ---------------------------------------------------------------------------


class GraphModuleBase(_ModelBase):
    """Shared base for VGAE, GAT, DGI — lazy setup, OOM guard, threshold metrics.

    Subclasses must implement ``_build()`` which constructs ``self.model`` and any
    other architecture components using ``self.hparams`` (populated by ``setup``).
    """

    automatic_optimization = True

    # -- setup + optimizers --------------------------------------------------

    def setup(self, datamodule=None):
        if datamodule is not None:
            already_built = getattr(self, "_built", False) or (
                getattr(self, "model", "_sentinel") not in (None, "_sentinel")
            )
            if not already_built:
                # Write to self directly: ``self.hparams`` is a computed
                # property — assigning ``self.hparams.x = v`` would land on a
                # throwaway SimpleNamespace.
                self.num_ids = datamodule.num_ids
                self.in_channels = datamodule.in_channels
                self.num_classes = datamodule.num_classes
                self._build()
                self._built = True
        super().setup(datamodule)

    def _build(self):
        raise NotImplementedError

    def _init_post(self, locals_dict: dict) -> None:
        """Default ``__init__`` tail for collapsed-arch subclasses.

        Mirrors declared kwargs onto ``self`` (via ``_store_init_kwargs``),
        normalizes ``id_encoder_kwargs`` (None → {}), and lazy-builds when
        ``num_ids`` is already known (e.g. tests instantiate without a
        datamodule). Sets ``self._built`` so ``setup()`` doesn't re-build.
        """
        self._store_init_kwargs(locals_dict)
        if hasattr(self, "id_encoder_kwargs"):
            self.id_encoder_kwargs = self.id_encoder_kwargs or {}
        self._built = False
        if int(getattr(self, "num_ids", 0)) > 0:
            self._build()
            self._built = True

    def _build_id_encoder(self, *, num_ids_offset: int = 0):
        """Construct the ID encoder from ``self.hparams`` settings.

        ``num_ids_offset`` adds reserved vocab slots (e.g. VGAE's mask_id).
        """
        from .id_encoding import build_encoder

        hp = self.hparams
        return build_encoder(
            hp.id_encoder_class_path,
            hp.num_ids + num_ids_offset,
            hp.embedding_dim,
            **(getattr(hp, "id_encoder_kwargs", None) or {}),
        )

    def build_optimizers(self, max_epochs: int) -> tuple[torch.optim.Optimizer | None, Any]:
        """Return ``(optimizer, scheduler_or_None)``. Called by Trainer."""
        lr = getattr(self.hparams, "lr", 1e-3)
        wd = getattr(self.hparams, "weight_decay", 0.0)
        opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)
        return opt, None

    def _init_threshold_metrics(self):
        """Call in ``__init__`` for modules that need a Youden-J threshold."""
        from ._metrics import BinaryYoudenJThreshold

        self.roc_metric = BinaryYoudenJThreshold()
        self.test_threshold: float | None = None

    # -- threshold-flavor test path (VGAE/DGI) -------------------------------

    def on_validation_epoch_end(self) -> None:
        """Override to flush epoch-level val metrics into _metric_acc before compute."""

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


# Re-exported from ._metrics so existing `from ..base import binary_test_metrics`
# imports keep working without dragging the factories' torchmetrics deps into
# this module's import time.
from ._metrics import (  # noqa: E402, F401
    binary_test_metrics,
    classification_test_metrics,
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
    # Legacy class_path remap: the *_module.py wrappers were collapsed into
    # the arch class file (Phase 1+2). Old ckpts saved the wrapper path; new
    # code only ships the collapsed class.
    _LEGACY_CLASS_PATHS = {
        "graphids.core.models.autoencoder.vgae_module.VGAEModule": "graphids.core.models.autoencoder.vgae.VGAE",
        "graphids.core.models.autoencoder.dgi_module.DGIModule": "graphids.core.models.autoencoder.dgi.DGI",
        "graphids.core.models.supervised.gat_module.GATModule": "graphids.core.models.supervised.gat.GAT",
    }
    dotted = _LEGACY_CLASS_PATHS.get(dotted, dotted)
    module_path, cls_name = dotted.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_path), cls_name)

    hp = ckpt.get("hyper_parameters", {})

    # Per-class hook for rebuilding excluded init kwargs (e.g. ``loss_fn``,
    # which can't be pickled into ``hyper_parameters``). Each class that needs
    # something rebuilt declares ``_rebuild_excluded_kwargs(hp) -> dict`` as a
    # classmethod or staticmethod. Default: nothing extra.
    rebuild = getattr(cls, "_rebuild_excluded_kwargs", None)
    extra_kwargs: dict = rebuild(hp) if rebuild is not None else {}

    # Reconstruct the module with saved hyperparameters
    init_kwargs = {**hp, **extra_kwargs}
    module = cls(**init_kwargs)
    state_dict = ckpt["state_dict"]
    # Old wrapper ckpts prefixed every key with ``model.`` (the
    # ``self.model = nn.Module(...)`` indirection collapsed away). Strip when
    # the loaded class declares no top-level ``model`` attribute — there's
    # no key collision because the new layer names don't start with ``model.``.
    if not hasattr(module, "model") and any(k.startswith("model.") for k in state_dict):
        state_dict = {k.removeprefix("model."): v for k, v in state_dict.items()}
    # VGAE: ``mask_token`` was a top-level (frozen) Parameter; it's now the
    # buffer ``masker.mask_token`` on a RandomNodeMasker submodule. Remap
    # legacy keys so old ckpts load cleanly. ``mask_id`` was a plain int and
    # was never in state_dict — no remap needed.
    if "mask_token" in state_dict and "masker.mask_token" not in state_dict:
        state_dict["masker.mask_token"] = state_dict.pop("mask_token")
    module.load_state_dict(state_dict)

    if hasattr(module, "on_load_checkpoint"):
        module.on_load_checkpoint(ckpt)

    return module


