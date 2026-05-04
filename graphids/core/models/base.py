"""Shared model infrastructure — base classes, utilities, contracts.

Graph family:
- ``GraphModuleBase`` — base for VGAE, GAT, DGI
- ``try_compile`` — safe torch.compile with conv-type gating
- ``eval_mode`` — context manager that restores training state

Shared:
- ``_ModelBase(pl.LightningModule)`` — mixin shared by ``GraphModuleBase`` +
  ``FusionModuleBase``. Lightning provides ``self.device``, ``self.log``,
  ``self.log_dict``, ``self.hparams``, ``self.trainer``, etc.
- ``safe_load_checkpoint`` — checkpoint loading via class_path registry
- ``strip_orig_mod_prefix`` — drop ``_orig_mod.`` keys from state_dicts
  produced under ``torch.compile``
"""

from __future__ import annotations

import contextlib
import importlib
import math
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import torch
import torch.nn as nn
from structlog import get_logger

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# state_dict utilities — used by safe_load_checkpoint and TauNormCallback
# ---------------------------------------------------------------------------


def strip_orig_mod_prefix(state: dict[str, Any]) -> dict[str, Any]:
    """Drop ``_orig_mod.`` prefix injected by ``torch.compile``'s OptimizedModule.

    ``_orig_mod.`` can appear mid-key (e.g. ``model._orig_mod.encoder.weight``)
    when compile wraps an inner submodule; ``replace`` handles every position.
    """
    return {k.replace("_orig_mod.", ""): v for k, v in state.items()}


# ---------------------------------------------------------------------------
# _ModelBase — shared mixin for every model module
# ---------------------------------------------------------------------------


class _ModelBase(pl.LightningModule):
    """Shared infra for every model module.

    Lightning provides ``self.device``, ``self.log``, ``self.log_dict``,
    ``self.hparams``, ``self.trainer``. graphids-specific additions:

    - ``_store_init_kwargs(locals())`` — mirrors every declared ``__init__``
      kwarg onto ``self`` AND populates ``self.hparams`` via Lightning's
      ``save_hyperparameters`` (skipping ``nn.Module`` values which belong
      in ``state_dict``).
    - ``prepare_from_datamodule(dm)`` — graphids-side hook called by
      :func:`graphids.orchestrate.run_row` BEFORE ``trainer.fit/test`` so
      lazy ``_build()`` runs with DM-supplied ``num_ids`` etc. Distinct
      from Lightning's ``setup(stage)`` hook.
    - per-test-set ``test_step`` plumbing (``_record_test_batch`` /
      ``on_test_epoch_*``).
    - ``on_save_checkpoint`` injects ``class_path`` and strips
      ``_orig_mod.`` from the saved ``state_dict``.

    Subclasses call ``self._store_init_kwargs(locals())`` from their own
    ``__init__`` immediately after ``super().__init__()``.
    """

    def _store_init_kwargs(self, locals_dict: dict) -> None:
        """Mirror declared kwargs onto self AND register with Lightning's hparams.

        Mirroring preserves the ``self.lr``/``self.num_ids`` access pattern
        every model uses internally; ``save_hyperparameters`` populates
        ``self.hparams`` so Lightning's ckpt round-trip works without
        per-class overrides. ``nn.Module`` values (e.g. ``loss_fn``) are
        excluded from hparams — they belong in ``state_dict``.
        """
        import inspect

        sig = inspect.signature(type(self).__init__)
        names = tuple(n for n in sig.parameters if n != "self")
        saved: dict[str, Any] = {}
        for n in names:
            if n in locals_dict:
                v = locals_dict[n]
                setattr(self, n, v)
                if not isinstance(v, nn.Module):
                    saved[n] = v
        self.save_hyperparameters(saved)

    # -- graphids-side preparation hook --------------------------------------
    #
    # Lightning's ``setup(stage)`` runs INSIDE ``trainer.fit/test``, after
    # the dataloaders have been resolved. graphids needs the model lazily
    # built (with DM-supplied ``num_ids`` / ``in_channels`` / ``num_classes``)
    # BEFORE the dataloader is constructed because the budget probe requires
    # an instantiated model. ``prepare_from_datamodule`` is the orchestrate-
    # level hook that fills the pre-fit gap.

    def prepare_from_datamodule(self, dm) -> None:
        """Capture per-test-set names from the DM. Subclasses (e.g.
        ``GraphModuleBase``) override to also lazy-build on first call."""
        ds = getattr(dm, "test_datasets", None)
        self._test_set_names = list(ds.keys()) if ds else ["test"]
        # Snapshot the dataset schema's attack-type code → name map so the
        # per-attack-type AUROC pass at on_test_epoch_end can label its keys
        # (`test/{subdir}/auroc_per_attack/{name}`). Falls back to a
        # benign-only map for datasets without a multiclass taxonomy.
        names_map: dict[int, str] | None = None
        if ds:
            first = next(iter(ds.values()))
            schema = getattr(first, "SCHEMA", None)
            if schema is not None:
                names_map = getattr(schema, "attack_type_names", None)
        self._attack_type_names = dict(names_map or {0: "benign"})

    # -- curriculum loss epoch sync ------------------------------------------
    #
    # Curriculum-aware losses (``CurriculumWeightedLoss``) read the current
    # epoch via ``set_epoch`` to evaluate the visibility schedule each step.
    # Hooking it here keeps every classifier model curriculum-compatible
    # without per-class plumbing — non-curriculum losses don't expose
    # ``set_epoch`` so this is a no-op for them.

    def on_train_epoch_start(self) -> None:
        set_epoch = getattr(getattr(self, "loss_fn", None), "set_epoch", None)
        if callable(set_epoch):
            set_epoch(int(self.current_epoch))

    def on_test_setup(self, datamodule, device) -> None:
        """Fired by orchestrate.evaluate after the model is on device + ckpt
        loaded, before ``model.eval()`` + the test loop. Default no-op.
        Score-based detectors (VGAE/DGI) override to fit calibration
        buffers from a fit-phase loader."""

    # -- per-test-set evaluation (issue #26) ---------------------------------

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
        self._test_buffers = {
            n: {"preds": [], "scores": [], "labels": [], "attack_type": []} for n in names
        }
        # Filled in on_test_epoch_end; stage.evaluate reads to persist to disk.
        self._test_predictions: dict[str, dict[str, torch.Tensor]] = {}

    def _record_test_batch(
        self, dataloader_idx: int, *, scores, labels, preds=None, attack_type=None
    ) -> None:
        """Buffer one batch's predictions under the right test-set bucket."""
        names = getattr(self, "_test_set_names", ["test"])
        name = names[dataloader_idx] if dataloader_idx < len(names) else names[-1]
        buf = self._test_buffers[name]
        buf["scores"].append(scores.detach().cpu())
        buf["labels"].append(labels.detach().cpu())
        if preds is not None:
            buf["preds"].append(preds.detach().cpu())
        if attack_type is not None:
            buf["attack_type"].append(attack_type.detach().cpu())

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
                class1 = probs[:, 1]
                self._log_operating_points(class1, labels, prefix=f"test/{name}/")
                if buf["attack_type"]:
                    self._log_per_attack_auroc(
                        name, class1, labels, torch.cat(buf["attack_type"])
                    )
            all_probs.append(probs)
            all_labels.append(labels)
        if all_probs:
            pooled_p, pooled_l = torch.cat(all_probs), torch.cat(all_labels)
            self.test_metrics.update(pooled_p, pooled_l)
            self.log_dict(self.test_metrics.compute())
            if pooled_p.ndim == 2 and pooled_p.shape[1] == 2:
                self._log_operating_points(pooled_p[:, 1], pooled_l, prefix="test/")

    def _log_per_attack_auroc(
        self,
        name: str,
        class1_scores: torch.Tensor,
        labels: torch.Tensor,
        attack_type: torch.Tensor,
    ) -> None:
        """One-vs-benign binary AUROC per attack code, plus macro mean.

        ``class1_scores`` is the 1-D positive-class score (class-1 prob for
        classifier flavor; raw score for threshold flavor). ``attack_type``
        is per-graph (benign = 0). For each non-zero code present, AUROC
        is computed over the benign∪{this-code} subset; codes lacking
        either class are skipped (binary AUROC undefined).

        Logged keys: ``test/{name}/auroc_per_attack/{attack_name}`` plus
        ``test/{name}/auroc_per_attack_macro`` over present codes.
        """
        if attack_type.numel() == 0:
            return
        from torchmetrics.functional.classification import binary_auroc

        scores = class1_scores.float()
        labels = labels.long()
        attack_type = attack_type.long()
        benign_mask = attack_type == 0
        names_map = getattr(self, "_attack_type_names", {0: "benign"})
        prefix = f"test/{name}/auroc_per_attack"
        per_attack: dict[str, float] = {}
        for code in attack_type.unique().tolist():
            if code == 0:
                continue
            subset = benign_mask | (attack_type == code)
            sub_scores = scores[subset]
            sub_labels = labels[subset]
            if sub_labels.unique().numel() < 2:
                continue
            attack_name = names_map.get(int(code), f"unknown_{int(code)}")
            value = float(binary_auroc(sub_scores, sub_labels))
            per_attack[f"{prefix}/{attack_name}"] = value
        if not per_attack:
            return
        per_attack[f"test/{name}/auroc_per_attack_macro"] = sum(per_attack.values()) / len(
            per_attack
        )
        self.log_dict(per_attack)

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
        # Use ``_at_`` instead of ``@`` — MLflow's metric-name alphabet is
        # ``[A-Za-z0-9_\-. :/]``; ``@`` would be rejected by ``log_batch``
        # and kill the whole row. Emit valid keys at the source rather than
        # sanitizing downstream.
        candidates = {
            f"{prefix}precision_at_{min_recall:g}recall": float(prec),
            f"{prefix}threshold_at_{min_recall:g}recall": float(thr_p),
            f"{prefix}recall_at_{min_precision:g}precision": float(rec),
            f"{prefix}threshold_at_{min_precision:g}precision": float(thr_r),
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

    # -- ckpt round-trip -----------------------------------------------------

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Inject ``class_path`` and strip ``_orig_mod.`` from the state_dict.

        Lightning's ckpt format has ``state_dict`` + ``hyper_parameters`` but
        no class identity; ``safe_load_checkpoint`` dispatches on a
        ``class_path`` we add here. Stripping ``_orig_mod.`` keeps the saved
        weights portable across runs with/without ``compile_model=True``.
        """
        cls = type(self)
        checkpoint["class_path"] = f"{cls.__module__}.{cls.__name__}"
        if "state_dict" in checkpoint:
            checkpoint["state_dict"] = strip_orig_mod_prefix(checkpoint["state_dict"])


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
    """Shared base for VGAE, GAT, DGI — lazy setup, threshold metrics.

    Subclasses must implement ``_build()`` which constructs ``self.model`` and any
    other architecture components using ``self.hparams`` (populated by
    ``prepare_from_datamodule``).
    """

    automatic_optimization = True
    _budget_cache: Any = None  # one BudgetResult per fit (see compute_budget)

    # -- VRAM budget ---------------------------------------------------------

    def compute_budget(self, train_dataset, dataset_name: str) -> Any:
        """Probe-once VRAM budget. ``conv_type`` / ``heads`` are model
        properties, so the probe lives here — not on the DataModule, which
        would have to mirror them as parallel hp.

        ``BudgetResult`` is cached on the model: ``bpn_node`` / ``bpn_edge``
        depend on the model + data, not on which split (train/val/test) is
        packing right now, so val and test loaders reuse the train-time probe.
        """
        if self._budget_cache is None:
            from graphids.core.budget import node_budget

            self._budget_cache = node_budget(
                dataset_name, model=self, train_dataset=train_dataset
            )
        return self._budget_cache

    # -- prepare + optimizers ------------------------------------------------

    def prepare_from_datamodule(self, dm) -> None:
        """Lazy-build with DM-supplied vocab / channel sizes, then capture
        per-test-set names from the DM (via ``super``)."""
        already_built = getattr(self, "_built", False) or (
            getattr(self, "model", "_sentinel") not in (None, "_sentinel")
        )
        if not already_built:
            # Mirror onto self AND into self.hparams so _build() (which reads
            # ``self.hparams.num_ids`` etc.) sees the DM-resolved values.
            for k in ("num_ids", "in_channels", "num_classes"):
                v = getattr(dm, k)
                setattr(self, k, v)
                self.hparams[k] = v
            self._build()
            self._built = True
        super().prepare_from_datamodule(dm)

    def _build(self):
        raise NotImplementedError

    def _init_post(self, locals_dict: dict) -> None:
        """Default ``__init__`` tail for collapsed-arch subclasses.

        Mirrors declared kwargs onto ``self`` (via ``_store_init_kwargs``),
        normalizes ``id_encoder_kwargs`` (None → {}), and lazy-builds when
        ``num_ids`` is already known (e.g. tests instantiate without a
        datamodule). Sets ``self._built`` so ``prepare_from_datamodule``
        doesn't re-build.
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

    def configure_optimizers(self):
        """Adam over all params, using ``self.hparams.lr`` /
        ``self.hparams.weight_decay`` (with sensible defaults). No scheduler;
        subclasses that need one override.
        """
        lr = getattr(self.hparams, "lr", 1e-3)
        wd = getattr(self.hparams, "weight_decay", 0.0)
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)

    def _init_threshold_metrics(self):
        """Call in ``__init__`` for modules that need a Youden-J threshold."""
        from ._metrics import BinaryYoudenJThreshold

        self.roc_metric = BinaryYoudenJThreshold()
        self.test_threshold: float | None = None

    # -- threshold-flavor test path (VGAE/DGI) -------------------------------

    def on_validation_epoch_end(self) -> None:
        """Override in subclasses to compute epoch-level val metrics."""

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
                # `accuracy`/`f1`/etc. above.
                prefix = f"test/{name}/"
                self.log_dict({f"{prefix}{k}": v for k, v in coll.compute().items()})
                self._log_operating_points(scores, labels, prefix=prefix)
                if buf["attack_type"]:
                    self._log_per_attack_auroc(
                        name, scores, labels, torch.cat(buf["attack_type"])
                    )
                # Materialize derived preds so _finalize_test_predictions persists them.
                buf["preds"] = [(scores >= self.test_threshold).long()]
        self._finalize_test_predictions()

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
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
    self-describing ``class_path`` injected by ``_ModelBase.on_save_checkpoint``.
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    from graphids._fs import atomic_load

    ckpt = atomic_load(ckpt_path, map_location=map_location, weights_only=True)
    dotted = ckpt.get("class_path")
    if not dotted:
        raise KeyError(
            f"Checkpoint {ckpt_path} missing 'class_path'. Re-train under the "
            "current LightningModule + on_save_checkpoint contract."
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

    # Lightning serializes self.hparams as an AttributeDict; coerce to plain
    # dict so the **-spread into init_kwargs is well-defined.
    hp = dict(ckpt.get("hyper_parameters", {}))

    # Per-class hook for rebuilding excluded init kwargs (e.g. ``loss_fn``,
    # which can't be pickled into ``hyper_parameters``). Each class that needs
    # something rebuilt declares ``_rebuild_excluded_kwargs(hp) -> dict`` as a
    # classmethod or staticmethod. Default: nothing extra.
    rebuild = getattr(cls, "_rebuild_excluded_kwargs", None)
    extra_kwargs: dict = rebuild(hp) if rebuild is not None else {}

    init_kwargs = {**hp, **extra_kwargs}
    module = cls(**init_kwargs)
    state_dict = strip_orig_mod_prefix(ckpt["state_dict"])
    # Old wrapper ckpts prefixed every key with ``model.`` (the
    # ``self.model = nn.Module(...)`` indirection collapsed away). Strip when
    # the loaded class declares no top-level ``model`` attribute — there's
    # no key collision because the new layer names don't start with ``model.``.
    if not hasattr(module, "model") and any(k.startswith("model.") for k in state_dict):
        state_dict = {k.removeprefix("model."): v for k, v in state_dict.items()}
    # VGAE: ``mask_token`` was a top-level (frozen) Parameter; it's now the
    # buffer ``masker.mask_token`` on a RandomNodeMasker submodule. Remap
    # legacy keys so old ckpts load cleanly.
    if "mask_token" in state_dict and "masker.mask_token" not in state_dict:
        state_dict["masker.mask_token"] = state_dict.pop("mask_token")
    module.load_state_dict(state_dict)

    if hasattr(module, "on_load_checkpoint"):
        module.on_load_checkpoint(ckpt)

    return module
