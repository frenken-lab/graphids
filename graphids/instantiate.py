"""Direct instantiation of Trainer/model/datamodule from a rendered config.

Phase 3 (2026-04-05) replaces the jsonargparse + ``LightningCLI`` call path
with explicit importlib-based instantiation. The rendered jsonnet dict is
validated by ``ValidatedConfig``, then this module:

1. Applies link_arguments (dataset/seed/conv_type etc. propagation) by
   copying values between ``data.init_args`` and ``model.init_args``.
2. Imports ``data.class_path`` + ``model.class_path`` via ``importlib``
   and instantiates them with their ``init_args``.
3. Constructs ``trainer.logger`` entries (list of ``{class_path, init_args}``
   dicts) and patches logger save dirs.
4. Constructs the forced callbacks (``ModelCheckpoint``, ``EarlyStopping``,
   ``DeviceStatsMonitor``, ``ResourceProfileCallback``, ``RunRecordCallback``)
   plus any user-provided ``trainer.callbacks``, and patches
   ``ModelCheckpoint.dirpath``.
5. Builds the Lightning ``Trainer``.

No jsonargparse, no ``LightningCLI``. Every knob that used to flow through
``add_lightning_class_args`` / ``link_arguments`` is materialized here.
"""

from __future__ import annotations

import copy
import importlib
import inspect
import os
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    DeviceStatsMonitor,
    EarlyStopping,
    ModelCheckpoint,
)

from graphids.config.constants import CKPT_SUBPATH
from graphids.config.schemas import ValidatedConfig
from graphids.core.monitoring.callbacks import ResourceProfileCallback, RunRecordCallback

_CKPT_DIR = str(PurePosixPath(CKPT_SUBPATH).parent)

# WandbLogger save_dir — env-driven because OSC scratch path varies by job.
# Not in config/constants.py because its only consumer is this module.
_WANDB_WRITE_DIR: str = os.environ.get("WANDB_DIR", "/fs/scratch/PAS1266/wandb")

# -----------------------------------------------------------------------------
# Link arguments — formerly parser.link_arguments in GraphIDSCLI. Each entry
# copies a value from ``src`` (dotted path) to ``tgt`` (dotted path) if the
# src is present and the target slot is missing or None.
# -----------------------------------------------------------------------------

_LINK_TARGETS: tuple[tuple[str, str], ...] = (
    ("data.init_args.dataset", "model.init_args.dataset"),
    ("data.init_args.lake_root", "model.init_args.lake_root"),
    ("seed_everything", "model.init_args.seed"),
    ("seed_everything", "data.init_args.seed"),
    ("model.init_args.conv_type", "data.init_args.conv_type"),
    ("model.init_args.heads", "data.init_args.heads"),
)


def _get_dotted(d: dict[str, Any], path: str) -> Any:
    """Walk a dotted path into nested dicts. Returns ``None`` on miss."""
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _set_dotted(d: dict[str, Any], path: str, value: Any) -> None:
    """Set a dotted path into nested dicts, creating intermediate dicts."""
    parts = path.split(".")
    cur: Any = d
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


def _init_kwargs(cls: type) -> set[str]:
    """Return the set of keyword-accepted parameter names on ``cls.__init__``.

    Wraps ``inspect.signature`` so lookups are cheap and tolerate missing
    annotations. Includes every param kind except ``self``, ``*args``, and
    ``**kwargs`` — LightningModule / DataModule inits in this repo use
    explicit typed kwargs (no ``**kwargs``), so a set is an exact match.
    """
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):  # pragma: no cover - C-implemented base
        return set()
    return {
        name
        for name, p in sig.parameters.items()
        if name != "self" and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    }


def _apply_link_arguments(
    merged: dict[str, Any],
    *,
    dm_cls: type,
    model_cls: type,
) -> None:
    """Copy linked values in-place, filtering by target-class signature.

    jsonargparse's ``parser.link_arguments`` only applied a link when the
    target class accepted the parameter — fusion models, for example, don't
    take ``dataset`` / ``conv_type`` / ``heads`` so VGAE's links were
    silently no-ops. We replicate that by introspecting ``dm_cls`` and
    ``model_cls`` once and skipping links whose last dotted segment isn't
    in the target signature.
    """
    dm_params = _init_kwargs(dm_cls)
    model_params = _init_kwargs(model_cls)

    for src, tgt in _LINK_TARGETS:
        src_val = _get_dotted(merged, src)
        if src_val is None:
            continue
        tgt_parts = tgt.split(".")
        leaf = tgt_parts[-1]
        if tgt_parts[0] == "data" and leaf not in dm_params:
            continue
        if tgt_parts[0] == "model" and leaf not in model_params:
            continue
        tgt_val = _get_dotted(merged, tgt)
        if tgt_val is None:
            _set_dotted(merged, tgt, src_val)


# -----------------------------------------------------------------------------
# Class-path import
# -----------------------------------------------------------------------------


def _import_class(class_path: str) -> type:
    module_name, _, cls_name = class_path.rpartition(".")
    if not module_name:
        raise ValueError(f"class_path must be dotted: {class_path!r}")
    mod = importlib.import_module(module_name)
    try:
        return getattr(mod, cls_name)
    except AttributeError as e:  # pragma: no cover - clearer error only
        raise ImportError(f"{cls_name!r} not found in {module_name!r}") from e


# -----------------------------------------------------------------------------
# Loss construction — KD is expressed as a composable loss wrapper rather
# than a cross-cutting concern on the LightningModule. This hook reads
# ``loss_config`` / ``distillation_config`` dicts from ``model.init_args``,
# builds the base loss, optionally wraps it for KD, and injects the result
# as ``loss_fn`` before ``model_cls(**init_args)`` is called.
#
# ``model_type`` dispatch (classification vs autoencoder) matches the
# field on the LightningModule itself — GAT → classification shape, VGAE →
# autoencoder shape. Fusion / DGI modules don't take ``loss_fn`` and are
# skipped.
# -----------------------------------------------------------------------------


_LOSS_MODEL_TYPES = frozenset({"gat", "vgae"})


def _build_loss(
    model_type: str,
    loss_config: dict[str, Any] | None,
    distillation_config: dict[str, Any] | None,
):
    """Return an ``nn.Module`` loss for ``model_type`` or ``None`` if N/A.

    For ``gat``: base is one of ``CrossEntropyLoss`` / ``WeightedCrossEntropyLoss``
    / ``FocalLoss`` keyed by ``loss_config["type"]`` (default ``ce``). When
    ``distillation_config`` is present, wraps the base in
    ``SoftLabelDistillation`` with a teacher loaded from
    ``distillation_config["teacher_ckpt"]``.

    For ``vgae``: base is ``VGAETaskLoss`` with the task weights from
    ``loss_config``. When ``distillation_config`` is present, wraps in
    ``FeatureDistillation`` with a teacher loaded from the teacher ckpt
    and an optional projection layer if student/teacher ``latent_dim`` differ.

    Returns ``None`` for model types that don't take an injected ``loss_fn``
    (fusion, dqn, dgi, etc).
    """
    if model_type not in _LOSS_MODEL_TYPES:
        return None

    import torch
    import torch.nn as nn

    from graphids.core.losses import (
        CrossEntropyLoss,
        FeatureDistillation,
        FocalLoss,
        SoftLabelDistillation,
        VGAETaskLoss,
        WeightedCrossEntropyLoss,
    )
    from graphids.core.models._training import load_inner_model

    loss_cfg = dict(loss_config or {})

    if model_type == "gat":
        loss_type = loss_cfg.pop("type", "ce")
        if loss_type == "focal":
            base: nn.Module = FocalLoss(gamma=loss_cfg.get("gamma", 2.0))
        elif loss_type == "weighted_ce":
            base = WeightedCrossEntropyLoss(weights=loss_cfg["weights"])
        elif loss_type == "ce":
            base = CrossEntropyLoss()
        else:
            raise ValueError(
                f"Unknown loss type {loss_type!r} for gat. Expected one of: ce, focal, weighted_ce."
            )

        if not distillation_config:
            return base

        teacher_ckpt = distillation_config["teacher_ckpt"]
        teacher, _ = load_inner_model("gat", teacher_ckpt, torch.device("cpu"))
        return SoftLabelDistillation(
            base,
            teacher,
            temperature=distillation_config.get("temperature", 4.0),
            alpha=distillation_config.get("alpha", 0.7),
        )

    # model_type == "vgae"
    base = VGAETaskLoss(
        canid_weight=loss_cfg.get("canid_weight", 0.1),
        nbr_weight=loss_cfg.get("nbr_weight", 0.05),
        kl_weight=loss_cfg.get("kl_weight", 0.01),
        k_neg=loss_cfg.get("k_neg", 32),
        # num_ids is populated later by VGAEModule.setup() from dm.num_ids
        num_ids=0,
    )

    if not distillation_config:
        return base

    teacher_ckpt = distillation_config["teacher_ckpt"]
    teacher, teacher_hparams = load_inner_model("vgae", teacher_ckpt, torch.device("cpu"))

    # Projection aligns student latent space → teacher latent space when
    # the two scales have different latent_dim. Read the student dim from
    # init_args (passed through) and the teacher dim from its hparams.
    projection: nn.Linear | None = None
    s_dim = distillation_config.get("student_latent_dim")
    t_dim = getattr(teacher_hparams, "latent_dim", None)
    if s_dim and t_dim and s_dim != t_dim:
        projection = nn.Linear(s_dim, t_dim)

    return FeatureDistillation(
        base,
        teacher,
        latent_weight=distillation_config.get("latent_weight", 1.0),
        recon_weight=distillation_config.get("recon_weight", 1.0),
        alpha=distillation_config.get("alpha", 0.7),
        projection=projection,
    )


def _inject_loss_fn(model_init_args: dict[str, Any]) -> dict[str, Any]:
    """Pop loss/distillation config from init_args, build loss, inject as ``loss_fn``.

    Returns a NEW dict — leaves the caller's copy alone. Silently drops
    any stale ``auxiliaries`` list (the pre-refactor KD config shape) —
    it has no meaning in the new layout.
    """
    init_args = dict(model_init_args)
    init_args.pop("auxiliaries", None)  # dead field from pre-Option-B config

    loss_cfg = init_args.pop("loss_config", None)
    kd_cfg = init_args.pop("distillation_config", None)
    model_type = init_args.get("model_type")

    # Student latent dim needs to flow into FeatureDistillation for the
    # projection layer decision — pass it through on the kd_cfg dict.
    if kd_cfg and model_type == "vgae" and "student_latent_dim" not in kd_cfg:
        latent_dim = init_args.get("latent_dim")
        if latent_dim is not None:
            kd_cfg = {**kd_cfg, "student_latent_dim": latent_dim}

    loss_fn = _build_loss(model_type, loss_cfg, kd_cfg)
    if loss_fn is not None:
        init_args["loss_fn"] = loss_fn
    return init_args


# -----------------------------------------------------------------------------
# Logger / callback construction
# -----------------------------------------------------------------------------


def _instantiate_block(block: dict[str, Any]) -> Any:
    """Instantiate a ``{class_path, init_args}`` dict."""
    cls = _import_class(block["class_path"])
    init_args = block.get("init_args") or {}
    return cls(**init_args)


def _build_loggers(
    trainer_cfg: dict[str, Any],
    default_root_dir: str | None,
) -> list | bool | None:
    """Construct logger instances from ``trainer.logger`` config.

    ``true`` / ``false`` / ``None`` pass through to ``pl.Trainer`` as-is.
    A list of ``{class_path, init_args}`` dicts is instantiated here with
    ``WandbLogger.save_dir`` patched to ``WANDB_WRITE_DIR`` and
    ``CSVLogger.save_dir`` patched to ``default_root_dir``.
    """
    logger_cfg = trainer_cfg.get("logger")
    if logger_cfg is None or isinstance(logger_cfg, bool):
        return logger_cfg

    if not isinstance(logger_cfg, list):
        # Single {class_path, init_args} — rare but tolerated.
        logger_cfg = [logger_cfg]

    loggers: list = []
    for entry in logger_cfg:
        if not isinstance(entry, dict) or "class_path" not in entry:
            raise ValueError(
                f"trainer.logger entry must be a {{class_path, init_args}} dict, got {entry!r}"
            )
        entry = copy.deepcopy(entry)
        init_args = entry.setdefault("init_args", {})
        cp = entry["class_path"]
        if "WandbLogger" in cp:
            init_args["save_dir"] = _WANDB_WRITE_DIR
        elif "CSVLogger" in cp and default_root_dir:
            init_args["save_dir"] = default_root_dir
        loggers.append(_instantiate_block(entry))
    return loggers


def _build_callbacks(
    merged: dict[str, Any],
    default_root_dir: str | None,
) -> list[pl.Callback]:
    """Construct the forced callback list + any user ``trainer.callbacks``.

    Forced callbacks (always present, in this order):
        - ModelCheckpoint(**checkpoint)          dirpath patched to {root}/checkpoints
        - EarlyStopping(**early_stopping)
        - DeviceStatsMonitor()
        - ResourceProfileCallback(log_every_n_steps=50)
        - RunRecordCallback(enabled=True)

    User callbacks from ``trainer.callbacks`` are appended after the forced
    set so a stage can still wire ``LearningRateMonitor`` etc.
    """
    ckpt_kwargs = dict(merged.get("checkpoint") or {})
    if default_root_dir:
        ckpt_kwargs["dirpath"] = f"{default_root_dir}/{_CKPT_DIR}"
    es_kwargs = dict(merged.get("early_stopping") or {})

    from graphids.core.data.sampler import CurriculumEpochCallback

    forced: list[pl.Callback] = [
        ModelCheckpoint(**ckpt_kwargs),
        EarlyStopping(**es_kwargs),
        DeviceStatsMonitor(),
        ResourceProfileCallback(log_every_n_steps=50),
        RunRecordCallback(enabled=True),
        CurriculumEpochCallback(),  # no-op when sampler != "curriculum"
    ]

    user: list[pl.Callback] = []
    for entry in (merged.get("trainer") or {}).get("callbacks") or []:
        if not isinstance(entry, dict) or "class_path" not in entry:
            raise ValueError(
                f"trainer.callbacks entry must be a {{class_path, init_args}} dict, got {entry!r}"
            )
        user.append(_instantiate_block(entry))

    return forced + user


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


@dataclass
class InstantiatedRun:
    """Output of :func:`instantiate` — mirrors the old ``build_cli`` return shape."""

    trainer: pl.Trainer
    model: pl.LightningModule
    datamodule: pl.LightningDataModule
    merged: dict[str, Any]  # final resolved dict (post link_arguments)


def instantiate(
    rendered: dict[str, Any],
    *,
    validated: ValidatedConfig | None = None,
    seed_everything: bool = True,
) -> InstantiatedRun:
    """Instantiate the full Lightning stack from a rendered config dict.

    Parameters
    ----------
    rendered:
        Output of ``graphids.config.jsonnet.render_config``. Will be
        deep-copied and mutated in place (link_arguments propagation).
    validated:
        Optional pre-validated view. If ``None``, ``validate_config`` is
        called internally so callers that already validated don't pay twice.
    seed_everything:
        If ``True`` (default), call ``pl.seed_everything(cfg.seed_everything)``
        before instantiation to match the old ``LightningCLI`` behavior.

    Returns
    -------
    InstantiatedRun
        Holds the wired Trainer + model + datamodule ready for
        ``trainer.fit`` / ``trainer.test``.
    """
    from graphids.config.schemas import validate_config

    merged = copy.deepcopy(rendered)
    if validated is None:
        validated = validate_config(merged)

    # Import classes first so link_arguments can filter by target signature.
    dm_cls = _import_class(merged["data"]["class_path"])
    model_cls = _import_class(merged["model"]["class_path"])

    _apply_link_arguments(merged, dm_cls=dm_cls, model_cls=model_cls)

    if seed_everything:
        pl.seed_everything(merged["seed_everything"], workers=True)

    # -- datamodule ----------------------------------------------------------
    datamodule = dm_cls(**(merged["data"].get("init_args") or {}))

    # -- model ---------------------------------------------------------------
    model_init = _inject_loss_fn(merged["model"].get("init_args") or {})
    model = model_cls(**model_init)

    # -- trainer -------------------------------------------------------------
    trainer_cfg = dict(merged.get("trainer") or {})
    default_root_dir = trainer_cfg.get("default_root_dir")
    # Lightning Trainer doesn't accept ``callbacks``/``logger`` as dicts — we
    # pass constructed instances built from the validated sections below.
    trainer_cfg.pop("callbacks", None)
    trainer_cfg["logger"] = _build_loggers(trainer_cfg, default_root_dir)
    callbacks = _build_callbacks(merged, default_root_dir)

    trainer = pl.Trainer(callbacks=callbacks, **trainer_cfg)

    # -- Wandb config forwarding (formerly WandbSaveConfigCallback) ---------
    # WandbLogger needs the full rendered config on its run.config so sweep
    # runs are searchable by any hyperparameter. Lightning's
    # SaveConfigCallback used to do this; in Phase 3 we just push the dict
    # ourselves once the logger is constructed.
    for lg in trainer.loggers or []:
        if type(lg).__name__ == "WandbLogger":
            try:
                lg.experiment.config.update(rendered, allow_val_change=True)
            except Exception:  # noqa: BLE001 — wandb offline / disabled is fine
                pass
            break

    return InstantiatedRun(
        trainer=trainer,
        model=model,
        datamodule=datamodule,
        merged=merged,
    )
