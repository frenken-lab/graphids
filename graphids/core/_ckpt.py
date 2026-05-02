"""Checkpoint save/load — paired schema, single owner.

The ckpt dict shape is keyed by Lightning-format conventions
(``state_dict``, ``optimizer_states[i]``, ``lr_schedulers[i]``,
``scaler``, ``epoch``, ``global_step``, ``hyper_parameters``,
``class_path``). Save and load must agree on those keys, so they live
together here. Trainer + ``ModelCheckpoint`` are the only callers.

``_orig_mod.`` prefix handling lets ckpts move between
``compile_model=True`` and ``compile_model=False`` runs without
strict-load failures.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from structlog import get_logger

from graphids._fs import atomic_load

if TYPE_CHECKING:
    from graphids.core.trainer import Trainer

_log = get_logger(__name__)


def strip_orig_mod_prefix(state: dict[str, Any]) -> dict[str, Any]:
    """Drop ``_orig_mod.`` prefix injected by ``torch.compile``'s OptimizedModule.

    ``_orig_mod.`` can appear mid-key (e.g. ``model._orig_mod.encoder.weight``)
    when compile wraps an inner submodule; ``replace`` handles every position.
    """
    return {k.replace("_orig_mod.", ""): v for k, v in state.items()}


def build_checkpoint(trainer: Trainer, model: nn.Module) -> dict[str, Any]:
    """Build a raw-PyTorch checkpoint dict."""
    cls = type(model)
    hp = model.hparams
    ckpt: dict[str, Any] = {
        "state_dict": strip_orig_mod_prefix(model.state_dict()),
        "epoch": trainer.current_epoch,
        "global_step": trainer.global_step,
        "class_path": f"{cls.__module__}.{cls.__name__}",
        "hyper_parameters": vars(hp) if hasattr(hp, "__dict__") else dict(hp),
    }
    if trainer.callback_metrics:
        ckpt["metrics"] = {k: float(v) for k, v in trainer.callback_metrics.items()}
    model.on_save_checkpoint(ckpt)
    if trainer._optimizers:
        ckpt["optimizer_states"] = [opt.state_dict() for opt in trainer._optimizers]
    if trainer._schedulers:
        ckpt["lr_schedulers"] = [s.state_dict() for s in trainer._schedulers if s is not None]
    return ckpt


def load_state_into_model(ckpt_path: str, model: nn.Module, device: torch.device) -> dict:
    """Load ckpt, restore weights, fire ``on_load_checkpoint``. Return raw dict.

    ``strict=False`` tolerates removed buffers (e.g. DGI ``svdd_calibrated``,
    dropped when centroid fit moved from state_dict to test-start). Logs
    unexpected/missing keys so architecture drift stays visible.
    """
    ckpt = atomic_load(ckpt_path, map_location=device, weights_only=True)
    stripped = strip_orig_mod_prefix(ckpt.get("state_dict", ckpt))
    # Align ckpt to target's compile-prefix convention. Save strips
    # ``_orig_mod.``; target may or may not have it depending on whether
    # this run has compile_model enabled. Remap via the target's keys.
    remap = {k.replace("_orig_mod.", ""): k for k in model.state_dict().keys()}
    state = {remap.get(k, k): v for k, v in stripped.items()}
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys or result.unexpected_keys:
        _log.info(
            "load_state_dict_partial",
            missing=list(result.missing_keys),
            unexpected=list(result.unexpected_keys),
        )
    model.on_load_checkpoint(ckpt)
    return ckpt


def restore_training_state(
    ckpt: dict,
    trainer: Trainer,
    opt: torch.optim.Optimizer | None,
    scheduler: Any,
    scaler: torch.amp.GradScaler,
) -> None:
    """Apply optimizer/scheduler/scaler/epoch state from a loaded ckpt dict."""
    trainer.current_epoch = ckpt.get("epoch", 0) + 1
    trainer.global_step = ckpt.get("global_step", 0)
    if opt and "optimizer_states" in ckpt:
        opt.load_state_dict(ckpt["optimizer_states"][0])
    if scheduler and "lr_schedulers" in ckpt:
        scheduler.load_state_dict(ckpt["lr_schedulers"][0])
    if "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
