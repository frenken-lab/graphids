"""Instantiation — Layer 3 of the orchestrate stack.

``build_run`` is the only public entry point. It composes model +
datamodule + trainer + callbacks + loggers from a rendered config dict
into an ``InstantiatedRun``. Every ``{class_path, init_args}`` block is
resolved via ``importlib``, with signature-filtered kwargs so jsonnet
can pass fields the target class doesn't accept without raising.
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Any

import torch.nn as nn

from graphids._reflect import filter_kwargs, import_class
from graphids.config.schemas import ValidatedConfig, validate_config
from graphids.core.losses.build import inject_loss_fn
from graphids.core.trainer import Trainer, TrainerConfig, seed_everything
from graphids.orchestrate.config import InstantiatedRun

# Derived once at import time so adding a field to TrainerConfig
# doesn't require editing this module.
_TRAINER_CONFIG_KEYS: frozenset[str] = frozenset(f.name for f in dataclasses.fields(TrainerConfig))


def _resolve_nested(value: Any) -> Any:
    """Recursively instantiate any ``{class_path, init_args}`` dicts inside ``value``.

    Lets jsonnet compose nested class_path blocks inside an outer
    ``init_args``. Plain dicts (including KD auxiliary config) are
    walked key-by-key and left alone if they don't carry a
    ``class_path`` key.
    """
    if isinstance(value, dict):
        if "class_path" in value:
            return _build_block(value)
        return {k: _resolve_nested(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_nested(v) for v in value]
    return value


def _build_block(block: dict[str, Any]) -> Any:
    """Instantiate a ``{class_path, init_args}`` dict (recursing into nested blocks)."""
    klass = import_class(block["class_path"])
    init_args = block.get("init_args") or {}
    resolved = {k: _resolve_nested(v) for k, v in init_args.items()}
    return klass(**resolved)


def _build_model(merged: dict[str, Any]) -> nn.Module:
    """Build model from rendered config with loss injection + signature filtering."""
    class_path = merged["model"]["class_path"]
    init_args = inject_loss_fn(
        merged["model"].get("init_args") or {},
        class_path=class_path,
    )
    klass = import_class(class_path)
    return klass(**filter_kwargs(klass, init_args))


def _build_callbacks(merged: dict[str, Any]) -> list:
    """Instantiate the callback list from ``trainer.callbacks``.

    ``ModelCheckpoint.dirpath`` is wired in jsonnet from ``run_dir``
    (see ``configs/_lib/defaults.libsonnet``) — no runtime patching.

    Appends a ``VRAMDriftCallback`` when CUDA is available so long runs
    surface co-resident-process / activation-leak drift without needing
    each stage jsonnet to opt in. Cheap to run (two ``mem_get_info``
    calls per epoch), and skipped entirely on login nodes.
    """
    import torch

    from graphids.config.settings import get_settings
    from graphids.core.callbacks import VRAMDriftCallback

    entries = (merged.get("trainer") or {}).get("callbacks") or []
    callbacks = [_build_block(entry) for entry in entries]
    if torch.cuda.is_available():
        callbacks.append(
            VRAMDriftCallback(threshold=get_settings().vram_drift_threshold),
        )
    return callbacks


def _build_loggers(merged: dict[str, Any]) -> list | bool | None:
    logger_cfg = (merged.get("trainer") or {}).get("logger")
    if isinstance(logger_cfg, (list, dict)):
        entries = logger_cfg if isinstance(logger_cfg, list) else [logger_cfg]
        return [_build_block(e) for e in entries]
    return logger_cfg  # None or bool


def _build_trainer(merged: dict[str, Any]) -> Trainer:
    trainer_dict = merged.get("trainer") or {}
    cfg_kwargs = {k: v for k, v in trainer_dict.items() if k in _TRAINER_CONFIG_KEYS}
    return Trainer(
        config=TrainerConfig(**cfg_kwargs),
        callbacks=_build_callbacks(merged),
        logger=_build_loggers(merged),
    )


def build_run(
    rendered: dict[str, Any],
    *,
    validated: ValidatedConfig | None = None,
    seed_all: bool = True,
) -> InstantiatedRun:
    merged = copy.deepcopy(rendered)
    if validated is None:
        validate_config(merged)
    if seed_all:
        seed_everything(merged["seed_everything"])

    return InstantiatedRun(
        model=_build_model(merged),
        datamodule=_build_block(merged["data"]),
        trainer=_build_trainer(merged),
    )
