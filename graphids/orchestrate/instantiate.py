"""Instantiation — Layer 3 of the orchestrate stack.

Flat module (no class wrapper): each verb is a module-level function.
``build_run`` composes model + datamodule + trainer + callbacks +
loggers from a rendered config dict into an ``InstantiatedRun``.
Every ``{class_path, init_args}`` block is resolved via ``importlib``,
with signature-filtered kwargs so jsonnet can pass fields the target
class doesn't accept without raising ``TypeError``.
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
_TRAINER_CONFIG_KEYS: frozenset[str] = frozenset(
    f.name for f in dataclasses.fields(TrainerConfig)
)

# Re-export for existing tests that import filter_kwargs from this module.
__all__ = [
    "build_block",
    "build_callbacks",
    "build_datamodule",
    "build_loggers",
    "build_model",
    "build_model_from_config",
    "build_run",
    "build_trainer",
    "filter_kwargs",
    "import_class",
]


def _resolve_nested(value: Any) -> Any:
    """Recursively instantiate any ``{class_path, init_args}`` dicts inside ``value``.

    Lets jsonnet compose nested class_path blocks inside an outer
    ``init_args``. Plain dicts (including KD auxiliary config) are
    walked key-by-key and left alone if they don't carry a
    ``class_path`` key.
    """
    if isinstance(value, dict):
        if "class_path" in value:
            return build_block(value)
        return {k: _resolve_nested(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_nested(v) for v in value]
    return value


def build_block(block: dict[str, Any]) -> Any:
    """Instantiate a ``{class_path, init_args}`` dict (recursing into nested blocks)."""
    klass = import_class(block["class_path"])
    init_args = block.get("init_args") or {}
    resolved = {k: _resolve_nested(v) for k, v in init_args.items()}
    return klass(**resolved)


def build_model(class_path: str, init_args: dict[str, Any]) -> nn.Module:
    klass = import_class(class_path)
    return klass(**filter_kwargs(klass, init_args))


def build_model_from_config(merged: dict[str, Any]) -> nn.Module:
    """Build model from rendered config with loss injection."""
    init_args = inject_loss_fn(
        merged["model"].get("init_args") or {},
        class_path=merged["model"]["class_path"],
    )
    return build_model(merged["model"]["class_path"], init_args)


def build_datamodule(merged: dict[str, Any]) -> Any:
    return build_block(merged["data"])


def build_callbacks(merged: dict[str, Any]) -> list:
    """Instantiate the callback list from ``trainer.callbacks``.

    ``ModelCheckpoint.dirpath`` is wired in jsonnet from ``run_dir``
    (see ``configs/_lib/defaults.libsonnet``) — no runtime patching.
    """
    entries = (merged.get("trainer") or {}).get("callbacks") or []
    return [build_block(entry) for entry in entries]


def build_loggers(merged: dict[str, Any]) -> list | bool | None:
    logger_cfg = (merged.get("trainer") or {}).get("logger")
    if isinstance(logger_cfg, (list, dict)):
        entries = logger_cfg if isinstance(logger_cfg, list) else [logger_cfg]
        return [build_block(e) for e in entries]
    return logger_cfg  # None or bool


def build_trainer(merged: dict[str, Any]) -> Trainer:
    trainer_dict = merged.get("trainer") or {}
    cfg_kwargs = {k: v for k, v in trainer_dict.items() if k in _TRAINER_CONFIG_KEYS}
    return Trainer(
        config=TrainerConfig(**cfg_kwargs),
        callbacks=build_callbacks(merged),
        logger=build_loggers(merged),
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
        model=build_model_from_config(merged),
        datamodule=build_datamodule(merged),
        trainer=build_trainer(merged),
    )
