"""Instantiate Trainer/model/datamodule from a rendered config dict.

The only path from jsonnet output to a runnable Lightning stack. Both
``cli/_training.py`` (dev) and ``orchestrate/actors.py`` (Monarch) call
``instantiate()``.
"""

from __future__ import annotations

import copy
import importlib
import inspect
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

import pytorch_lightning as pl

from graphids.config.constants import CKPT_SUBPATH
from graphids.config.schemas import ValidatedConfig

_CKPT_DIR = str(PurePosixPath(CKPT_SUBPATH).parent)


def _import_class(class_path: str) -> type:
    """Import a class from a dotted ``module.ClassName`` string."""
    module_name, _, cls_name = class_path.rpartition(".")
    if not module_name:
        raise ValueError(f"class_path must be dotted: {class_path!r}")
    mod = importlib.import_module(module_name)
    try:
        return getattr(mod, cls_name)
    except AttributeError as e:
        raise ImportError(f"{cls_name!r} not found in {module_name!r}") from e


def _instantiate_block(block: dict[str, Any]) -> Any:
    """Instantiate a ``{class_path, init_args}`` dict."""
    cls = _import_class(block["class_path"])
    return cls(**(block.get("init_args") or {}))


@dataclass
class InstantiatedRun:
    """Output of :func:`instantiate`."""

    trainer: pl.Trainer
    model: pl.LightningModule
    datamodule: pl.LightningDataModule
    merged: dict[str, Any]


def instantiate(
    rendered: dict[str, Any],
    *,
    validated: ValidatedConfig | None = None,
    seed_everything: bool = True,
) -> InstantiatedRun:
    """Instantiate the full Lightning stack from a rendered config dict."""
    from graphids.config.schemas import validate_config
    from graphids.core.losses.build import inject_loss_fn

    merged = copy.deepcopy(rendered)
    if validated is None:
        validated = validate_config(merged)

    if seed_everything:
        pl.seed_everything(merged["seed_everything"], workers=True)

    # -- datamodule --
    datamodule = _instantiate_block(merged["data"])

    # -- model (signature-filtered: skip kwargs the class doesn't accept) --
    model_cls = _import_class(merged["model"]["class_path"])
    model_init = inject_loss_fn(
        merged["model"].get("init_args") or {},
        class_path=merged["model"]["class_path"],
    )
    try:
        sig = inspect.signature(model_cls.__init__)
    except (TypeError, ValueError):
        sig = None
    if sig is not None:
        accepted = {
            name for name, p in sig.parameters.items()
            if name != "self" and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
        }
        model_init = {k: v for k, v in model_init.items() if k in accepted}
    model = model_cls(**model_init)

    # -- trainer (callbacks + loggers from config) --
    trainer_cfg = dict(merged.get("trainer") or {})
    default_root_dir = trainer_cfg.get("default_root_dir")

    # Callbacks: patch ModelCheckpoint.dirpath at runtime
    callbacks = []
    for entry in trainer_cfg.pop("callbacks", []):
        if "ModelCheckpoint" in entry.get("class_path", "") and default_root_dir:
            entry = copy.deepcopy(entry)
            entry.setdefault("init_args", {})["dirpath"] = f"{default_root_dir}/{_CKPT_DIR}"
        callbacks.append(_instantiate_block(entry))
    trainer_cfg["callbacks"] = callbacks

    # Loggers: handle bool / list / dict
    logger_cfg = trainer_cfg.pop("logger", None)
    if isinstance(logger_cfg, (list, dict)):
        if not isinstance(logger_cfg, list):
            logger_cfg = [logger_cfg]
        trainer_cfg["logger"] = [_instantiate_block(e) for e in logger_cfg]
    else:
        trainer_cfg["logger"] = logger_cfg  # None or bool passthrough

    trainer = pl.Trainer(**trainer_cfg)

    return InstantiatedRun(trainer=trainer, model=model, datamodule=datamodule, merged=merged)
