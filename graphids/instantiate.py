"""Direct instantiation of Trainer/model/datamodule from a rendered config.

Callbacks and loggers are declared in jsonnet config (defaults.libsonnet).
This module is a generic "class_path dict → instance" loop with two
runtime patches: ModelCheckpoint.dirpath and logger save_dir.
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

from graphids.config.constants import CKPT_SUBPATH
from graphids.config.schemas import ValidatedConfig

_CKPT_DIR = str(PurePosixPath(CKPT_SUBPATH).parent)
_WANDB_WRITE_DIR: str = os.environ.get("WANDB_DIR", "/fs/scratch/PAS1266/wandb")


def _import_class(class_path: str) -> type:
    module_name, _, cls_name = class_path.rpartition(".")
    if not module_name:
        raise ValueError(f"class_path must be dotted: {class_path!r}")
    mod = importlib.import_module(module_name)
    try:
        return getattr(mod, cls_name)
    except AttributeError as e:
        raise ImportError(f"{cls_name!r} not found in {module_name!r}") from e


def _init_kwargs(cls: type) -> set[str]:
    """Return keyword-accepted parameter names on ``cls.__init__``."""
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return set()
    return {
        name
        for name, p in sig.parameters.items()
        if name != "self" and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    }


def _instantiate_block(block: dict[str, Any]) -> Any:
    """Instantiate a ``{class_path, init_args}`` dict."""
    cls = _import_class(block["class_path"])
    return cls(**(block.get("init_args") or {}))


def _instantiate_callbacks(
    callback_cfgs: list[dict[str, Any]],
    default_root_dir: str | None,
) -> list[pl.Callback]:
    """Instantiate callback list, patching ModelCheckpoint.dirpath at runtime."""
    callbacks = []
    for entry in callback_cfgs:
        if "ModelCheckpoint" in entry.get("class_path", "") and default_root_dir:
            entry = copy.deepcopy(entry)
            entry.setdefault("init_args", {})["dirpath"] = f"{default_root_dir}/{_CKPT_DIR}"
        callbacks.append(_instantiate_block(entry))
    return callbacks


def _instantiate_loggers(
    logger_cfg: bool | list | dict | None,
    default_root_dir: str | None,
) -> list | bool | None:
    """Instantiate logger list, patching save_dir at runtime."""
    if logger_cfg is None or isinstance(logger_cfg, bool):
        return logger_cfg
    if not isinstance(logger_cfg, list):
        logger_cfg = [logger_cfg]
    loggers = []
    for entry in logger_cfg:
        entry = copy.deepcopy(entry)
        init_args = entry.setdefault("init_args", {})
        cp = entry.get("class_path", "")
        if "WandbLogger" in cp:
            init_args["save_dir"] = _WANDB_WRITE_DIR
        elif "CSVLogger" in cp and default_root_dir:
            init_args["save_dir"] = default_root_dir
        loggers.append(_instantiate_block(entry))
    return loggers


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

    # -- model --
    model_cls = _import_class(merged["model"]["class_path"])
    model_init = inject_loss_fn(
        merged["model"].get("init_args") or {},
        class_path=merged["model"]["class_path"],
    )
    accepted = _init_kwargs(model_cls)
    model = model_cls(**{k: v for k, v in model_init.items() if k in accepted})

    # -- trainer --
    trainer_cfg = dict(merged.get("trainer") or {})
    default_root_dir = trainer_cfg.get("default_root_dir")
    trainer_cfg["callbacks"] = _instantiate_callbacks(
        trainer_cfg.pop("callbacks", []), default_root_dir
    )
    trainer_cfg["logger"] = _instantiate_loggers(trainer_cfg.pop("logger", None), default_root_dir)
    trainer = pl.Trainer(**trainer_cfg)

    # -- wandb config forwarding --
    for lg in trainer.loggers or []:
        if type(lg).__name__ == "WandbLogger":
            try:
                lg.experiment.config.update(rendered, allow_val_change=True)
            except Exception:
                pass
            break

    return InstantiatedRun(trainer=trainer, model=model, datamodule=datamodule, merged=merged)
