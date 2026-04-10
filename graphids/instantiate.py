"""Instantiate training components from rendered config dicts.

Each component (model, datamodule, callbacks, loggers, trainer) has its
own method. ``build_run`` composes them. Callers pick the granularity
they need.
"""

from __future__ import annotations

import copy
import importlib
import inspect
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

import torch.nn as nn

from graphids.config.constants import CKPT_SUBPATH
from graphids.config.schemas import ValidatedConfig
from graphids.core.trainer import Trainer, TrainerConfig, seed_everything

_CKPT_DIR = str(PurePosixPath(CKPT_SUBPATH).parent)

# Keys consumed by TrainerConfig (popped from the trainer dict before construction)
_TRAINER_CONFIG_KEYS = {
    "max_epochs", "precision", "gradient_clip_val", "log_every_n_steps",
    "accelerator", "devices", "default_root_dir",
}


@dataclass
class InstantiatedRun:
    trainer: Trainer
    model: nn.Module
    datamodule: Any
    merged: dict[str, Any]


class Instantiator:
    """Construct training objects from rendered config dicts."""

    # -- primitives --

    @staticmethod
    def import_class(class_path: str) -> type:
        module_name, _, cls_name = class_path.rpartition(".")
        if not module_name:
            raise ValueError(f"class_path must be dotted: {class_path!r}")
        mod = importlib.import_module(module_name)
        try:
            return getattr(mod, cls_name)
        except AttributeError as e:
            raise ImportError(f"{cls_name!r} not found in {module_name!r}") from e

    @staticmethod
    def filter_kwargs(cls: type, kwargs: dict[str, Any]) -> dict[str, Any]:
        try:
            sig = inspect.signature(cls.__init__)
        except (TypeError, ValueError):
            return kwargs
        if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
            return kwargs
        accepted = {
            name for name, p in sig.parameters.items()
            if name != "self" and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
        }
        return {k: v for k, v in kwargs.items() if k in accepted}

    @classmethod
    def build_block(cls, block: dict[str, Any]) -> Any:
        """Instantiate a ``{class_path, init_args}`` dict."""
        klass = cls.import_class(block["class_path"])
        return klass(**(block.get("init_args") or {}))

    # -- components --

    @classmethod
    def build_model(cls, class_path: str, init_args: dict[str, Any]) -> nn.Module:
        klass = cls.import_class(class_path)
        return klass(**cls.filter_kwargs(klass, init_args))

    @classmethod
    def build_model_from_spec(
        cls, model_type: str, scale: str, *,
        num_ids: int, in_channels: int, conv_type: str | None = None,
    ) -> nn.Module:
        """Render model jsonnet -> inject runtime params -> build."""
        from graphids.config.constants import CONFIG_DIR, FAMILY_FOR_MODEL_TYPE
        from graphids.config.jsonnet import render
        from graphids.core.losses.build import build_loss

        family = FAMILY_FOR_MODEL_TYPE[model_type]
        model_cfg = render(
            CONFIG_DIR / "models" / "_expand.jsonnet",
            tla={"family": family, "model_type": model_type, "scale": scale},
        )
        init_args = dict(model_cfg["model"].get("init_args", {}))
        init_args["num_ids"] = num_ids
        init_args["in_channels"] = in_channels
        if conv_type is not None:
            init_args["conv_type"] = conv_type
        loss_fn = build_loss(model_type, init_args.pop("loss_config", None), distillation_config=None)
        if loss_fn is not None:
            init_args["loss_fn"] = loss_fn
        return cls.build_model(model_cfg["model"]["class_path"], init_args)

    @classmethod
    def build_model_from_config(cls, merged: dict[str, Any]) -> nn.Module:
        """Build model from rendered config with loss injection."""
        from graphids.core.losses.build import inject_loss_fn

        init_args = inject_loss_fn(
            merged["model"].get("init_args") or {},
            class_path=merged["model"]["class_path"],
        )
        return cls.build_model(merged["model"]["class_path"], init_args)

    @classmethod
    def build_datamodule(cls, merged: dict[str, Any]) -> Any:
        return cls.build_block(merged["data"])

    @classmethod
    def build_callbacks(cls, merged: dict[str, Any]) -> list:
        trainer_cfg = merged.get("trainer") or {}
        default_root_dir = trainer_cfg.get("default_root_dir")
        callbacks = []
        for entry in (trainer_cfg.get("callbacks") or []):
            if "ModelCheckpoint" in entry.get("class_path", "") and default_root_dir:
                entry = copy.deepcopy(entry)
                entry.setdefault("init_args", {})["dirpath"] = f"{default_root_dir}/{_CKPT_DIR}"
            callbacks.append(cls.build_block(entry))
        return callbacks

    @classmethod
    def build_loggers(cls, merged: dict[str, Any]) -> list | bool | None:
        logger_cfg = (merged.get("trainer") or {}).get("logger")
        if isinstance(logger_cfg, (list, dict)):
            entries = logger_cfg if isinstance(logger_cfg, list) else [logger_cfg]
            return [cls.build_block(e) for e in entries]
        return logger_cfg  # None or bool

    @classmethod
    def build_trainer(cls, merged: dict[str, Any]) -> Trainer:
        trainer_dict = dict(merged.get("trainer") or {})

        # Extract TrainerConfig fields
        cfg_kwargs = {}
        for key in list(trainer_dict):
            if key in _TRAINER_CONFIG_KEYS:
                cfg_kwargs[key] = trainer_dict.pop(key)
        # Remove consumed keys that aren't TrainerConfig fields
        trainer_dict.pop("callbacks", None)
        trainer_dict.pop("logger", None)

        config = TrainerConfig(**cfg_kwargs)
        callbacks = cls.build_callbacks(merged)
        logger = cls.build_loggers(merged)

        return Trainer(config=config, callbacks=callbacks, logger=logger)

    # -- composition --

    @classmethod
    def build_run(
        cls, rendered: dict[str, Any], *,
        validated: ValidatedConfig | None = None,
        seed_all: bool = True,
    ) -> InstantiatedRun:
        from graphids.config.schemas import validate_config

        merged = copy.deepcopy(rendered)
        if validated is None:
            validated = validate_config(merged)
        if seed_all:
            seed_everything(merged["seed_everything"])

        return InstantiatedRun(
            model=cls.build_model_from_config(merged),
            datamodule=cls.build_datamodule(merged),
            trainer=cls.build_trainer(merged),
            merged=merged,
        )


instantiate = Instantiator.build_run


def _init_kwargs(cls: type) -> set[str]:
    """Return the set of accepted keyword argument names for ``cls.__init__``."""
    import inspect

    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return set()
    return {
        name
        for name, p in sig.parameters.items()
        if name != "self" and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    }
