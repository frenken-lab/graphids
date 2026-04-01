"""Programmatic training entrypoint — direct instantiation, no LightningCLI.

Pipeline runs resolve the YAML config chain into a single dict, write a
snapshot for reproducibility, then instantiate model/data/trainer directly.
"""

from __future__ import annotations

import importlib
from pathlib import Path, PurePosixPath
from typing import Any

from graphids.cli import resolve_configs
from graphids.config import CKPT_SUBPATH, WANDB_WRITE_DIR
from graphids.config.yaml_utils import write_yaml
from graphids.core.contracts import TrainingContract, TrainingSpec

# Values copied between namespaces before instantiation.
# Replaces GraphIDSCLI.add_arguments_to_parser link_arguments.
_LINK_TARGETS: list[tuple[str, str]] = [
    ("data.init_args.dataset", "model.init_args.dataset"),
    ("data.init_args.lake_root", "model.init_args.lake_root"),
    ("seed_everything", "model.init_args.seed"),
    ("seed_everything", "data.init_args.seed"),
    ("model.init_args.conv_type", "data.init_args.conv_type"),
    ("model.init_args.heads", "data.init_args.heads"),
]


def _import_class(class_path: str) -> type:
    module_path, class_name = class_path.rsplit(".", 1)
    return getattr(importlib.import_module(module_path), class_name)


def _get_dotted(d: dict, key: str, default=None):
    cur = d
    for part in key.split("."):
        if not isinstance(cur, dict):
            return default
        cur = cur.get(part, default)
    return cur


def _set_dotted(d: dict, key: str, value) -> None:
    parts = key.split(".")
    for part in parts[:-1]:
        d = d.setdefault(part, {})
    d[parts[-1]] = value


def _apply_links(resolved: dict) -> None:
    """Propagate linked values (replaces GraphIDSCLI.link_arguments)."""
    for src, tgt in _LINK_TARGETS:
        val = _get_dotted(resolved, src)
        if val is not None:
            _set_dotted(resolved, tgt, val)


def _patch_paths(resolved: dict) -> None:
    """Patch logger save_dirs + checkpoint dirpath.

    Replaces GraphIDSCLI.before_instantiate_classes.
    """
    root_dir = resolved.get("trainer", {}).get("default_root_dir")
    if not root_dir:
        return
    ckpt_dir = f"{root_dir}/{PurePosixPath(CKPT_SUBPATH).parent}"
    if "checkpoint" in resolved:
        resolved["checkpoint"]["dirpath"] = ckpt_dir
    loggers = resolved.get("trainer", {}).get("logger")
    if isinstance(loggers, list):
        for lg in loggers:
            cp = lg.get("class_path", "")
            if "WandbLogger" in cp:
                lg.setdefault("init_args", {})["save_dir"] = WANDB_WRITE_DIR
            elif "CSVLogger" in cp:
                lg.setdefault("init_args", {})["save_dir"] = root_dir


def run_training_from_resolved(resolved: dict) -> None:
    """Instantiate model, data, trainer and fit. No LightningCLI."""
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

    _apply_links(resolved)
    _patch_paths(resolved)

    # Seed
    seed = resolved.pop("seed_everything", 42)
    pl.seed_everything(seed)

    # Model
    model_cfg = resolved["model"]
    model = _import_class(model_cfg["class_path"])(**model_cfg.get("init_args", {}))

    # Data
    data_cfg = resolved["data"]
    data = _import_class(data_cfg["class_path"])(**data_cfg.get("init_args", {}))

    # Callbacks — forced (top-level) + trainer-configured
    callbacks = []
    if "checkpoint" in resolved:
        callbacks.append(ModelCheckpoint(**resolved.pop("checkpoint")))
    if "early_stopping" in resolved:
        callbacks.append(EarlyStopping(**resolved.pop("early_stopping")))
    trainer_cfg = dict(resolved.get("trainer", {}))
    for cb_cfg in trainer_cfg.pop("callbacks", None) or []:
        callbacks.append(
            _import_class(cb_cfg["class_path"])(**cb_cfg.get("init_args", {}))
        )

    # Loggers
    loggers = []
    for lg_cfg in trainer_cfg.pop("logger", None) or []:
        if isinstance(lg_cfg, dict) and "class_path" in lg_cfg:
            loggers.append(
                _import_class(lg_cfg["class_path"])(**lg_cfg.get("init_args", {}))
            )

    # Trainer
    trainer_cfg["callbacks"] = callbacks
    if loggers:
        trainer_cfg["logger"] = loggers
    trainer = pl.Trainer(**trainer_cfg)

    # Fit
    ckpt_path = resolved.pop("ckpt_path", None)
    trainer.fit(model, datamodule=data, ckpt_path=ckpt_path)


def run_training_from_spec(spec: TrainingSpec) -> None:
    """Resolve config chain from spec, write snapshot, run training."""
    overrides = TrainingContract.to_override_dict(spec)
    resolved = resolve_configs(spec.config_files, overrides)

    # Write snapshot for reproducibility
    rd = Path(spec.run_dir)
    rd.mkdir(parents=True, exist_ok=True)
    write_yaml(resolved, rd / "config_snapshot.yaml")

    run_training_from_resolved(resolved)


def run_training_from_payload(payload: dict[str, Any]) -> None:
    run_training_from_spec(TrainingContract.from_envelope(payload))
