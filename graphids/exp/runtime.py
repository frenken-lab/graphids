"""Framework-independent construction for experiment and benchmark runners."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from numbers import Real
from typing import Any

from pydantic import BaseModel


def resolve_spec(spec: Any) -> Any:
    """Resolve a nested component/class-path spec without calling ``build()``."""
    if isinstance(spec, BaseModel):
        spec = spec.model_dump(mode="json")
    if isinstance(spec, list):
        return [resolve_spec(item) for item in spec]
    if not isinstance(spec, Mapping):
        return spec
    if "class_path" in spec:
        class_path = str(spec["class_path"])
        module_path, _, class_name = class_path.rpartition(".")
        cls = getattr(importlib.import_module(module_path), class_name)
        init_args = {
            k: resolve_spec(v) for k, v in dict(spec.get("init_args") or {}).items()
        }
        return cls(**init_args)

    if "type" in spec:
        match spec["type"]:
            case "can_event_dm":
                from graphids.data.config import CANEventDataConfig

                config_class = CANEventDataConfig
            case "event_graph_mamba":
                from graphids.models.config import EventGraphMambaConfig

                config_class = EventGraphMambaConfig
            case _:
                raise ValueError(f"unknown component type: {spec['type']!r}")
        return config_class.model_validate(
            {k: resolve_spec(v) for k, v in spec.items()}
        )

    return {k: resolve_spec(v) for k, v in spec.items()}


def build_component(spec: Any) -> Any:
    """Build a component config, or return an instantiated class-path object.

    ``build`` is the optional construction hook at this config boundary.
    Errors from the builder propagate unchanged.
    """
    resolved = resolve_spec(spec)
    builder = getattr(resolved, "build", None)
    return builder() if builder is not None else resolved


def _scalar_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    """Extract numeric scalars from Lightning metrics, ignoring vector metrics."""
    from torch import Tensor

    out = {}
    for key, value in metrics.items():
        if isinstance(value, Tensor):
            if value.numel() != 1:
                continue
            value = value.detach().cpu().item()
        if isinstance(value, Real):
            out[str(key)] = float(value)
    return out


def _compile_model_if_requested(model: Any, cfg: Mapping[str, Any]) -> None:
    compile_cfg = cfg.get("compile")
    if compile_cfg in (None, False):
        return
    if compile_cfg is True:
        compile_kwargs: dict[str, Any] = {}
    else:
        resolved = resolve_spec(compile_cfg)
        if not isinstance(resolved, Mapping):
            raise TypeError("compile config must resolve to a mapping")
        compile_kwargs = dict(resolved)
        if compile_kwargs.pop("enabled", True) is False:
            return
    model.compile(**compile_kwargs)


def seed_run(cfg: Mapping[str, Any]) -> None:
    import lightning.pytorch as pl

    seed = cfg.get("seed_everything", cfg.get("seed"))
    if seed is not None:
        pl.seed_everything(int(seed), workers=True)


def build_model(cfg: Mapping[str, Any], data: Any) -> Any:
    model = build_component(cfg["model"])
    # Custom class-path models may omit this event-data preparation hook.
    prepare = getattr(model, "prepare_from_datamodule", None)
    if prepare is not None:
        prepare(data)
    _compile_model_if_requested(model, cfg)
    return model


def trainer_components(cfg: Mapping[str, Any]) -> dict[str, Any]:
    trainer_cfg = resolve_spec(cfg.get("trainer", {}))
    if not isinstance(trainer_cfg, Mapping):
        raise TypeError("trainer config must resolve to a mapping of Trainer kwargs")
    kwargs = dict(trainer_cfg)
    specs = cfg.get("callbacks") or {}
    kwargs["callbacks"] = [
        build_component(spec)
        for spec in (specs.values() if isinstance(specs, Mapping) else specs)
    ]
    return kwargs
