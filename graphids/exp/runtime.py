"""Framework-independent construction for experiment and benchmark runners."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from inspect import signature
from typing import Any


def resolve_spec(spec: Any) -> Any:
    """Resolve a nested primitive/class-path spec without calling ``build()``."""
    if hasattr(spec, "model_dump") and not isinstance(spec, Mapping):
        spec = spec.model_dump(mode="json")
    if isinstance(spec, list):
        return [resolve_spec(item) for item in spec]
    if not isinstance(spec, Mapping):
        return spec
    from graphids.migration import reject_retired

    reject_retired(dict(spec))

    if "class_path" in spec:
        class_path = str(spec["class_path"])
        module_path, _, class_name = class_path.rpartition(".")
        cls = getattr(importlib.import_module(module_path), class_name)
        init_args = {
            k: resolve_spec(v) for k, v in dict(spec.get("init_args") or {}).items()
        }
        return cls(**init_args)

    if "type" in spec:
        from graphids import primitives as primitive_mod

        factory = getattr(primitive_mod, str(spec["type"]), None)
        if callable(factory):
            kwargs = {k: resolve_spec(v) for k, v in spec.items() if k != "type"}
            return factory(**kwargs)

    return {k: resolve_spec(v) for k, v in spec.items()}


def build_component(spec: Any, **build_kwargs: Any) -> Any:
    """Resolve a spec and call ``build()`` when the resolved object supports it."""
    resolved = resolve_spec(spec)
    builder = getattr(resolved, "build", None)
    if callable(builder):
        if build_kwargs:
            params = signature(builder).parameters
            filtered = {k: v for k, v in build_kwargs.items() if k in params}
            try:
                return builder(**filtered)
            except TypeError:
                pass
        return builder()
    return resolved


def _scalar_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in metrics.items():
        item = value
        detach = getattr(item, "detach", None)
        if callable(detach):
            item = detach()
        cpu = getattr(item, "cpu", None)
        if callable(cpu):
            item = cpu()
        scalar = getattr(item, "item", None)
        if callable(scalar):
            try:
                out[str(key)] = float(scalar())
                continue
            except (TypeError, ValueError, RuntimeError):
                continue
        if isinstance(item, int | float):
            out[str(key)] = float(item)
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
    compile_method = getattr(model, "compile", None)
    if not callable(compile_method):
        raise TypeError(
            f"{type(model).__name__} does not support LightningModule.compile()"
        )
    compile_method(**compile_kwargs)


def seed_run(cfg: Mapping[str, Any]) -> None:
    import lightning.pytorch as pl

    seed = cfg.get("seed_everything", cfg.get("seed"))
    if seed is not None:
        pl.seed_everything(int(seed), workers=True)


def build_model(cfg: Mapping[str, Any], data: Any) -> Any:
    loss_fn = (
        build_component(cfg["loss_fn"]) if cfg.get("loss_fn") is not None else None
    )
    model_spec = resolve_spec(cfg["model"])
    if hasattr(model_spec, "build"):
        model = (
            model_spec.build(loss_fn=loss_fn)
            if loss_fn is not None
            else model_spec.build()
        )
    else:
        model = model_spec
    if hasattr(model, "prepare_from_datamodule"):
        model.prepare_from_datamodule(data)
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
