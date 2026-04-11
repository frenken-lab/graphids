"""Reflection helpers for class-path instantiation.

Used by ``orchestrate.instantiate`` (to build the full trainer stack)
and ``core.models.factory`` (for ad-hoc model instantiation from a
``(model_type, scale)`` spec). One source of truth so the two code
paths can't diverge on VAR_KEYWORD/VAR_POSITIONAL edge cases.
"""

from __future__ import annotations

import importlib
import inspect
from functools import lru_cache
from typing import Any


def import_class(class_path: str) -> type:
    """Resolve a dotted ``module.ClassName`` path to the class object."""
    module_name, _, cls_name = class_path.rpartition(".")
    if not module_name:
        raise ValueError(f"class_path must be dotted: {class_path!r}")
    mod = importlib.import_module(module_name)
    try:
        return getattr(mod, cls_name)
    except AttributeError as e:
        raise ImportError(f"{cls_name!r} not found in {module_name!r}") from e


@lru_cache(maxsize=256)
def _accepted_kwargs(cls: type) -> frozenset[str] | None:
    """Return the set of kwargs ``cls.__init__`` accepts, or None for **kwargs."""
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return None
    if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
        return None
    return frozenset(
        name for name, p in sig.parameters.items()
        if name != "self" and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    )


def filter_kwargs(cls: type, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop kwargs whose names aren't in ``cls.__init__``'s signature.

    Returns ``kwargs`` unchanged if the signature can't be inspected or
    accepts ``**kwargs``. Cached per-class so re-instantiation is cheap.
    """
    accepted = _accepted_kwargs(cls)
    if accepted is None:
        return kwargs
    return {k: v for k, v in kwargs.items() if k in accepted}
