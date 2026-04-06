"""Auto-generate Pydantic schemas from ``__init__`` signatures.

Single public helper: ``schema_for(cls)`` introspects a class's
``__init__`` and returns a Pydantic ``BaseModel`` subclass mirroring its
keyword arguments. Used by ``core/models/schemas.py`` and
``core/data/schemas.py`` to keep per-model / per-datamodule schemas in
lockstep with their target classes automatically — no drift risk, no
hand-maintained field lists.

Required kwargs (no default) become required schema fields; kwargs with
defaults become optional fields with the declared default. Positional
``*args`` and keyword-only ``**kwargs`` parameters are skipped.

If a specific class needs enum constraints, range checks, or cross-field
validation beyond what the type hints declare, subclass the auto-generated
schema at the call site and attach ``@field_validator`` / ``@model_validator``.
"""

from __future__ import annotations

import inspect
from typing import Any, get_type_hints

from pydantic import BaseModel, ConfigDict, create_model


def schema_for(cls: type, *, name: str | None = None) -> type[BaseModel]:
    """Build a Pydantic schema mirroring ``cls.__init__`` kwargs.

    Parameters
    ----------
    cls:
        Class whose ``__init__`` signature becomes the schema shape.
        ``inspect.signature`` walks the MRO, so subclasses that inherit
        a parent's ``__init__`` produce the same schema as the parent.
    name:
        Optional schema class name. Defaults to ``f"{cls.__name__}Config"``.
    """
    sig = inspect.signature(cls.__init__)

    # ``get_type_hints`` resolves PEP 563 string annotations against the
    # defining module's globals. Fall back to raw ``__annotations__`` if
    # any forward reference can't be resolved (bad type import, etc.).
    try:
        hints = get_type_hints(cls.__init__, include_extras=True)
    except NameError:
        hints = getattr(cls.__init__, "__annotations__", {})

    fields: dict[str, tuple[Any, Any]] = {}
    for param_name, p in sig.parameters.items():
        if param_name == "self" or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        annotation = hints.get(param_name, Any)
        default = ... if p.default is p.empty else p.default
        fields[param_name] = (annotation, default)

    return create_model(
        name or f"{cls.__name__}Config",
        __config__=ConfigDict(extra="forbid", arbitrary_types_allowed=True),
        **fields,
    )
