"""Jsonnet config rendering via the ``_jsonnet`` C bindings.

Evaluates a ``.jsonnet`` file with top-level arguments and returns the
parsed JSON as a dict.  TLAs are passed through ``tla_codes`` so jsonnet
receives real typed values (ints stay ints, bools stay bools, ``None``
becomes jsonnet ``null``).

No torch dep, no package-level side effects — safe to import on login
nodes even without ``_jsonnet`` installed (the C binding is imported
lazily inside ``render``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JsonnetError(RuntimeError):
    """Raised when jsonnet evaluation fails."""


def render(
    jsonnet_path: str | Path,
    tla: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Render a ``.jsonnet`` file with top-level arguments.

    TLAs are JSON-serialized via ``json.dumps`` so jsonnet receives real
    typed values (ints stay ints, bools stay bools, ``None`` → jsonnet ``null``).
    Raises ``JsonnetError`` on evaluation failure or non-object output.
    """
    import _jsonnet  # noqa: PLC0415  — deferred so module import is safe without the binding

    tla_codes = {k: json.dumps(v) for k, v in tla.items()} if tla else {}
    try:
        raw = _jsonnet.evaluate_file(str(jsonnet_path), tla_codes=tla_codes)
    except RuntimeError as exc:
        raise JsonnetError(f"jsonnet failed for {jsonnet_path}:\n{exc}") from exc

    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise JsonnetError(
            f"jsonnet output for {jsonnet_path} is {type(parsed).__name__}, expected object"
        )
    return parsed
