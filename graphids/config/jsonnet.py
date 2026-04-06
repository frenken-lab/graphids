"""Jsonnet config rendering via the ``_jsonnet`` C bindings.

Evaluates a ``.jsonnet`` file with top-level arguments and returns the
parsed JSON as a dict.  TLAs are passed through ``tla_codes`` so jsonnet
receives real typed values (ints stay ints, bools stay bools, ``None``
becomes jsonnet ``null``).

No torch dep, no package-level side effects — safe to import on login nodes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import _jsonnet


class JsonnetError(RuntimeError):
    """Raised when jsonnet evaluation fails."""


def render_config(
    jsonnet_path: str | Path,
    tla: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Render a ``.jsonnet`` file with top-level arguments.

    Parameters
    ----------
    jsonnet_path:
        Path to a ``.jsonnet`` file.  May be a top-level function of TLAs
        or a plain object.
    tla:
        Top-level argument dict.  Keys must match the jsonnet function's
        parameter names; values are JSON-serialized via ``json.dumps``.

    Returns
    -------
    dict[str, Any]
        Parsed JSON output.  Always a dict — jsonnet programs that emit
        arrays or scalars raise ``JsonnetError``.
    """
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


def render(jsonnet_path: str | Path, tla: dict[str, Any] | None = None) -> dict[str, Any]:
    """Short alias to render a Jsonnet config."""
    return render_config(jsonnet_path, tla)
