"""Jsonnet config rendering via the ``_jsonnet`` C bindings.

Evaluates a ``.jsonnet`` file with top-level arguments and returns the
parsed JSON as a dict. TLAs are passed through ``tla_codes`` so jsonnet
receives real typed values (ints stay ints, bools stay bools, ``None``
becomes jsonnet ``null``).

Project-global state — ``run_root``, the same value as ``GRAPHIDS_RUN_ROOT``
(per-user run/checkpoint root, distinct from the shared ``LAKE_ROOT``) —
is injected once via ``ext_codes`` instead of being a TLA default
duplicated across every preset.

Path derivation (``run_dir``, ``vgae_ckpt``, ``states_dir``) is exposed via
``native_callbacks`` so jsonnet presets call into the same Python
:mod:`graphids.config.paths` module that ``slurm/dag.py`` uses. One
implementation, no drift between Python and jsonnet.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def render(
    jsonnet_path: str | Path,
    tla: dict[str, Any] | None = None,
    set_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Render a ``.jsonnet`` file with top-level arguments.

    TLAs are JSON-serialized via ``json.dumps`` so jsonnet receives real
    typed values (ints stay ints, bools stay bools, ``None`` → jsonnet ``null``).
    ``run_root`` is exposed as ``std.extVar('run_root')`` from settings.
    Path-derivation natives are registered as
    ``std.native('paths.run_dir')(dataset, group, variant, seed)`` etc.
    ``set_overrides`` (already-nested form of ``--set a.b.c=v`` flags) flows
    through as ``std.extVar('overrides')`` and is applied by ``std.mergePatch``
    at each ablation preset's apex.
    Raises ``RuntimeError`` on evaluation failure or non-object output.
    """
    import _jsonnet  # noqa: PLC0415  — deferred so module import is safe without the binding

    from graphids.config import paths  # noqa: PLC0415
    from graphids.config.constants import RUN_ROOT  # noqa: PLC0415

    tla_codes = {k: json.dumps(v) for k, v in tla.items()} if tla else {}
    ext_codes = {
        "run_root": json.dumps(RUN_ROOT),
        "overrides": json.dumps(set_overrides or {}),
    }
    # Single-letter parameter names work around an upstream `_jsonnet` C-binding
    # bug: when a multi-letter callback param name (e.g. "seed") is the LAST
    # positional arg in a call where another arg is also a local-variable
    # reference (typical of presets passing `(dataset, ..., seed)`), the binding
    # raises ``binding parameter a second time``. Single-letter names dodge it.
    native_callbacks = {
        "paths.run_dir": (("a", "b", "c", "d"), paths.run_dir),
        "paths.best_ckpt": (("a", "b", "c", "d"), paths.best_ckpt),
        "paths.vgae_ckpt": (("a", "b"), paths.vgae_ckpt),
        "paths.states_dir": (("a", "b"), paths.states_dir),
    }
    try:
        raw = _jsonnet.evaluate_file(
            str(jsonnet_path),
            tla_codes=tla_codes,
            ext_codes=ext_codes,
            native_callbacks=native_callbacks,
        )
    except RuntimeError as exc:
        raise RuntimeError(f"jsonnet failed for {jsonnet_path}:\n{exc}") from exc

    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise RuntimeError(
            f"jsonnet output for {jsonnet_path} is {type(parsed).__name__}, expected object"
        )
    return parsed
