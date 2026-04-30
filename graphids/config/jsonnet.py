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
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def dotted_to_nested(overrides: Sequence[tuple[str, Any]] | None) -> dict[str, Any]:
    """Expand ``[(dotted.path, value), ...]`` into a nested dict.

    Output is fed to ``render(set_overrides=...)`` which passes it as the
    ``overrides`` ``std.extVar`` consumed by every ablation preset's
    ``std.mergePatch(...)`` apex. Single entry point for ``--set`` flag
    shaping.
    """
    out: dict[str, Any] = {}
    for key, typed_val in overrides or []:
        parts = key.split(".")
        cur = out
        for part in parts[:-1]:
            nxt = cur.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                cur[part] = nxt
            cur = nxt
        cur[parts[-1]] = typed_val
    return out


def render_with_flags(
    preset: str | Path,
    tla: Sequence[tuple[str, Any]] | None = None,
    set_: Sequence[tuple[str, Any]] | None = None,
) -> dict[str, Any]:
    """Render a preset from Typer-parsed ``--tla`` / ``--set`` flag pairs.

    Convenience wrapper that turns ``[(key, value), ...]`` lists (the shape
    Typer's ``parser=`` callback produces) into ``render()``'s ``tla`` dict
    + ``set_overrides`` nested dict in one call. Used by ``cli/training.py``
    and ``slurm/submit.py`` so the flag-list → render-input transform lives
    in one place.
    """
    return render(
        preset,
        tla=dict(tla or []) or None,
        set_overrides=dotted_to_nested(set_),
    )


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
    from graphids.config.settings import get_settings as _get_settings  # noqa: PLC0415

    tla_codes = {k: json.dumps(v) for k, v in tla.items()} if tla else {}
    ext_codes = {
        "run_root": json.dumps(_get_settings().run_root),
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
