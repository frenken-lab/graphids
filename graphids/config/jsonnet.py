"""Render jsonnet plans/specs via gojsonnet with graphids native callbacks.

Single render path. Plans are evaluated on the login node — the rendered
JSON array travels to compute nodes via the blueprint, never the jsonnet.

Bridge surface (graphids ↔ jsonnet):
    tla_codes:        per-call top-level args (dataset, seed, scale, ...)
    native_callbacks: paths.run_dir, paths.best_ckpt, paths.states_dir
        → all three resolve to :mod:`graphids.config.catalog` (single source).

`run_root` is read from `$GRAPHIDS_RUN_ROOT` inside the catalog functions.
There is no default — fail-fast on missing config.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import _gojsonnet  # type: ignore[import-not-found]

from graphids.config import catalog

# gojsonnet expects `(arg_names_tuple, fn)` per native. Catalog functions read
# `$GRAPHIDS_RUN_ROOT` themselves, so no closure is needed.
_NATIVES: dict[str, tuple[tuple[str, ...], Any]] = {
    "paths.run_dir": (
        ("dataset", "group", "variant", "seed"),
        catalog.run_dir,
    ),
    "paths.best_ckpt": (
        ("dataset", "group", "variant", "seed"),
        catalog.best_ckpt,
    ),
    "paths.states_dir": (
        ("dataset", "seed"),
        catalog.states_dir,
    ),
}


def render(path: str | Path, *, tla: dict[str, Any] | None = None) -> Any:
    """Render `path` (plan or spec jsonnet), return parsed JSON value.

    `tla` keys are JSON-encoded and passed as `tla_codes` so values may be
    primitives, lists, or objects.
    """
    # Resolve run_root upfront so a missing env var raises the friendly
    # `RuntimeError` from catalog._run_root() — once gojsonnet enters its
    # native-callback path, exceptions get wrapped as opaque "code: 0" errors.
    catalog._run_root()
    tla_codes = {k: json.dumps(v) for k, v in (tla or {}).items()}
    rendered = _gojsonnet.evaluate_file(
        str(path),
        tla_codes=tla_codes,
        native_callbacks=_NATIVES,
    )
    return json.loads(rendered)
