"""Pipeline state persistence with atomic writes.

Factored from sweep_pipeline.py. Provides JSON-based state tracking for
both the sweep pipeline and the SLURM coordinator. State files survive
process restarts — the coordinator can resume from where it left off.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

log = logging.getLogger(__name__)

StageStatus = Literal[
    "pending",
    "submitted",
    "running",
    "completed",
    "failed",
    "retry_pending",
    "abandoned",
    "paused",
]


def load_state(path: Path) -> dict[str, Any]:
    """Load pipeline state from JSON file. Returns empty state if missing."""
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_state(state: dict[str, Any], path: Path) -> None:
    """Atomic write: tmpfile + os.replace to prevent corruption on crash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def update_stage(
    state: dict[str, Any],
    stage_key: str,
    status: StageStatus,
    path: Path,
    **extra: Any,
) -> None:
    """Update a stage's status and persist immediately."""
    stages = state.setdefault("stages", {})
    if stage_key not in stages:
        stages[stage_key] = {}
    stages[stage_key]["status"] = status
    stages[stage_key]["updated"] = datetime.now(UTC).isoformat()
    stages[stage_key].update(extra)
    save_state(state, path)


def now_iso() -> str:
    return datetime.now(UTC).isoformat()
