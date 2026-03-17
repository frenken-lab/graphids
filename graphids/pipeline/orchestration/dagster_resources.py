"""Retry state helpers for Dagster SLURM orchestration.

Persists per-asset retry metadata (failure reason, node, checkpoint path)
to JSON files so that subsequent Dagster retry attempts can scale resources
appropriately (e.g., 2x memory after OOM).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from graphids.config.constants import PROJECT_ROOT

log = logging.getLogger(__name__)

_RETRY_STATE_DIR = Path(PROJECT_ROOT) / "slurm_logs" / "dagster_retry"


def save_retry_state(
    asset_key: str,
    reason: str,
    node: str | None = None,
    ckpt_path: str | None = None,
) -> None:
    """Write retry metadata to slurm_logs/dagster_retry/{asset_key}.json."""
    _RETRY_STATE_DIR.mkdir(parents=True, exist_ok=True)
    state = {
        "reason": reason,
        "node": node,
        "ckpt_path": ckpt_path,
    }
    path = _RETRY_STATE_DIR / f"{asset_key}.json"
    path.write_text(json.dumps(state, indent=2))
    log.info("Saved retry state for %s: %s", asset_key, state)


def load_retry_state(asset_key: str) -> dict | None:
    """Read retry metadata from previous attempt, if any."""
    path = _RETRY_STATE_DIR / f"{asset_key}.json"
    if not path.exists():
        return None
    try:
        state = json.loads(path.read_text())
        log.info("Loaded retry state for %s: %s", asset_key, state)
        return state
    except (json.JSONDecodeError, OSError) as e:
        log.warning("Failed to load retry state for %s: %s", asset_key, e)
        return None


def clear_retry_state(asset_key: str) -> None:
    """Remove retry state after successful completion."""
    path = _RETRY_STATE_DIR / f"{asset_key}.json"
    if path.exists():
        path.unlink()
        log.debug("Cleared retry state for %s", asset_key)
