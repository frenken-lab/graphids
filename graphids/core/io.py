"""Filesystem I/O for run artifacts.

All NFS/GPFS writes live here:

- Run record sidecars (``write_run_record`` / ``read_run_record``)
- Rendered-config snapshots next to runs (``snapshot_config``)

Resolvers (catalog, checkpoint probes, identity parsing, lake-write
gate) live in ``graphids.config.paths``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from graphids.config.constants import RUN_RECORD_FILENAME
from graphids.config.paths import require_lake_write

if TYPE_CHECKING:
    from graphids.core.run_record import RunRecord


# Re-export so existing ``from graphids.core.io import ...`` callers
# that haven't been updated yet don't break.  Remove once all consumers
# point at ``graphids.config.paths``.
from graphids.config.paths import (  # noqa: E402, F401
    LakeWriteError,
    catalog_path,
    dataset_names,
    load_catalog,
    parse_identity_from_run_dir,
    resolve_checkpoint,
)

# -----------------------------------------------------------------------------
# Atomic text write (NFS/GPFS-safe: temp + fsync + rename)
# -----------------------------------------------------------------------------


def _atomic_write_text(path: Path, data: str) -> None:
    """Write text to ``path`` atomically.

    Guarantees ``path`` either contains the full new content or is
    unchanged. Leaves no partial files on crash.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp.open("w") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.rename(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


# -----------------------------------------------------------------------------
# Run record sidecar (JSON)
# -----------------------------------------------------------------------------


def write_run_record(record: RunRecord, run_dir: Path) -> Path:
    """Write run_record.json atomically."""
    require_lake_write()
    path = run_dir / RUN_RECORD_FILENAME
    _atomic_write_text(path, record.model_dump_json(indent=2))
    return path


def read_run_record(run_dir: Path) -> RunRecord | None:
    """Read run_record.json if it exists, else None."""
    from graphids.core.run_record import RunRecord  # lazy: avoid cycle

    path = run_dir / RUN_RECORD_FILENAME
    if not path.exists():
        return None
    return RunRecord.model_validate_json(path.read_text())


# -----------------------------------------------------------------------------
# Rendered-config snapshot
# -----------------------------------------------------------------------------


def snapshot_config(rendered: dict[str, Any], run_dir: str | Path) -> None:
    """Write the rendered config to ``{run_dir}/config_snapshot.json``."""
    _atomic_write_text(
        Path(run_dir) / "config_snapshot.json",
        json.dumps(rendered, indent=2, sort_keys=True),
    )
