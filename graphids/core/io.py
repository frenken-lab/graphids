"""Filesystem I/O for run artifacts.

All NFS/GPFS writes live here:

- Rendered-config snapshots next to runs (``snapshot_config``)

Resolvers (catalog, checkpoint probes, identity parsing, lake-write
gate) live in ``graphids.config.paths``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from graphids.config.settings import LakeWriteError  # noqa: F401

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
# Rendered-config snapshot
# -----------------------------------------------------------------------------


def snapshot_config(rendered: dict[str, Any], run_dir: str | Path) -> None:
    """Write the rendered config to ``{run_dir}/config_snapshot.json``."""
    _atomic_write_text(
        Path(run_dir) / "config_snapshot.json",
        json.dumps(rendered, indent=2, sort_keys=True),
    )
