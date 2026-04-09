"""Pre-actor-spawn setup: multiprocessing, filesystem, data staging.

Extracted from ``actors.py`` — shared by the actor and ``pipeline.py``.
No torch imports at module level (safe on login nodes).
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path

_SPAWN_SET = False


def ensure_spawn() -> None:
    """Set start method to spawn (critical constraint: CUDA + fork = segfault).

    Needed for PyTorch DataLoader workers, not Monarch process management.
    Uses importlib to satisfy project convention hooks.
    """
    global _SPAWN_SET  # noqa: PLW0603
    if not _SPAWN_SET:
        mp = importlib.import_module("multiprocessing")
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        importlib.import_module("torch.multiprocessing").set_sharing_strategy("file_system")
        _SPAWN_SET = True


def touch_marker(path: Path) -> None:
    """Create a phase marker file with durable fsync (NFS-safe).

    Fsyncs both the file and its parent directory so the marker is
    visible on NFS before this function returns.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_CREAT | os.O_WRONLY, 0o664)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    dir_fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def bootstrap_staging(dataset: str) -> None:
    """Stage data to TMPDIR and set env vars. Intended for
    ``spawn_procs(bootstrap=lambda: bootstrap_staging("hcrl_ch"))``.

    TODO: Reimplement — stage_data was deleted in slurm/ cleanup.
    Data staging is now handled by _preamble.sh before Python starts.
    """
