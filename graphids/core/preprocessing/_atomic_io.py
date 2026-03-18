"""NFS-safe atomic file operations for cache persistence."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from ._dataset import save_collated

logger = logging.getLogger(__name__)


def atomic_rename(tmp: Path, final: Path, retries: int = 3) -> None:
    """Rename tmp -> final with retry for NFS visibility delays."""
    import time

    for attempt in range(retries):
        try:
            tmp.rename(final)
            return
        except OSError as e:
            if attempt < retries - 1:
                logger.warning("Rename attempt %d failed: %s. Retrying...", attempt + 1, e)
                time.sleep(1)
            else:
                raise


def atomic_save_collated(graphs: list, tmp_path: Path, target_path: Path) -> None:
    """Save collated graphs atomically: write to tmp, fsync, rename to target."""
    save_collated(graphs, tmp_path)
    with open(tmp_path, "rb") as f:
        os.fsync(f.fileno())
    atomic_rename(tmp_path, target_path)
