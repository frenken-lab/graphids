"""NFS-safe filesystem helpers."""

from __future__ import annotations

import os
from pathlib import Path


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
