"""NFS-safe I/O primitives shared across dataset adapters."""

from __future__ import annotations

import fcntl
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

import torch


@contextmanager
def nfs_lock(lock_path: Path):
    """Advisory file lock safe on NFS."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = open(lock_path, "w")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        fd.close()


def atomic_save(data: list, path: Path) -> None:
    """torch.save with NFS-safe atomic write (tmpfile -> fsync -> rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        torch.save(data, tmp)
        os.fsync(fd)
        os.close(fd)
        fd = -1
        os.rename(tmp, path)
    except BaseException:
        if fd >= 0:
            os.close(fd)
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
