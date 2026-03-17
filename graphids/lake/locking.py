"""GPFS-safe advisory file locking for cache writes.

Uses fcntl.flock() which is supported by GPFS (unlike POSIX fcntl locks
which have issues on some network filesystems). Prevents race conditions
when two users preprocess the same dataset simultaneously.
"""

from __future__ import annotations

import fcntl
import logging
from contextlib import contextmanager
from pathlib import Path

log = logging.getLogger(__name__)


@contextmanager
def cache_lock(cache_dir: Path, timeout_msg: str = ""):
    """Advisory file lock for cache directory writes.

    Creates a .lock file in the cache directory parent and holds an
    exclusive flock() on it for the duration of the context.

    GPFS supports flock() natively. On NFS, flock() is emulated via
    fcntl locks (Linux 2.6.12+), which is sufficient for our use case.

    Usage::

        with cache_lock(cache_dir):
            # safe to write to cache_dir
            save_collated(graphs, cache_file)
    """
    lock_dir = cache_dir.parent
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_file = lock_dir / f".{cache_dir.name}.lock"

    fd = None
    try:
        fd = open(lock_file, "w")
        log.debug("Acquiring lock: %s", lock_file)
        fcntl.flock(fd, fcntl.LOCK_EX)
        log.debug("Lock acquired: %s", lock_file)
        yield
    finally:
        if fd is not None:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
            fd.close()
            log.debug("Lock released: %s", lock_file)
