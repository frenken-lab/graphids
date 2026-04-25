"""NFS-safe filesystem helpers."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any


def _fsync_dir(directory: Path) -> None:
    """Fsync a directory so a child create/rename is visible to other NFS clients."""
    fd = os.open(str(directory), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


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
    _fsync_dir(path.parent)


def _sha256_file(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def atomic_save(obj: Any, path: Path) -> None:
    """Write a torch object atomically (temp + fsync + rename) with a
    ``<path>.sha256`` sidecar so corruption is detectable on load.
    Parent dir is fsynced after the rename so other NFS clients see
    the new ckpt without waiting for attribute-cache expiry.
    """
    import torch  # heavy — keep out of module scope so _fs stays import-light

    tmp = path.with_suffix(".tmp")
    torch.save(obj, str(tmp))
    fd = os.open(str(tmp), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    tmp.rename(path)
    path.with_suffix(path.suffix + ".sha256").write_text(_sha256_file(path) + "\n")
    _fsync_dir(path.parent)


def atomic_write_text(path: Path, content: str) -> None:
    """Write ``content`` to ``path`` atomically (tempfile + fsync + rename).

    NFS-safe: the temp file is fsynced before rename, and the parent dir
    is fsynced after rename so other clients see the new file without
    waiting for attribute-cache expiry. Use for JSON / other text payloads;
    ``atomic_save`` is the torch-binary equivalent.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.rename(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    _fsync_dir(path.parent)


def atomic_load(path: Path | str, **kwargs: Any) -> Any:
    """``torch.load(path, **kwargs)`` with SHA256 sidecar verify.

    If ``<path>.sha256`` exists (written by ``atomic_save``), the file
    is hashed via ``hashlib.file_digest`` and compared before load —
    mismatch raises. Old checkpoints without sidecars load unverified
    for forward compat with pre-sidecar runs.
    """
    import torch

    p = Path(path)
    sidecar = p.with_suffix(p.suffix + ".sha256")
    if sidecar.exists():
        actual = _sha256_file(p)
        expected = sidecar.read_text().strip()
        if actual != expected:
            raise RuntimeError(
                f"sha256 mismatch for {p}: got {actual[:16]}…, expected {expected[:16]}…"
            )
    return torch.load(str(p), **kwargs)
