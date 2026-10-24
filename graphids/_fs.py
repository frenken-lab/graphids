"""NFS-safe filesystem helpers."""

from __future__ import annotations

import hashlib
import json
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


def _sha256_file(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Publish indented, sorted JSON with a same-directory atomic replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_save(obj: Any, path: Path) -> None:
    """Write a torch object atomically (temp + fsync + rename) with a
    ``<path>.sha256`` sidecar so corruption is detectable on load.
    Parent dir is fsynced after the rename so other NFS clients see
    the new ckpt without waiting for attribute-cache expiry.
    """
    import torch  # heavy — keep out of module scope so _fs stays import-light

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        torch.save(obj, str(tmp))
        fd = os.open(str(tmp), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        tmp.rename(path)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise
    path.with_suffix(path.suffix + ".sha256").write_text(_sha256_file(path) + "\n")
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
