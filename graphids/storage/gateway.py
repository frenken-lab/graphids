"""Transport layer. Domain-ignorant. Coordinates → path + NFS-safe I/O.

Follows Metaflow DataStoreStorage interface: bytes in/out, path resolution,
backend-swappable. Atomic writes follow google/renameio pattern.
Locking absorbs graphids/core/preprocessing/_locking.py.

No imports from graphids.config, graphids.pipeline, or graphids.core.
"""

from __future__ import annotations

import fcntl
import json
import structlog
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from .paths import lake_run_dir

if TYPE_CHECKING:
    from collections.abc import Generator

log = structlog.get_logger()


class StorageGateway:
    """Transport layer. Domain-ignorant. Coordinates → path + NFS-safe I/O.

    Dual-init: accepts either a PipelineConfig object (via ``cfg``) or raw
    identity coordinates (``lake_root``, ``dataset``, ``model_type``, ``scale``).
    Both modes delegate to ``lake_run_dir()`` for path derivation.
    """

    def __init__(
        self,
        *,
        cfg=None,
        lake_root: str | Path | None = None,
        dataset: str | None = None,
        model_type: str | None = None,
        scale: str | None = None,
        auxiliaries: str = "none",
        seed: int = 42,
        production: bool = False,
    ):
        if cfg is not None:
            self._lake_root = cfg.lake_root
            self._dataset = cfg.dataset
            self._model_type = cfg.model_type
            self._scale = cfg.scale
            self._aux = cfg.auxiliaries[0].type if cfg.auxiliaries else ""
            self._seed = cfg.seed
            self._production = cfg.production
        elif lake_root is not None and dataset is not None and model_type is not None and scale is not None:
            self._lake_root = str(lake_root)
            self._dataset = dataset
            self._model_type = model_type
            self._scale = scale
            self._aux = auxiliaries if auxiliaries != "none" else ""
            self._seed = seed
            self._production = production
        else:
            raise ValueError(
                "StorageGateway requires either cfg= or "
                "(lake_root=, dataset=, model_type=, scale=)"
            )

    # ------------------------------------------------------------------
    # Path resolution
    # ------------------------------------------------------------------

    def resolve(
        self,
        stage: str,
        name: str | None = None,
        model_type: str | None = None,
    ) -> Path:
        """Logical coordinates → physical path.

        If ``model_type`` differs from this gateway's model_type, derives
        the cross-model path (e.g. loading VGAE checkpoint from GAT config).
        """
        mt = model_type or self._model_type
        path = lake_run_dir(
            lake_root=self._lake_root,
            dataset=self._dataset,
            model_type=mt,
            scale=self._scale,
            stage=stage,
            aux=self._aux,
            seed=self._seed,
            production=self._production,
        )
        if name is not None:
            return path / name
        return path

    def exists(
        self,
        stage: str,
        name: str,
        model_type: str | None = None,
    ) -> bool:
        """Check if an artifact exists on the filesystem."""
        return self.resolve(stage, name, model_type=model_type).exists()

    def require(
        self,
        stage: str,
        name: str,
        model_type: str | None = None,
    ) -> Path:
        """Get artifact path or raise FileNotFoundError."""
        path = self.resolve(stage, name, model_type=model_type)
        if not path.exists():
            raise FileNotFoundError(f"Artifact not found: {path}")
        return path

    def ensure_dir(self, stage: str) -> Path:
        """Create stage directory, return path."""
        path = self.resolve(stage)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def list_artifacts(self, stage: str) -> list[str]:
        """List artifact filenames in a stage directory."""
        sdir = self.resolve(stage)
        if not sdir.exists():
            return []
        return [f.name for f in sdir.iterdir() if f.is_file()]

    # ------------------------------------------------------------------
    # NFS-safe I/O (atomic writes via tmpfile + fsync + rename)
    # ------------------------------------------------------------------

    def _atomic_write(self, path: Path, data: bytes) -> None:
        """Write bytes atomically: tmpfile in same dir → fsync → rename."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = tempfile.NamedTemporaryFile(
            dir=path.parent, delete=False, suffix=".tmp"
        )
        try:
            fd.write(data)
            fd.flush()
            os.fsync(fd.fileno())
            fd.close()
            _atomic_rename(Path(fd.name), path)
        except BaseException:
            fd.close()
            Path(fd.name).unlink(missing_ok=True)
            raise

    def write_bytes(self, path: Path, data: bytes) -> None:
        """Atomic write of raw bytes."""
        self._atomic_write(path, data)

    def read_bytes(self, path: Path) -> bytes:
        """Read raw bytes from a file."""
        return path.read_bytes()

    def write_json(self, path: Path, data: dict) -> None:
        """Atomic write of JSON data."""
        self._atomic_write(path, json.dumps(data, indent=2).encode())

    def read_json(self, path: Path) -> dict:
        """Read and parse a JSON file."""
        return json.loads(path.read_bytes())

    # ------------------------------------------------------------------
    # File locking (absorbed from core/preprocessing/_locking.py)
    # ------------------------------------------------------------------

    @contextmanager
    def lock(self, path: Path) -> Generator[None, None, None]:
        """Advisory file lock for a directory or file.

        Creates a .lock file in the path's parent and holds an exclusive
        flock() on it for the duration of the context.

        GPFS supports flock() natively. On NFS, flock() is emulated via
        fcntl locks (Linux 2.6.12+).
        """
        lock_dir = path.parent if path.is_file() or not path.exists() else path.parent
        # For directories, put the lock file in the parent
        if path.is_dir():
            lock_dir = path.parent
            lock_name = f".{path.name}.lock"
        else:
            lock_dir = path.parent
            lock_name = f".{path.name}.lock"

        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_file = lock_dir / lock_name

        fd = None
        try:
            fd = open(lock_file, "w")
            log.debug("acquiring_lock", lock_file=str(lock_file))
            fcntl.flock(fd, fcntl.LOCK_EX)
            log.debug("lock_acquired", lock_file=str(lock_file))
            yield
        finally:
            if fd is not None:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                except OSError:
                    pass
                fd.close()
                log.debug("lock_released", lock_file=str(lock_file))


# ---------------------------------------------------------------------------
# Module-level helpers (no domain imports)
# ---------------------------------------------------------------------------


def _atomic_rename(tmp: Path, final: Path, retries: int = 3) -> None:
    """Rename tmp -> final with retry for NFS visibility delays."""
    for attempt in range(retries):
        try:
            tmp.rename(final)
            return
        except OSError as e:
            if attempt < retries - 1:
                log.warning("rename_retry", attempt=attempt + 1, error=str(e))
                time.sleep(1)
            else:
                raise
