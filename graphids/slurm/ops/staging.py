"""Stage data from permanent storage to scratch/TMPDIR for fast job I/O.

Operation layer — argparse surface lives in ``graphids.commands.stage_data``.

Data flow: ESS (permanent NFS) → Scratch (GPFS, 90-day purge) → TMPDIR (per-job local SSD)

Smart caching: skips ESS→scratch copy if a marker file exists and the source
file count hasn't changed. Scratch 90-day purge deletes the marker, triggering
a fresh sync automatically.

Prints export statements for bash eval:
    eval $(python -m graphids stage-data --cache)
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from graphids.log import get_logger

log = get_logger(__name__)


def _count_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.rglob("*") if _.is_file())


def _needs_sync(src: Path, dst: Path) -> bool:
    marker = dst / ".staged_marker"
    if not dst.exists() or not marker.exists():
        return True
    try:
        staged_count = int(marker.read_text().strip())
    except (ValueError, OSError):
        return True
    src_count = _count_files(src)
    if src_count != staged_count:
        log.info("file_count_changed", source=src_count, staged=staged_count)
        return True
    return False


def _write_marker(src: Path, dst: Path) -> None:
    count = _count_files(src)
    (dst / ".staged_marker").write_text(str(count))


def _rsync(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    rc = os.system(f"rsync -a --info=progress2 '{src}/' '{dst}/'")
    if rc != 0:
        log.error("rsync_failed", src=str(src), dst=str(dst), exit_code=rc)
        sys.exit(1)


def stage_data(
    *,
    cache_only: bool = False,
    raw_only: bool = False,
    skip_tmpdir: bool = False,
    dataset: str = "",
) -> None:
    """Stage cache and/or raw data from ESS → scratch → TMPDIR.

    ``cache_only`` suppresses the raw tier, ``raw_only`` suppresses the
    cache tier (the two are mutually exclusive — set at most one). When
    ``skip_tmpdir`` is true, leaves data on scratch and exports point to
    the scratch tier. Prints ``export KEY=value`` lines on stdout for
    eval by the caller shell.
    """
    stage_raw = not cache_only
    stage_cache = not raw_only

    from graphids.config.settings import get_settings

    _s = get_settings()
    lake_root = _s.lake_root
    if _s.scratch is None:
        log.error("KD_GAT_SCRATCH not set. Source .env before running.")
        sys.exit(1)
    scratch = _s.scratch
    scratch_data = scratch / "kd-gat-data"
    tmpdir = os.environ.get("TMPDIR", "")

    # Primary: ESS lake root (has raw/ subdir). Fallback: KD_GAT_DATA_ROOT env var.
    if lake_root and Path(lake_root, "raw").is_dir():
        data_root = Path(lake_root)
        log.info("using_ess_lake", path=str(data_root))
    elif _s.data_root is not None:
        data_root = _s.data_root
    else:
        log.error(
            "No data source found. Set KD_GAT_LAKE_ROOT (with raw/ subdir) or KD_GAT_DATA_ROOT."
        )
        sys.exit(1)

    log.info(
        "staging_start",
        source=str(data_root),
        scratch=str(scratch_data),
        tmpdir=tmpdir or "<not set>",
        dataset=dataset or "all",
    )

    exports: dict[str, str] = {}

    # --- Step 1: ESS → Scratch (rsync, incremental) ---
    if stage_raw and (data_root / "raw").is_dir():
        src, dst = data_root / "raw", scratch_data / "raw"
        if _needs_sync(src, dst):
            log.info("staging_raw_to_scratch")
            _rsync(src, dst)
            _write_marker(src, dst)
        else:
            log.info("scratch_raw_fresh")

    if stage_cache and (data_root / "cache").is_dir():
        if dataset:
            src = data_root / "cache" / dataset
            dst = scratch_data / "cache" / dataset
        else:
            src = data_root / "cache"
            dst = scratch_data / "cache"

        if src.is_dir():
            if _needs_sync(src, dst):
                log.info("staging_cache_to_scratch", scope=dataset or "all")
                _rsync(src, dst)
                _write_marker(src, dst)
            else:
                log.info("scratch_cache_fresh")

    # --- Step 2: Scratch → TMPDIR (per-job local SSD) ---
    if not skip_tmpdir and tmpdir:
        tmpdir_data = Path(tmpdir) / "kd-gat-data"
        tmpdir_data.mkdir(parents=True, exist_ok=True)

        if stage_cache:
            if dataset:
                scratch_cache = scratch_data / "cache" / dataset
                tmpdir_cache = tmpdir_data / "cache" / dataset
            else:
                scratch_cache = scratch_data / "cache"
                tmpdir_cache = tmpdir_data / "cache"

            if scratch_cache.is_dir() and not tmpdir_cache.exists():
                log.info("staging_cache_to_tmpdir", scope=dataset or "all")
                tmpdir_cache.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(str(scratch_cache), str(tmpdir_cache))
            else:
                log.info("tmpdir_cache_exists")
            exports["KD_GAT_CACHE_ROOT"] = str(tmpdir_cache)

        if stage_raw and (scratch_data / "raw").is_dir():
            tmpdir_raw = tmpdir_data / "raw"
            if not tmpdir_raw.exists():
                log.info("staging_raw_to_tmpdir")
                shutil.copytree(str(scratch_data / "raw"), str(tmpdir_raw))
            else:
                log.info("tmpdir_raw_exists")
            exports["KD_GAT_DATA_ROOT"] = str(tmpdir_data)
    else:
        # No TMPDIR: use scratch as fastest tier
        if scratch_data.is_dir():
            exports["KD_GAT_DATA_ROOT"] = str(scratch_data)
            if dataset:
                exports["KD_GAT_CACHE_ROOT"] = str(scratch_data / "cache" / dataset)
            else:
                exports["KD_GAT_CACHE_ROOT"] = str(scratch_data / "cache")

    log.info("staging_complete", **exports)

    # Print export statements for bash eval
    for k, v in exports.items():
        print(f"export {k}={v}")
