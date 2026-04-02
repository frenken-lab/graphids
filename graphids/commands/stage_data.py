"""Stage data from permanent storage to scratch/TMPDIR for fast job I/O.

Data flow: ESS (permanent NFS) → Scratch (GPFS, 90-day purge) → TMPDIR (per-job local SSD)

Smart caching: skips ESS→scratch copy if a marker file exists and the source
file count hasn't changed. Scratch 90-day purge deletes the marker, triggering
a fresh sync automatically.

Prints export statements for bash eval:
    eval $(python -m graphids stage-data --cache)

Usage:
    python -m graphids stage-data --cache              # cache only (training)
    python -m graphids stage-data --raw                # raw only (preprocessing)
    python -m graphids stage-data --dataset set_01     # single dataset
    python -m graphids stage-data --skip-tmpdir        # scratch only, no TMPDIR
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import structlog

structlog.configure(
    logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
)
log = structlog.get_logger()


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


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Stage data from ESS/NFS to scratch/TMPDIR for fast job I/O"
    )
    parser.add_argument("--cache", action="store_true", help="Cache only (skip raw)")
    parser.add_argument("--raw", action="store_true", help="Raw only (skip cache)")
    parser.add_argument("--skip-tmpdir", action="store_true", help="Scratch only, no TMPDIR copy")
    parser.add_argument("--dataset", default="", help="Single dataset to stage")
    args = parser.parse_args(argv)

    stage_raw = not args.cache
    stage_cache = not args.raw
    skip_tmpdir = args.skip_tmpdir
    dataset = args.dataset

    lake_root = os.environ.get("KD_GAT_LAKE_ROOT", "")
    scratch_env = os.environ.get("KD_GAT_SCRATCH")
    if not scratch_env:
        log.error("KD_GAT_SCRATCH not set. Source .env before running.")
        sys.exit(1)
    scratch = Path(scratch_env)
    scratch_data = scratch / "kd-gat-data"
    tmpdir = os.environ.get("TMPDIR", "")

    # Primary: ESS lake root (has raw/ subdir). Fallback: KD_GAT_DATA_ROOT env var.
    if lake_root and Path(lake_root, "raw").is_dir():
        data_root = Path(lake_root)
        log.info("using_ess_lake", path=str(data_root))
    else:
        data_root_env = os.environ.get("KD_GAT_DATA_ROOT")
        if not data_root_env:
            log.error("No data source found. Set KD_GAT_LAKE_ROOT (with raw/ subdir) or KD_GAT_DATA_ROOT.")
            sys.exit(1)
        data_root = Path(data_root_env)

    log.info("staging_start", source=str(data_root), scratch=str(scratch_data),
             tmpdir=tmpdir or "<not set>", dataset=dataset or "all")

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
