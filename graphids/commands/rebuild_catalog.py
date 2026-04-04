"""Rebuild DuckDB experiment catalog from run_record.json sidecars.

Scans {lake_root}/dev/ for sidecars, ingests via DuckDB read_json_auto().
Backfills legacy runs (no sidecar) from config_snapshot.yaml + metrics.csv + markers.

Usage:
    python -m graphids rebuild-catalog                  # full rebuild
    python -m graphids rebuild-catalog --backfill-only  # only write missing sidecars
    python -m graphids rebuild-catalog --dry-run        # show what would happen
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from graphids.config import PHASE_MARKERS, RUN_RECORD_FILENAME
from graphids.config.runtime import LAKE_ROOT
from graphids.log import get_logger

log = get_logger(__name__)

_CATALOG_SUBPATH = "catalog/kd_gat.duckdb"


def _backfill_legacy_runs(lake_root: str, *, dry_run: bool = False) -> int:
    """Write run_record.json for runs that don't have one yet.

    Reconstructs from config_snapshot.yaml + phase markers + metrics.csv.
    Returns count of sidecars written.
    """
    from graphids.core.contracts.run_record import (
        RunRecord,
        _parse_identity_from_run_dir,
        write_run_record,
    )

    dev_root = Path(lake_root) / "dev"
    if not dev_root.exists():
        return 0

    count = 0
    for seed_dir in sorted(dev_root.glob("*/*/seed_*")):
        sidecar = seed_dir / RUN_RECORD_FILENAME
        if sidecar.exists():
            continue

        run_dir = str(seed_dir)
        try:
            identity = _parse_identity_from_run_dir(run_dir)
        except (IndexError, ValueError):
            log.warning("backfill_skip_parse", run_dir=run_dir)
            continue

        # Determine status from markers
        phases = {
            phase: (seed_dir / marker).exists()
            for phase, marker in PHASE_MARKERS.items()
        }
        has_complete = (seed_dir / ".complete").exists()
        status = "completed" if has_complete or phases.get("train", False) else "started"

        # Extract metrics from CSVLogger if available
        metrics = _read_csv_metrics(seed_dir)

        # Read config_snapshot.yaml for version info
        graphids_version = "unknown"
        snapshot = seed_dir / "config_snapshot.yaml"
        if snapshot.exists():
            graphids_version = "pre-sidecar"

        # Approximate started_at from directory mtime
        try:
            mtime = seed_dir.stat().st_mtime
            started_at = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
        except OSError:
            started_at = datetime.now(timezone.utc).isoformat()

        record = RunRecord(
            status=status,
            run_dir=run_dir,
            stage=identity["stage"],
            model_family=identity["model_family"],
            scale=identity["scale"],
            dataset=identity["dataset"],
            seed=identity["seed"],
            identity_hash=identity["identity_hash"],
            kd_tag=identity["kd_tag"],
            user=identity["user"],
            graphids_version=graphids_version,
            started_at=started_at,
            source="cli",
            metrics=metrics,
            phases=phases,
        )

        if dry_run:
            log.info("backfill_dry_run", run_dir=run_dir, status=status,
                     metrics_count=len(metrics))
        else:
            write_run_record(record, seed_dir)
            log.info("backfill_written", run_dir=run_dir, status=status)
        count += 1

    return count


def _read_csv_metrics(seed_dir: Path) -> dict[str, float]:
    """Read final metrics from CSVLogger metrics.csv using polars."""
    versions = sorted(seed_dir.glob("lightning_logs/version_*"), key=lambda p: p.stat().st_mtime)
    if not versions:
        return {}

    csv_path = versions[-1] / "metrics.csv"
    if not csv_path.exists():
        return {}

    try:
        import polars as pl

        df = pl.read_csv(csv_path)
        if df.is_empty():
            return {}

        metrics: dict[str, float] = {}
        for col in df.columns:
            if col in ("step", "epoch"):
                continue
            series = df[col].drop_nulls()
            if series.is_empty():
                continue
            metrics[col] = round(float(series[-1]), 6)

        # Count epochs
        if "epoch" in df.columns:
            metrics["epochs_run"] = float(df["epoch"].drop_nulls().max() + 1)

        return metrics
    except Exception:
        return {}


def _rebuild_catalog(lake_root: str, *, dry_run: bool = False) -> int:
    """Ingest all run_record.json files into DuckDB catalog. Returns row count."""
    catalog_path = Path(lake_root) / _CATALOG_SUBPATH
    glob_pattern = str(Path(lake_root) / "dev" / "**" / RUN_RECORD_FILENAME)

    # Check if any sidecars exist
    sidecars = list(Path(lake_root).glob(f"dev/**/{RUN_RECORD_FILENAME}"))
    if not sidecars:
        log.info("no_sidecars_found", lake_root=lake_root)
        return 0

    if dry_run:
        log.info("rebuild_dry_run", sidecar_count=len(sidecars))
        return len(sidecars)

    from graphids.config import require_lake_write

    require_lake_write()
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    db = duckdb.connect(str(catalog_path))

    try:
        # Drop old experiments table if it exists (legacy schema)
        db.execute("DROP TABLE IF EXISTS experiments")

        # Ingest all sidecars in one shot
        db.execute("""
            CREATE OR REPLACE TABLE runs AS
            SELECT *, current_timestamp AS catalog_updated_at
            FROM read_json_auto(?, maximum_object_size=1048576, union_by_name=true)
        """, [glob_pattern])

        count = db.execute("SELECT COUNT(*) FROM runs").fetchone()[0]

        # Summary stats
        summary = db.execute("""
            SELECT status, COUNT(*) AS n
            FROM runs GROUP BY status ORDER BY n DESC
        """).fetchall()

        log.info("catalog_rebuilt", total=count,
                 breakdown={s: n for s, n in summary},
                 catalog_path=str(catalog_path))

        return count
    finally:
        db.close()


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m graphids rebuild-catalog",
        description="Rebuild DuckDB catalog from run_record.json sidecars",
    )
    parser.add_argument("--lake-root", default=LAKE_ROOT,
                        help=f"Lake root directory (default: {LAKE_ROOT})")
    parser.add_argument("--backfill-only", action="store_true",
                        help="Only write missing sidecars, skip catalog rebuild")
    parser.add_argument("--skip-backfill", action="store_true",
                        help="Skip legacy backfill, only rebuild catalog from existing sidecars")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would happen without writing")
    args = parser.parse_args(argv)

    lake_root = args.lake_root

    if not args.skip_backfill:
        backfilled = _backfill_legacy_runs(lake_root, dry_run=args.dry_run)
        print(f"Backfilled {backfilled} legacy run(s)")

    if args.backfill_only:
        return

    count = _rebuild_catalog(lake_root, dry_run=args.dry_run)
    if args.dry_run:
        print(f"Would ingest {count} sidecar(s)")
    else:
        print(f"Catalog rebuilt: {count} run(s) in {lake_root}/{_CATALOG_SUBPATH}")
