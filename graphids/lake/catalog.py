"""DuckDB catalog rebuild + query helpers.

Scans production/ and dev/ for _manifest.json, config.json, and metrics.json,
then builds a queryable DuckDB database. The catalog is disposable —
rebuilt from files in seconds.

Requires: duckdb>=1.0.0 (optional dependency).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def rebuild_catalog(lake_root: Path, catalog_path: Path | None = None) -> Path:
    """Rebuild the DuckDB catalog from manifest/config/metrics files.

    Scans ``production/`` and ``dev/`` under ``lake_root`` for run directories
    containing ``_manifest.json``. Joins with ``config.json`` and ``metrics.json``
    for a flat queryable table.

    Parameters
    ----------
    lake_root : Path
        Root of the data lake (e.g. ``/fs/ess/PAS1266/kd-gat``).
    catalog_path : Path | None
        Output DuckDB file. Defaults to ``lake_root/catalog/kd_gat.duckdb``.

    Returns
    -------
    Path
        The catalog file path.
    """
    import duckdb

    if catalog_path is None:
        catalog_path = lake_root / "catalog" / "kd_gat.duckdb"

    catalog_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect all run directories that have a manifest
    manifest_files = []
    for tier_dir in [lake_root / "production", lake_root / "dev"]:
        if tier_dir.exists():
            manifest_files.extend(tier_dir.rglob("_manifest.json"))

    if not manifest_files:
        log.warning("No _manifest.json files found under %s", lake_root)
        return catalog_path

    log.info("Found %d manifest files", len(manifest_files))

    # Build rows by joining manifest + config + metrics
    rows = []
    for mf in manifest_files:
        run_dir = mf.parent
        try:
            manifest = json.loads(mf.read_text())
        except (json.JSONDecodeError, OSError) as e:
            log.warning("Skipping %s: %s", mf, e)
            continue

        # Determine tier from path
        rel = run_dir.relative_to(lake_root)
        tier = rel.parts[0]  # "production" or "dev"

        row = {
            "run_dir": str(run_dir),
            "tier": tier,
            "dataset": manifest.get("dataset", ""),
            "model_type": manifest.get("model_type", ""),
            "scale": manifest.get("scale", ""),
            "stage": manifest.get("stage", ""),
            "auxiliaries": manifest.get("auxiliaries", "none"),
            "seed": manifest.get("seed", 42),
            "created_at": manifest.get("created_at", ""),
            "graphids_version": manifest.get("graphids_version", ""),
            "git_sha": manifest.get("git_sha", ""),
            "slurm_job_id": manifest.get("slurm_job_id", ""),
            "num_artifacts": len(manifest.get("artifacts", [])),
        }

        # Merge config.json fields (flat)
        config_path = run_dir / "config.json"
        if config_path.exists():
            try:
                config = json.loads(config_path.read_text())
                # Extract key training params
                training = config.get("training", {})
                row["lr"] = training.get("lr")
                row["max_epochs"] = training.get("max_epochs")
                row["batch_size"] = training.get("batch_size")
                row["precision"] = training.get("precision")
                row["has_kd"] = bool(config.get("auxiliaries"))
            except (json.JSONDecodeError, OSError):
                pass

        # Merge metrics.json fields (flat)
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text())
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        row[f"metric_{k}"] = v
            except (json.JSONDecodeError, OSError):
                pass

        rows.append(row)

    if not rows:
        log.warning("No valid runs found")
        return catalog_path

    # Write rows to a temp JSON file, then load into DuckDB
    import tempfile

    tmp_json = Path(tempfile.mktemp(suffix=".json"))
    try:
        tmp_json.write_text(json.dumps(rows))

        db = duckdb.connect(str(catalog_path))
        try:
            db.execute("DROP TABLE IF EXISTS experiments")
            db.execute(
                f"CREATE TABLE experiments AS SELECT * FROM read_json_auto('{tmp_json}', union_by_name=true, maximum_object_size=10485760)"
            )
            count = db.execute("SELECT count(*) FROM experiments").fetchone()[0]
            log.info("Catalog rebuilt: %d runs → %s", count, catalog_path)
        finally:
            db.close()
    finally:
        tmp_json.unlink(missing_ok=True)

    return catalog_path


def query_catalog(catalog_path: Path, sql: str) -> list[dict]:
    """Run a SQL query against the catalog and return results as dicts."""
    import duckdb

    db = duckdb.connect(str(catalog_path), read_only=True)
    try:
        result = db.execute(sql)
        columns = [desc[0] for desc in result.description]
        return [dict(zip(columns, row)) for row in result.fetchall()]
    finally:
        db.close()


def catalog_status(catalog_path: Path) -> dict:
    """Get summary statistics from the catalog."""
    import duckdb

    if not catalog_path.exists():
        return {"exists": False}

    db = duckdb.connect(str(catalog_path), read_only=True)
    try:
        total = db.execute("SELECT count(*) FROM experiments").fetchone()[0]
        by_stage = db.execute(
            "SELECT stage, count(*) as n FROM experiments GROUP BY stage ORDER BY n DESC"
        ).fetchall()
        by_dataset = db.execute(
            "SELECT dataset, count(*) as n FROM experiments GROUP BY dataset ORDER BY n DESC"
        ).fetchall()
        return {
            "exists": True,
            "total_runs": total,
            "by_stage": {r[0]: r[1] for r in by_stage},
            "by_dataset": {r[0]: r[1] for r in by_dataset},
        }
    finally:
        db.close()
