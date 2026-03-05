"""Build analytics DuckDB from datalake Parquet files.

Reads:
  - data/datalake/*.parquet (primary — created by migrate_datalake.py, updated by lakehouse.py)
  - Falls back to filesystem scan if datalake doesn't exist

Writes:
  - data/datalake/analytics.duckdb (views + convenience queries over Parquet)

Usage:
    python -m graphids.pipeline.build_analytics              # Rebuild views
    python -m graphids.pipeline.build_analytics --dry-run    # Show what would be built
    duckdb data/datalake/analytics.duckdb           # Interactive queries

Example queries:
    SELECT * FROM v_leaderboard ORDER BY best_f1 DESC;
    SELECT * FROM v_kd_impact ORDER BY f1_delta DESC;
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

_data_root = os.environ.get("KD_GAT_DATA_ROOT")
DATALAKE_ROOT = Path(_data_root) / "datalake" if _data_root else Path("data/datalake")


def build(dry_run: bool = False) -> Path:
    """Build (or rebuild) analytics.duckdb from datalake Parquet files."""
    import duckdb

    datalake = str(DATALAKE_ROOT)
    db_path = DATALAKE_ROOT / "analytics.duckdb"

    if not (DATALAKE_ROOT / "runs.parquet").exists():
        log.error("Datalake not initialized. Run: python -m graphids.pipeline.migrate_datalake")
        raise FileNotFoundError(f"{DATALAKE_ROOT}/runs.parquet not found")

    if dry_run:
        con = duckdb.connect()
        runs = con.execute(f"SELECT COUNT(*) FROM '{datalake}/runs.parquet'").fetchone()[0]
        metrics = con.execute(f"SELECT COUNT(*) FROM '{datalake}/metrics.parquet'").fetchone()[0]
        datasets = con.execute(f"SELECT COUNT(*) FROM '{datalake}/datasets.parquet'").fetchone()[0]
        configs = con.execute(f"SELECT COUNT(*) FROM '{datalake}/configs.parquet'").fetchone()[0]
        artifacts = con.execute(f"SELECT COUNT(*) FROM '{datalake}/artifacts.parquet'").fetchone()[
            0
        ]
        con.close()
        print(f"Would rebuild: {db_path}")
        print(f"  runs: {runs}, metrics: {metrics}, datasets: {datasets}")
        print(f"  configs: {configs}, artifacts: {artifacts}")
        return db_path

    # Remove old DB and create fresh views
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    con = duckdb.connect(str(db_path))
    try:
        # Core views over Parquet
        con.execute(f"CREATE VIEW runs AS SELECT * FROM read_parquet('{datalake}/runs.parquet')")
        con.execute(
            f"CREATE VIEW metrics AS SELECT * FROM read_parquet('{datalake}/metrics.parquet')"
        )
        con.execute(
            f"CREATE VIEW configs AS SELECT * FROM read_parquet('{datalake}/configs.parquet')"
        )
        con.execute(
            f"CREATE VIEW datasets AS SELECT * FROM read_parquet('{datalake}/datasets.parquet')"
        )
        con.execute(
            f"CREATE VIEW artifacts AS SELECT * FROM read_parquet('{datalake}/artifacts.parquet')"
        )

        # Leaderboard: best metrics per dataset × model × scale × kd × data_version
        con.execute("""
            CREATE VIEW v_leaderboard AS
            SELECT r.dataset, r.model_type, r.scale, r.has_kd, r.data_version,
                   m.model, MAX(m.f1) AS best_f1, MAX(m.accuracy) AS best_accuracy,
                   MAX(m.auc) AS best_auc, MAX(m.mcc) AS best_mcc
            FROM runs r
            JOIN metrics m USING (run_id)
            WHERE r.stage = 'evaluation' AND r.success
            GROUP BY r.dataset, r.model_type, r.scale, r.has_kd, r.data_version, m.model
        """)

        # KD impact: small+KD vs small+noKD vs large teacher (grouped by data_version)
        con.execute("""
            CREATE VIEW v_kd_impact AS
            SELECT
                kd.dataset, kd.model, kd.data_version,
                kd.best_f1 AS kd_f1, nokd.best_f1 AS nokd_f1,
                kd.best_f1 - nokd.best_f1 AS f1_delta,
                teacher.best_f1 AS teacher_f1
            FROM v_leaderboard kd
            JOIN v_leaderboard nokd
                ON kd.dataset = nokd.dataset
                AND kd.model = nokd.model
                AND kd.data_version IS NOT DISTINCT FROM nokd.data_version
                AND kd.scale = 'small' AND kd.has_kd = true
                AND nokd.scale = 'small' AND nokd.has_kd = false
            LEFT JOIN v_leaderboard teacher
                ON teacher.dataset = kd.dataset
                AND teacher.model = kd.model
                AND teacher.data_version IS NOT DISTINCT FROM kd.data_version
                AND teacher.scale = 'large'
        """)

        # Summary
        counts = {}
        for view in ["runs", "metrics", "datasets", "configs", "artifacts"]:
            counts[view] = con.execute(f"SELECT COUNT(*) FROM {view}").fetchone()[0]
        log.info("Analytics DB built: %s", counts)
        print(f"Built {db_path} ({db_path.stat().st_size / 1024:.0f} KB)")
        for view, count in counts.items():
            print(f"  {view}: {count} rows")
    finally:
        con.close()

    return db_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Build analytics DuckDB from datalake Parquet")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be built")
    args = parser.parse_args()
    build(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
