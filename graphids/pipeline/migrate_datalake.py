"""One-time migration: add missing columns to runs.parquet and metrics.parquet.

Adds columns that were in the INSERT statement but missing from the initial schema,
plus new enrichment columns (run_uuid, run_type, sweep_id, teacher_run_id, config_hash, tags).

Uses DuckDB SQL with explicit CAST to ensure correct column types (pandas infers
all-NULL columns as INTEGER, which breaks INSERT BY NAME later).

Safe to re-run.

Usage:
    python -m graphids.pipeline.migrate_datalake
    python -m graphids.pipeline.migrate_datalake --dry-run
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path


def _make_run_uuid(run_id: str) -> str:
    return hashlib.sha256(run_id.encode()).hexdigest()[:16]


# Column name → SQL type for runs.parquet columns that need type enforcement
_RUNS_COL_TYPES: dict[str, str] = {
    "peak_gpu_mb": "DOUBLE",
    "slurm_job_id": "VARCHAR",
    "gpu_name": "VARCHAR",
    "batch_size_used": "INTEGER",
    "failure_reason": "VARCHAR",
    "run_uuid": "VARCHAR",
    "run_type": "VARCHAR",
    "sweep_id": "VARCHAR",
    "teacher_run_id": "VARCHAR",
    "config_hash": "VARCHAR",
    "tags": "VARCHAR",
    # Pre-existing columns with wrong types from initial schema
    "data_version": "VARCHAR",
    "wandb_run_id": "VARCHAR",
}


def migrate(dry_run: bool = False) -> None:
    import duckdb

    data_root = os.environ.get("KD_GAT_DATA_ROOT")
    datalake = Path(data_root) / "datalake" if data_root else Path("data/datalake")

    # --- Migrate runs.parquet ---
    runs_path = datalake / "runs.parquet"
    if not runs_path.exists():
        print(f"No runs.parquet at {runs_path} — nothing to migrate")
        return

    con = duckdb.connect()

    # Get existing columns and types
    existing = {
        row[0]: row[1] for row in con.execute(f"DESCRIBE SELECT * FROM '{runs_path}'").fetchall()
    }
    count = con.execute(f"SELECT COUNT(*) FROM '{runs_path}'").fetchone()[0]
    print(f"runs.parquet: {count} rows")
    print(f"  existing columns: {list(existing.keys())}")

    # Build SELECT with new columns added as CAST(NULL AS type) and type fixes
    select_parts = []
    added = []
    type_fixed = []
    for col, existing_type in existing.items():
        if col in _RUNS_COL_TYPES and existing_type != _RUNS_COL_TYPES[col]:
            # Type mismatch — cast to correct type
            select_parts.append(f"CAST({col} AS {_RUNS_COL_TYPES[col]}) AS {col}")
            type_fixed.append(f"{col}: {existing_type} → {_RUNS_COL_TYPES[col]}")
        else:
            select_parts.append(col)

    for col, sql_type in _RUNS_COL_TYPES.items():
        if col not in existing:
            select_parts.append(f"CAST(NULL AS {sql_type}) AS {col}")
            added.append(col)

    if dry_run:
        print(f"  would add: {added}")
        print(f"  would fix types: {type_fixed}")
        # Check metrics too
        metrics_path = datalake / "metrics.parquet"
        if metrics_path.exists():
            m_existing = {
                row[0] for row in con.execute(f"DESCRIBE SELECT * FROM '{metrics_path}'").fetchall()
            }
            print(
                f"\nmetrics.parquet: dataset column {'present' if 'dataset' in m_existing else 'missing'}"
            )
        con.close()
        return

    # Write with correct types
    select_sql = ", ".join(select_parts)
    con.execute(
        f"COPY (SELECT {select_sql} FROM '{runs_path}') TO '{runs_path}' (FORMAT PARQUET, OVERWRITE)"
    )

    # Backfill run_uuid and run_type using a Python UDF registered in DuckDB
    con.create_function("make_run_uuid", _make_run_uuid, [str], str)
    con.execute(f"""
        COPY (
            SELECT * REPLACE (
                make_run_uuid(run_id) AS run_uuid,
                COALESCE(run_type, 'production') AS run_type
            )
            FROM '{runs_path}'
        ) TO '{runs_path}' (FORMAT PARQUET, OVERWRITE)
    """)

    print(f"  added: {added}")
    print(f"  fixed types: {type_fixed}")
    print(f"  backfilled: run_uuid, run_type")

    # --- Migrate metrics.parquet ---
    metrics_path = datalake / "metrics.parquet"
    if metrics_path.exists():
        m_existing = {
            row[0] for row in con.execute(f"DESCRIBE SELECT * FROM '{metrics_path}'").fetchall()
        }
        if "dataset" not in m_existing:
            con.execute(f"""
                COPY (
                    SELECT *, split_part(run_id, '/', 1) AS dataset
                    FROM '{metrics_path}'
                ) TO '{metrics_path}' (FORMAT PARQUET, OVERWRITE)
            """)
            print("metrics.parquet: added dataset column")
        else:
            print("metrics.parquet: dataset column already present")

    # Verify final types
    print("\nFinal runs.parquet schema:")
    for name, dtype, *_ in con.execute(f"DESCRIBE SELECT * FROM '{runs_path}'").fetchall():
        marker = ""
        if name in _RUNS_COL_TYPES:
            expected = _RUNS_COL_TYPES[name]
            if dtype != expected:
                marker = f"  ⚠ expected {expected}"
        print(f"  {name}: {dtype}{marker}")

    con.close()
    print("\nMigration complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Migrate datalake Parquet schemas")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    migrate(dry_run=args.dry_run)
