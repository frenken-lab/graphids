#!/usr/bin/env python3
"""Push experiment data to HF Dataset for dashboard consumption.

Usage:
    python scripts/data/push_experiments_to_hf.py

Data flow:
    _manifest.json files → rebuild_catalog() → DuckDB → experiments.parquet → HF Dataset
"""

from __future__ import annotations

import os
from datetime import UTC, datetime

import structlog

log = structlog.get_logger()

HF_DATASET_REPO = "buckeyeguy/kd-gat-experiments"


def push():
    import duckdb
    from huggingface_hub import HfApi

    from graphids.storage.paths import lake_catalog_path, lake_root_from_env
    from graphids.storage.catalog import rebuild_catalog

    lake_root = lake_root_from_env()
    if lake_root is None:
        log.error("lake_root_not_set")
        return

    # Rebuild catalog from manifests, then query
    catalog_path = rebuild_catalog(lake_root)
    con = duckdb.connect(str(catalog_path), read_only=True)
    runs = con.execute("SELECT * FROM experiments").fetchdf()
    con.close()

    if runs.empty:
        log.warning("no_runs_found")
        return

    log.info("runs_found", count=len(runs))

    # Write to temp parquet
    out_path = "/tmp/kd_gat_experiments.parquet"
    runs.to_parquet(out_path, index=False)

    # Also write to ESS exports/
    exports_dir = lake_root / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    lake_parquet = exports_dir / "experiments.parquet"
    runs.to_parquet(str(lake_parquet), index=False)
    log.info("lake_export_written", path=str(lake_parquet))

    token = os.environ.get("HF_TOKEN")
    if not token:
        log.warning("hf_token_not_set")
        return

    api = HfApi(token=token)
    api.create_repo(repo_id=HF_DATASET_REPO, repo_type="dataset", private=True, exist_ok=True)
    api.upload_file(
        path_or_fileobj=out_path,
        path_in_repo="experiments.parquet",
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        commit_message=f"Update experiments ({datetime.now(UTC).strftime('%Y-%m-%d %H:%M')})",
    )
    log.info("hf_push_complete", repo=HF_DATASET_REPO)


if __name__ == "__main__":
    from graphids.logging import configure_logging
    configure_logging()
    push()
