#!/usr/bin/env python3
"""Push MLflow experiment data to HF Dataset for dashboard consumption.

Usage:
    python scripts/data/push_experiments_to_hf.py

Data flow:
    MLflow SQLite → mlflow.search_runs() → experiments.parquet → HF Dataset
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime

log = logging.getLogger(__name__)

HF_DATASET_REPO = "buckeyeguy/kd-gat-experiments"


def push():
    import mlflow
    from huggingface_hub import HfApi

    from graphids.config import MLFLOW_TRACKING_URI

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Search all runs across all experiments
    runs = mlflow.search_runs(search_all_experiments=True)
    if runs.empty:
        log.warning("No MLflow runs found — nothing to push")
        return

    log.info("Found %d MLflow runs", len(runs))

    # Write to temp parquet
    out_path = "/tmp/kd_gat_experiments.parquet"
    runs.to_parquet(out_path, index=False)

    token = os.environ.get("HF_TOKEN")
    if not token:
        log.warning("HF_TOKEN not set — skipping HF push")
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
    log.info("Pushed experiments.parquet to %s", HF_DATASET_REPO)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    push()
