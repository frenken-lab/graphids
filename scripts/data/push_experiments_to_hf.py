#!/usr/bin/env python3
"""Push experiment data to HF Dataset for dashboard consumption.

Usage:
    python scripts/data/push_experiments_to_hf.py

Data flow:
    metrics.csv + hparams.yaml files → pandas → experiments.parquet → HF Dataset
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import structlog

log = structlog.get_logger()

HF_DATASET_REPO = "buckeyeguy/kd-gat-experiments"


def push():
    import pandas as pd
    import yaml
    from huggingface_hub import HfApi

    from graphids.config.paths import lake_exports_dir, lake_root_from_env

    lake_root = lake_root_from_env()
    if lake_root is None:
        log.error("lake_root_not_set")
        return

    csv_files = list(Path(lake_root).rglob("metrics.csv"))
    if not csv_files:
        log.warning("no_metrics_files_found")
        return

    frames = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
        except Exception as e:
            log.warning("csv_read_failed", path=str(f), error=str(e))
            continue
        df["run_dir"] = str(f.parent)
        hparams = f.parent / "hparams.yaml"
        if hparams.exists():
            try:
                hp = yaml.safe_load(hparams.read_text())
                if isinstance(hp, dict) and "cfg" in hp:
                    cfg = hp["cfg"]
                    for key in ("dataset", "model_type", "scale", "seed"):
                        df[key] = cfg.get(key, "")
            except Exception as e:
                log.warning("hparams_read_failed", path=str(hparams), error=str(e))
        frames.append(df)

    if not frames:
        log.warning("no_valid_runs")
        return

    runs = pd.concat(frames, ignore_index=True)
    log.info("runs_found", count=len(runs))

    out_path = "/tmp/kd_gat_experiments.parquet"
    runs.to_parquet(out_path, index=False)

    exports = lake_exports_dir(lake_root)
    exports.mkdir(parents=True, exist_ok=True)
    lake_parquet = exports / "experiments.parquet"
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
