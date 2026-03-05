"""Parse Ray Tune results and push to HF Dataset for dashboard consumption.

Data flow:
    ~/ray_results/tune_{stage}_{dataset}_{scale}/
    → data/datalake/sweeps.parquet (local, DuckDB-queryable)
    → HF Dataset: buckeyeguy/kd-gat-sweeps (private)
    → HF Space: buckeyeguy/kd-gat-sweep-dashboard (Streamlit)

Usage:
    from graphids.pipeline.sweep_export import ingest_and_push
    ingest_and_push(Path("ray_results/tune_autoencoder_hcrl_ch_large"), "autoencoder", "hcrl_ch", "large")
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

_data_root = os.environ.get("KD_GAT_DATA_ROOT")
_DATALAKE_ROOT = Path(_data_root) / "datalake" if _data_root else Path("data/datalake")
_SWEEPS_PARQUET = _DATALAKE_ROOT / "sweeps.parquet"

HF_DATASET_REPO = "buckeyeguy/kd-gat-sweeps"


def ingest_sweep(experiment_dir: Path, stage: str, dataset: str, scale: str) -> pd.DataFrame:
    """Parse all trials from a Ray Tune results directory into a DataFrame."""
    sweep_id = experiment_dir.name
    rows: list[dict] = []

    for trial_dir in sorted(experiment_dir.iterdir()):
        if not trial_dir.is_dir() or not trial_dir.name.startswith("_trainable_"):
            continue

        trial_id = trial_dir.name.split("_")[2]  # e.g. 32942427

        # Read params
        params_path = trial_dir / "params.json"
        if not params_path.exists():
            continue
        try:
            params = json.loads(params_path.read_text())
        except (json.JSONDecodeError, OSError):
            log.warning("Skipping trial %s: unreadable params.json", trial_id)
            continue

        # Read result
        result_path = trial_dir / "result.json"
        val_loss = None
        duration_s = None
        timestamp = None
        status = "ERROR"

        if result_path.exists():
            try:
                result = json.loads(result_path.read_text())
                val_loss = result.get("val_loss")
                duration_s = result.get("time_total_s")
                timestamp = result.get("date")
                # If val_loss is inf, trial failed
                if val_loss is not None and val_loss != float("inf"):
                    status = "TERMINATED"
                else:
                    status = "ERROR"
            except (json.JSONDecodeError, OSError):
                log.warning("Skipping trial %s: unreadable result.json", trial_id)

        row = {
            "sweep_id": sweep_id,
            "trial_id": trial_id,
            "stage": stage,
            "dataset": dataset,
            "scale": scale,
            "status": status,
            "val_loss": val_loss,
            "duration_s": duration_s,
            "timestamp": timestamp or datetime.now(UTC).isoformat(),
        }

        # Flatten HPs with hp_ prefix
        for key, value in params.items():
            row[f"hp_{key}"] = value

        rows.append(row)

    if not rows:
        log.warning("No trials found in %s", experiment_dir)
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    log.info("Ingested %d trials from %s", len(df), experiment_dir)
    return df


def append_to_lakehouse(df: pd.DataFrame) -> Path:
    """Append trial data to data/datalake/sweeps.parquet."""
    if df.empty:
        return _SWEEPS_PARQUET

    import duckdb

    _DATALAKE_ROOT.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()

    if _SWEEPS_PARQUET.exists():
        # Deduplicate: remove existing rows with same sweep_id + trial_id
        existing = con.execute(
            f"SELECT * FROM '{_SWEEPS_PARQUET}' "
            "WHERE (sweep_id, trial_id) NOT IN "
            "(SELECT sweep_id, trial_id FROM df)"
        ).fetchdf()
        combined = pd.concat([existing, df], ignore_index=True)
    else:
        combined = df

    con.execute(f"COPY combined TO '{_SWEEPS_PARQUET}' (FORMAT PARQUET, OVERWRITE)")
    con.close()

    log.info("Lakehouse updated: %s (%d total rows)", _SWEEPS_PARQUET, len(combined))
    return _SWEEPS_PARQUET


def push_to_hf(parquet_path: Path) -> None:
    """Upload sweeps.parquet to private HF Dataset."""
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        log.warning("HF_TOKEN not set — skipping HF push")
        return

    api = HfApi(token=token)

    # Ensure repo exists (creates if needed)
    api.create_repo(
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        private=True,
        exist_ok=True,
    )

    api.upload_file(
        path_or_fileobj=str(parquet_path),
        path_in_repo="sweeps.parquet",
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        commit_message=f"Update sweeps data ({datetime.now(UTC).strftime('%Y-%m-%d %H:%M')})",
    )
    log.info("Pushed %s to %s", parquet_path, HF_DATASET_REPO)


def ingest_and_push(experiment_dir: Path, stage: str, dataset: str, scale: str) -> None:
    """Full pipeline: parse → lakehouse → HF push."""
    df = ingest_sweep(experiment_dir, stage, dataset, scale)
    if df.empty:
        log.warning("No trials to export from %s", experiment_dir)
        return
    parquet_path = append_to_lakehouse(df)
    push_to_hf(parquet_path)
