"""Sync structured experiment results to datalake (Parquet).

Write path: append to data/datalake/*.parquet via DuckDB INSERT INTO.

Failures are logged but never crash the training pipeline.

Downstream consumption:
    duckdb -c "SELECT * FROM 'data/datalake/runs.parquet'"
    duckdb data/datalake/analytics.duckdb  # pre-built views
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from pathlib import Path

log = logging.getLogger(__name__)

_data_root = os.environ.get("KD_GAT_DATA_ROOT")
_DATALAKE_ROOT = Path(_data_root) / "datalake" if _data_root else Path("data/datalake")

# Core metrics to extract from nested metrics.json
_CORE_METRIC_COLS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "specificity",
    "balanced_accuracy",
    "mcc",
    "fpr",
    "fnr",
    "auc",
    "n_samples",
]


# ---------------------------------------------------------------------------
# Parquet datalake writes (primary)
# ---------------------------------------------------------------------------


def _append_to_datalake(
    run_id: str,
    dataset: str,
    model_type: str,
    scale: str,
    stage: str,
    has_kd: bool,
    metrics: dict | None,
    success: bool,
    failure_reason: str | None,
    *,
    started_at: str | None = None,
    completed_at: str | None = None,
    duration_seconds: float | None = None,
    peak_gpu_mb: float | None = None,
    slurm_job_id: str | None = None,
    gpu_name: str | None = None,
    batch_size_used: int | None = None,
    data_version: str | None = None,
) -> bool:
    """Append run + metrics to datalake Parquet files via DuckDB."""
    try:
        import duckdb
    except ImportError:
        log.debug("duckdb not installed — datalake append skipped")
        return False

    if not (_DATALAKE_ROOT / "runs.parquet").exists():
        log.debug(
            "Datalake not initialized — run `python -m graphids.pipeline.migrate_datalake` first"
        )
        return False

    try:
        now = datetime.now(UTC).isoformat()
        datalake = str(_DATALAKE_ROOT)
        con = duckdb.connect()

        # Upsert run record
        con.execute(
            f"""
            INSERT INTO '{datalake}/runs.parquet'
            BY NAME (SELECT
                ? AS run_id, ? AS dataset, ? AS model_type, ? AS scale,
                ? AS stage, ? AS has_kd, '' AS auxiliaries, ? AS success,
                ? AS completed_at, ? AS started_at, ? AS duration_seconds,
                ? AS peak_gpu_mb, ? AS slurm_job_id, ? AS gpu_name,
                ? AS batch_size_used,
                ? AS data_version, NULL AS wandb_run_id, 'pipeline' AS source
            )
        """,
            [
                run_id,
                dataset,
                model_type,
                scale,
                stage,
                has_kd,
                success,
                completed_at or now,
                started_at or now,
                duration_seconds,
                peak_gpu_mb,
                slurm_job_id,
                gpu_name,
                batch_size_used,
                data_version,
            ],
        )

        # Append metrics if evaluation stage
        if metrics:
            for model_key, model_data in metrics.items():
                if model_key == "test":
                    continue
                if not isinstance(model_data, dict) or "core" not in model_data:
                    continue
                core = model_data["core"]
                values = [run_id, model_key]
                for col in _CORE_METRIC_COLS:
                    val = core.get(col)
                    values.append(float(val) if isinstance(val, (int, float)) else None)
                placeholders = ", ".join(["?"] * len(values))
                cols = "run_id, model, " + ", ".join(
                    f'"{c}"' if c == "precision" else c for c in _CORE_METRIC_COLS
                )
                con.execute(
                    f"INSERT INTO '{datalake}/metrics.parquet' ({cols}) VALUES ({placeholders})",
                    values,
                )

        con.close()
        log.info("Datalake append OK: %s", run_id)
        return True
    except Exception as e:
        log.warning("Datalake append failed for %s: %s", run_id, e)
        return False


def register_artifacts(run_id: str, run_dir: Path) -> bool:
    """Scan a run directory and append artifact records to datalake."""
    try:
        import duckdb
    except ImportError:
        return False

    if not (_DATALAKE_ROOT / "artifacts.parquet").exists():
        return False

    ARTIFACT_TYPES = {
        "best_model.pt": "checkpoint",
        "embeddings.npz": "embeddings",
        "attention_weights.npz": "attention",
        "dqn_policy.json": "policy",
        "metrics.json": "metrics",
        "config.json": "config",
        "explanations.npz": "explanations",
        "cka_matrix.json": "cka",
    }

    try:
        records = []
        for filename, artifact_type in ARTIFACT_TYPES.items():
            fpath = run_dir / filename
            if fpath.exists():
                records.append((run_id, artifact_type, str(fpath), fpath.stat().st_size))

        if not records:
            return True

        datalake = str(_DATALAKE_ROOT)
        con = duckdb.connect()
        con.executemany(
            f"INSERT INTO '{datalake}/artifacts.parquet' VALUES (?, ?, ?, ?)",
            records,
        )
        con.close()
        log.info("Registered %d artifacts for %s", len(records), run_id)
        return True
    except Exception as e:
        log.warning("Artifact registration failed for %s: %s", run_id, e)
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def sync_to_lakehouse(
    run_id: str,
    dataset: str,
    model_type: str,
    scale: str,
    stage: str,
    has_kd: bool,
    metrics: dict | None = None,
    success: bool = True,
    failure_reason: str | None = None,
    *,
    started_at: str | None = None,
    completed_at: str | None = None,
    duration_seconds: float | None = None,
    peak_gpu_mb: float | None = None,
    slurm_job_id: str | None = None,
    gpu_name: str | None = None,
    batch_size_used: int | None = None,
    data_version: str | None = None,
) -> bool:
    """Append run results to datalake (Parquet). Returns True on success.

    This function is intentionally fire-and-forget: it catches all
    exceptions and logs warnings instead of raising.  Training must
    never fail because of a lakehouse sync issue.
    """
    # Default to current preprocessing version if not specified
    if data_version is None:
        from graphids.config.constants import PREPROCESSING_VERSION

        data_version = PREPROCESSING_VERSION

    return _append_to_datalake(
        run_id,
        dataset,
        model_type,
        scale,
        stage,
        has_kd,
        metrics,
        success,
        failure_reason,
        started_at=started_at,
        completed_at=completed_at,
        duration_seconds=duration_seconds,
        peak_gpu_mb=peak_gpu_mb,
        slurm_job_id=slurm_job_id,
        gpu_name=gpu_name,
        batch_size_used=batch_size_used,
        data_version=data_version,
    )
