"""Per-run structured sidecar: source of truth for catalog rebuilds.

Written atomically to {run_dir}/run_record.json by RunRecordCallback
(Lightning) and finalize-record (post test+analyze).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from graphids.config.runtime import RUN_RECORD_FILENAME


class RunRecord(BaseModel):
    """Per-run structured sidecar."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    status: Literal["started", "completed", "failed"]

    # Identity (enough to rebuild a catalog row)
    run_dir: str
    stage: str
    model_family: str
    scale: str
    dataset: str
    seed: int
    identity_hash: str
    kd_tag: str = ""
    user: str
    graphids_version: str

    # Timing
    started_at: str  # ISO 8601 UTC
    completed_at: str | None = None
    wall_time_seconds: float | None = None

    # SLURM context
    slurm_job_id: int | None = None
    slurm_partition: str | None = None

    # Execution source
    source: Literal["dagster", "cli"]

    # Metrics (populated on completion) — model-specific keys
    metrics: dict[str, float] = Field(default_factory=dict)

    # Phase markers (populated by finalize-record)
    phases: dict[str, bool] = Field(default_factory=dict)

    # Failure info
    error_message: str | None = None


def write_run_record(record: RunRecord, run_dir: Path) -> Path:
    """Write run_record.json atomically (NFS/GPFS-safe: temp → fsync → rename).

    Follows the same pattern as ``yaml_utils.write_yaml``.
    """
    from graphids.config import require_lake_write

    require_lake_write()
    path = run_dir / RUN_RECORD_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        data = record.model_dump_json(indent=2)
        with tmp.open("w") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.rename(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return path


def read_run_record(run_dir: Path) -> RunRecord | None:
    """Read run_record.json if it exists, else None."""
    path = run_dir / RUN_RECORD_FILENAME
    if not path.exists():
        return None
    return RunRecord.model_validate_json(path.read_text())


def _parse_identity_from_run_dir(run_dir: str) -> dict[str, Any]:
    """Extract identity fields from a run_dir path.

    Path convention: {lake_root}/dev/{user}/{dataset}/{model}_{scale}_{stage}{identity}{kd_tag}/seed_{seed}
    """
    parts = Path(run_dir).parts
    # seed_N is the last component
    seed_part = parts[-1]  # "seed_42"
    seed = int(seed_part.split("_", 1)[1])
    # {model}_{scale}_{stage}{identity}{kd_tag} is second-to-last
    dir_name = parts[-2]
    # dataset is third-to-last
    dataset = parts[-3]
    # user is fourth-to-last
    user = parts[-4]

    # Parse dir_name: vgae_small_autoencoder_8e6b9f70 or vgae_small_autoencoder_8e6b9f70_kd
    kd_tag = ""
    if dir_name.endswith("_kd"):
        kd_tag = "_kd"
        dir_name = dir_name[: -len("_kd")]

    # identity_hash is last _XXXXXXXX (8 hex chars after underscore)
    last_underscore = dir_name.rfind("_")
    identity_hash = "_" + dir_name[last_underscore + 1:]
    remainder = dir_name[:last_underscore]

    # remainder is model_type_scale_stage — split on known stages
    from graphids.config.topology import STAGES

    stage = ""
    for s in STAGES:
        suffix = f"_{s}"
        if remainder.endswith(suffix):
            stage = s
            remainder = remainder[: -len(suffix)]
            break

    # remainder is now model_type_scale
    last_us = remainder.rfind("_")
    model_type = remainder[:last_us]
    scale = remainder[last_us + 1:]

    return {
        "dataset": dataset,
        "user": user,
        "seed": seed,
        "model_family": model_type,
        "scale": scale,
        "stage": stage,
        "identity_hash": identity_hash,
        "kd_tag": kd_tag,
    }
