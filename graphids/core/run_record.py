"""Per-run structured sidecar schema.

Pure Pydantic. Filesystem I/O for this record (atomic write, read,
run-dir identity parsing) lives in ``graphids.core.io``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


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
