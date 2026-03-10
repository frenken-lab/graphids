"""Scheduler-agnostic job definition models.

Pydantic v2 frozen models for describing jobs, resources, and dependencies.
No scheduler concepts (SLURM, Flux) leak into these definitions.

DAG adjacency uses opaque UUIDs with parent arrays (WfFormat pattern).
Parameters are typed fields, not encoded into keys.
"""

from __future__ import annotations

from datetime import timedelta
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ResourceSpec(BaseModel, frozen=True):
    """Scheduler-agnostic resource requirements."""

    nodes: int = 1
    gpus: int = 0
    cpus: int = 4
    memory_gb: int = 20
    walltime: timedelta = timedelta(hours=3)

    @property
    def walltime_str(self) -> str:
        """Format as H:MM:SS for scheduler submission."""
        total = int(self.walltime.total_seconds())
        h, remainder = divmod(total, 3600)
        m, s = divmod(remainder, 60)
        return f"{h}:{m:02d}:{s:02d}"

    def scale_memory(self, factor: float) -> ResourceSpec:
        """Return a new spec with scaled memory."""
        return self.model_copy(update={"memory_gb": max(4, int(self.memory_gb * factor))})

    def scale_walltime(self, factor: float) -> ResourceSpec:
        """Return a new spec with scaled walltime."""
        new_seconds = int(self.walltime.total_seconds() * factor)
        return self.model_copy(update={"walltime": timedelta(seconds=new_seconds)})


class JobState(str, Enum):
    """Job lifecycle states."""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"
    ABANDONED = "abandoned"

    @property
    def is_terminal(self) -> bool:
        return self in (JobState.COMPLETED, JobState.FAILED, JobState.CANCELED, JobState.ABANDONED)


class JobSpec(BaseModel, frozen=True):
    """A single unit of work in the pipeline DAG.

    Parameters are stored as typed fields, not encoded into string keys.
    Dependencies reference parent job UUIDs, not reconstructed paths.
    """

    id: UUID = Field(default_factory=uuid4)
    name: str  # human-readable label (e.g. "hcrl_sa/large/vgae_autoencoder/seed_42")
    executable: str = ""  # populated by executor; planner can leave blank
    arguments: list[str] = []

    # Domain parameters — queryable fields, not key-encoded
    parameters: dict[str, Any] = {}

    # Resources
    resources: ResourceSpec = ResourceSpec()

    # DAG: parent UUIDs (WfFormat-style opaque references)
    parents: list[UUID] = []

    # Environment and metadata
    environment: dict[str, str] = {}
    tags: dict[str, str] = {}

    def with_parents(self, parent_ids: list[UUID]) -> JobSpec:
        """Return a copy with updated parent list."""
        return self.model_copy(update={"parents": parent_ids})

    def with_resources(self, resources: ResourceSpec) -> JobSpec:
        """Return a copy with updated resources."""
        return self.model_copy(update={"resources": resources})
