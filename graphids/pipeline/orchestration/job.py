"""Job definition models for pipeline state tracking.

Pydantic v2 frozen models for describing jobs, resources, and state.
Used by sweep_pipeline for resumable HPO orchestration.
"""

from __future__ import annotations

from datetime import timedelta
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ResourceSpec(BaseModel, frozen=True):
    """Resource requirements for a pipeline job."""

    nodes: int = 1
    gpus: int = 0
    cpus: int = 4
    memory_gb: int = 20
    walltime: timedelta = timedelta(hours=3)


class JobState(str, Enum):
    """Job lifecycle states."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"

    @property
    def is_terminal(self) -> bool:
        return self in (JobState.COMPLETED, JobState.FAILED, JobState.CANCELED)


class JobSpec(BaseModel, frozen=True):
    """A single unit of work in the pipeline DAG."""

    id: UUID = Field(default_factory=uuid4)
    name: str
    executable: str = ""
    arguments: list[str] = []
    parameters: dict[str, Any] = {}
    resources: ResourceSpec = ResourceSpec()
    parents: list[UUID] = []
    environment: dict[str, str] = {}
    tags: dict[str, str] = {}
