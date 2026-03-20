"""Shared fixtures for orchestration tests.

Resource spec factories avoid repeating timedelta boilerplate.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from graphids.pipeline.orchestration.job import ResourceSpec


@pytest.fixture()
def gpu_resources() -> ResourceSpec:
    """Standard GPU resource spec for tests."""
    return ResourceSpec(
        partition="gpu", gpus=1, cpus=4,
        memory_gb=16, walltime=timedelta(hours=3),
    )


@pytest.fixture()
def cpu_resources() -> ResourceSpec:
    """Standard CPU resource spec for tests."""
    return ResourceSpec(
        partition="cpu", gpus=0, cpus=8,
        memory_gb=32, walltime=timedelta(hours=1),
    )
