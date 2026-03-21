"""Shared fixtures for orchestration tests."""

from __future__ import annotations

import pytest


@pytest.fixture()
def gpu_resources() -> dict:
    return {"partition": "gpu", "gpus": 1, "cpus": 4, "memory_gb": 16, "walltime_min": 180}


@pytest.fixture()
def cpu_resources() -> dict:
    return {"partition": "cpu", "gpus": 0, "cpus": 8, "memory_gb": 32, "walltime_min": 60}
