"""Shared fixtures for orchestration tests."""

from __future__ import annotations

import pytest

from graphids.pipeline.orchestration.dag import _normalize


@pytest.fixture()
def gpu_resources() -> dict:
    """Standard GPU resource dict for tests."""
    return _normalize({"partition": "gpu", "gpus": 1, "cpus": 4, "mem": "16G", "walltime": "3:00:00"})


@pytest.fixture()
def cpu_resources() -> dict:
    """Standard CPU resource dict for tests."""
    return _normalize({"partition": "cpu", "gpus": 0, "cpus": 8, "mem": "32G", "walltime": "1:00:00"})
