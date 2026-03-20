"""Tests for ResourceSpec Pydantic model (job.py).

Covers YAML factory, immutability, defaults.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from graphids.pipeline.orchestration.job import ResourceSpec


# ---------------------------------------------------------------------------
# from_yaml factory
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("yaml_input", "expected_mem", "expected_partition"),
    [
        pytest.param(
            {"partition": "gpu", "gpus": 1, "cpus": 4, "mem": "20G", "walltime": "3:00:00"},
            20, "gpu",
            id="mem-string",
        ),
        pytest.param(
            {"partition": "cpu", "gpus": 0, "cpus": 8, "memory_gb": 32, "walltime": "1:00:00"},
            32, "cpu",
            id="memory-gb-int",
        ),
        pytest.param(
            {"mem": "512M"},
            1, "cpu",  # 512M rounds to 0, clamped to 1
            id="megabytes-clamp",
        ),
        pytest.param(
            {},
            20, "cpu",  # all defaults
            id="empty-defaults",
        ),
    ],
)
def test_from_yaml(yaml_input, expected_mem, expected_partition):
    res = ResourceSpec.from_yaml(yaml_input)
    assert res.memory_gb == expected_mem
    assert res.partition == expected_partition
    assert isinstance(res, ResourceSpec)


def test_from_yaml_walltime_parsing():
    res = ResourceSpec.from_yaml({"walltime": "2:30:00"})
    assert res.walltime == timedelta(hours=2, minutes=30)


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------


def test_frozen_rejects_mutation(gpu_resources):
    with pytest.raises(Exception):
        gpu_resources.gpus = 2  # type: ignore[misc]


def test_model_copy_preserves_type(gpu_resources):
    updated = gpu_resources.model_copy(update={"memory_gb": 64})
    assert isinstance(updated, ResourceSpec)
    assert updated.memory_gb == 64
    assert updated.gpus == gpu_resources.gpus  # other fields unchanged


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_default_exclude_nodes():
    assert ResourceSpec().exclude_nodes == ""


def test_default_partition():
    assert ResourceSpec().partition == "cpu"
