"""Tests for resource profile parsing (dag.py _normalize)."""

from __future__ import annotations

import pytest

from graphids.pipeline.orchestration.dag import _normalize, scale_resources


# ---------------------------------------------------------------------------
# _normalize (YAML dict → submitit-ready dict)
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
def test_normalize(yaml_input, expected_mem, expected_partition):
    res = _normalize(yaml_input)
    assert res["memory_gb"] == expected_mem
    assert res["partition"] == expected_partition


def test_normalize_walltime_parsing():
    res = _normalize({"walltime": "2:30:00"})
    assert res["walltime_min"] == 150


# ---------------------------------------------------------------------------
# scale_resources
# ---------------------------------------------------------------------------


def test_oom_doubles_memory():
    res = _normalize({"mem": "20G"})
    scaled = scale_resources(res, "OUT_OF_MEMORY")
    assert scaled["memory_gb"] == 40


def test_timeout_scales_time():
    res = _normalize({"walltime": "2:00:00"})
    scaled = scale_resources(res, "TIMEOUT")
    assert scaled["walltime_min"] == 180


def test_unknown_reason_no_change():
    res = _normalize({"mem": "20G", "walltime": "3:00:00"})
    scaled = scale_resources(res, "UNKNOWN_REASON")
    assert scaled == res
