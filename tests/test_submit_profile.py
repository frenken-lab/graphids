"""Regression tests for ``graphids submit-profile`` auto-sizing.

# INVARIANT: monotonicity — larger dataset never reduces time or mem
# CONTRACT: submit.sh reads 8 whitespace-separated fields on one line
# REGRESSION: static profiles (tests/analyze/profile) must stay unchanged
# REGRESSION: import-time validator catches bad scale_mult keys
"""

from __future__ import annotations

import json
import pathlib

import pytest

from graphids.cli.app import (
    _eval_scaling,
    _format_time,
    _resolve_resources,
    _size_from_scaling,
)
from graphids.config.topology import _validate_submit_profiles

SCALED_PROFILE = {
    "partition": "cpu",
    "mode": "cpu",
    "cpus": 6,
    "signal": "",
    "command": "python -m graphids rebuild-caches",
    "scaling": {
        "time_min": {"base": 2.0, "per_mraw": 0.083},
        "mem_gb": {"base": 5.0, "per_mraw": 0.61},
    },
    "defaults": {"time": "1:00:00", "mem": "54G"},
}

STATIC_PROFILE = {
    "partition": "cpu",
    "mode": "cpu",
    "cpus": 8,
    "mem": "16G",
    "time": "1:00:00",
    "signal": "",
    "command": "python -m pytest",
}


def _parse_time(s: str) -> int:
    h, m, sec = s.split(":")
    return int(h) * 60 + int(m) + (1 if int(sec) else 0)


def _parse_mem(s: str) -> int:
    assert s.endswith("G")
    return int(s[:-1])


def test_monotonic_in_dataset_size():
    # INVARIANT: bigger dataset never shrinks allocation.
    small = _resolve_resources(SCALED_PROFILE, num_raw=2_000_000, scale="small")
    big = _resolve_resources(SCALED_PROFILE, num_raw=70_000_000, scale="small")
    assert big[0] == small[0]  # fixed cpus
    assert _parse_mem(big[1]) > _parse_mem(small[1])
    assert _parse_time(big[2]) > _parse_time(small[2])


def test_defaults_fallback_when_no_dataset():
    # CONTRACT: omitting --dataset falls back to the `defaults` block verbatim.
    cpus, mem, time = _resolve_resources(SCALED_PROFILE, num_raw=None, scale="small")
    assert (cpus, mem, time) == (6, "54G", "1:00:00")


def test_static_profile_unchanged():
    # REGRESSION: profiles with no scaling block must emit their literal fields.
    cpus, mem, time = _resolve_resources(STATIC_PROFILE, num_raw=99_999_999, scale="large")
    assert (cpus, mem, time) == (8, "16G", "1:00:00")


def test_scaling_profile_uses_coefficients():
    # CONTRACT: rebuild-caches coefficients from the 6-dataset fit cover the range.
    cpus, mem, time = _resolve_resources(SCALED_PROFILE, num_raw=1_908_595, scale="small")
    assert cpus == 6
    assert _parse_time(time) >= 1
    assert _parse_mem(mem) >= 4


def test_eval_scaling_linear_and_scale_mult():
    # CONTRACT of _eval_scaling: base + per_mraw*mraw, then × scale_mult[scale].
    block = {"base": 10.0, "per_mraw": 1.0, "scale_mult": {"small": 1.0, "large": 2.0}}
    assert _eval_scaling(block, 0, "small") == 10.0
    assert _eval_scaling(block, 5_000_000, "small") == 15.0
    assert _eval_scaling(block, 5_000_000, "large") == 30.0
    assert _eval_scaling(block, None, "small") == 10.0


def test_format_time_ceils_minutes():
    # CONTRACT: sub-minute values round up to a valid sbatch duration.
    assert _format_time(0.1) == "0:01:00"
    assert _format_time(59.0) == "0:59:00"
    assert _format_time(60.0) == "1:00:00"
    assert _format_time(61.5) == "1:02:00"


def test_size_from_scaling_applies_scale_mult():
    # CONTRACT: _size_from_scaling returns (cpus, mem_gb, time_min) triple.
    cpus, mem_gb, time_min = _size_from_scaling(SCALED_PROFILE, num_raw=10_000_000, scale="small")
    assert cpus == 6
    assert mem_gb > 5.0
    assert time_min > 2.0


def test_submit_profiles_validator_rejects_bad_scale(tmp_path, monkeypatch):
    # REGRESSION: a scale_mult key outside VALID_SCALES should fail at import.
    import graphids.config.topology as topology_mod

    orig = pathlib.Path("configs/resources/submit_profiles.json").read_text()
    bad = json.loads(orig)
    bad["submit_profiles"]["rebuild-caches"]["scaling"]["time_min"]["scale_mult"] = {"huge": 3.0}
    (tmp_path / "resources").mkdir()
    (tmp_path / "resources" / "submit_profiles.json").write_text(json.dumps(bad))
    monkeypatch.setattr(topology_mod, "_CONFIGS_DIR", tmp_path)
    with pytest.raises(ValueError, match="unknown scale"):
        _validate_submit_profiles()
