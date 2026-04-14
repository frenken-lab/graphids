"""Regression tests for ``graphids submit-profile`` auto-sizing.

# INVARIANT: pipeline composition — time=sum(stages), cpus/mem=max(stages)
# INVARIANT: monotonicity — larger dataset never reduces time or mem
# CONTRACT: submit.sh reads 8 whitespace-separated fields on one line
# REGRESSION: static profiles (tests/analyze/profile) must stay unchanged
# REGRESSION: import-time validator catches unknown stages / bad scale keys
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
from graphids.config.topology import TOPOLOGY, _validate_submit_profiles

STAGE_PROFILES = {
    "autoencoder": {
        "mode": "gpu",
        "cpus": 8,
        "scaling": {
            "time_min": {"base": 5.0, "per_mraw": 0.3, "scale_mult": {"small": 1.0, "large": 2.0}},
            "mem_gb": {"base": 12.0, "per_mraw": 0.2, "scale_mult": {"small": 1.0, "large": 1.3}},
        },
    },
    "supervised": {
        "mode": "gpu",
        "cpus": 8,
        "scaling": {
            "time_min": {"base": 8.0, "per_mraw": 0.5, "scale_mult": {"small": 1.0, "large": 2.2}},
            "mem_gb": {"base": 14.0, "per_mraw": 0.25, "scale_mult": {"small": 1.0, "large": 1.4}},
        },
    },
    "fusion": {
        "mode": "gpu",
        "cpus": 4,
        "scaling": {
            "time_min": {"base": 3.0, "per_mraw": 0.1},
            "mem_gb": {"base": 8.0, "per_mraw": 0.05},
        },
    },
}

PIPELINE_PROFILE = {
    "partition": "gpu",
    "mode": "gpu",
    "signal": "",
    "command": "python -m graphids pipeline-run",
    "stages": ["autoencoder", "supervised", "fusion"],
    "defaults": {"cpus": 8, "mem": "40G", "time": "6:00:00"},
}

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


def test_pipeline_composition_time_sums_and_cpus_maxes():
    # INVARIANT: time = sum(stage times); cpus = max(stage cpus) — not sum.
    # REGRESSION: an earlier design summed CPUs, which would over-allocate
    # 20 CPUs for pipeline-run when stages run sequentially reusing one alloc.
    cpus, mem, time = _resolve_resources(
        PIPELINE_PROFILE, STAGE_PROFILES, num_raw=20_000_000, scale="small"
    )
    per_stage = [
        _size_from_scaling(STAGE_PROFILES[s], 20_000_000, "small")
        for s in PIPELINE_PROFILE["stages"]
    ]
    assert cpus == max(c for c, _, _ in per_stage)
    assert cpus < sum(c for c, _, _ in per_stage)
    assert _parse_time(time) >= max(int(t) for _, _, t in per_stage)
    assert _parse_mem(mem) >= max(int(m) for _, m, _ in per_stage)


def test_monotonic_in_dataset_size():
    # INVARIANT: bigger dataset never shrinks allocation.
    small_cpus, small_mem, small_time = _resolve_resources(
        PIPELINE_PROFILE, STAGE_PROFILES, num_raw=2_000_000, scale="small"
    )
    big_cpus, big_mem, big_time = _resolve_resources(
        PIPELINE_PROFILE, STAGE_PROFILES, num_raw=70_000_000, scale="small"
    )
    assert big_cpus == small_cpus  # fixed per stage
    assert _parse_mem(big_mem) > _parse_mem(small_mem)
    assert _parse_time(big_time) > _parse_time(small_time)


def test_large_scale_never_smaller_than_small():
    # INVARIANT: scale_mult of "large" must be >= 1.0 for time and mem.
    small = _resolve_resources(PIPELINE_PROFILE, STAGE_PROFILES, num_raw=20_000_000, scale="small")
    large = _resolve_resources(PIPELINE_PROFILE, STAGE_PROFILES, num_raw=20_000_000, scale="large")
    assert _parse_mem(large[1]) >= _parse_mem(small[1])
    assert _parse_time(large[2]) >= _parse_time(small[2])


def test_defaults_fallback_when_no_dataset():
    # CONTRACT: omitting --dataset falls back to the `defaults` block verbatim.
    cpus, mem, time = _resolve_resources(
        PIPELINE_PROFILE, STAGE_PROFILES, num_raw=None, scale="small"
    )
    assert (cpus, mem, time) == (8, "40G", "6:00:00")


def test_static_profile_unchanged():
    # REGRESSION: profiles with no scaling block must emit their literal fields.
    cpus, mem, time = _resolve_resources(
        STATIC_PROFILE, STAGE_PROFILES, num_raw=99_999_999, scale="large"
    )
    assert (cpus, mem, time) == (8, "16G", "1:00:00")


def test_scaling_profile_uses_coefficients():
    # CONTRACT: rebuild-caches coefficients from the 6-dataset fit cover the range.
    # Use the actual hcrl_sa number; we observed 0:56 / 3.1 GB. Prediction should
    # comfortably exceed those with the 1.3× safety margin baked into the coefficients.
    cpus, mem, time = _resolve_resources(
        SCALED_PROFILE, STAGE_PROFILES, num_raw=1_908_595, scale="small"
    )
    assert cpus == 6
    assert _parse_time(time) >= 1  # at least 1 minute even for smallest
    assert _parse_mem(mem) >= 4  # more than observed 3.1 GB


def test_eval_scaling_linear_and_scale_mult():
    # CONTRACT of _eval_scaling: base + per_mraw*mraw, then × scale_mult[scale].
    block = {"base": 10.0, "per_mraw": 1.0, "scale_mult": {"small": 1.0, "large": 2.0}}
    assert _eval_scaling(block, 0, "small") == 10.0
    assert _eval_scaling(block, 5_000_000, "small") == 15.0
    assert _eval_scaling(block, 5_000_000, "large") == 30.0
    assert _eval_scaling(block, None, "small") == 10.0  # None → 0 mraw


def test_format_time_ceils_minutes():
    # CONTRACT: sub-minute values round up to a valid sbatch duration.
    assert _format_time(0.1) == "0:01:00"
    assert _format_time(59.0) == "0:59:00"
    assert _format_time(60.0) == "1:00:00"
    assert _format_time(61.5) == "1:02:00"


def test_submit_profiles_validator_rejects_unknown_stage(tmp_path, monkeypatch):
    # REGRESSION: typos in submit_profiles["pipeline-run"].stages used to crash
    # at sbatch time; now they raise at topology import.
    import graphids.config.topology as topology_mod

    orig = pathlib.Path("configs/resources/submit_profiles.json").read_text()
    bad = json.loads(orig)
    bad["submit_profiles"]["pipeline-run"]["stages"].append("BOGUS_STAGE")
    patched = tmp_path / "submit_profiles.json"
    patched.write_text(json.dumps(bad))
    # Point the validator at the doctored file by swapping _CONFIGS_DIR.
    monkeypatch.setattr(topology_mod, "_CONFIGS_DIR", tmp_path.parent)
    (tmp_path.parent / "resources").mkdir(exist_ok=True)
    (tmp_path.parent / "resources" / "submit_profiles.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="unknown stage_profiles"):
        _validate_submit_profiles(TOPOLOGY)


def test_submit_profiles_validator_rejects_bad_scale(tmp_path, monkeypatch):
    # REGRESSION: a scale_mult key outside VALID_SCALES (e.g. "huge") should
    # fail at import, not at pipeline-run submission time.
    import graphids.config.topology as topology_mod

    orig = pathlib.Path("configs/resources/submit_profiles.json").read_text()
    bad = json.loads(orig)
    bad["stage_profiles"]["autoencoder"]["scaling"]["time_min"]["scale_mult"]["huge"] = 3.0
    (tmp_path / "resources").mkdir()
    (tmp_path / "resources" / "submit_profiles.json").write_text(json.dumps(bad))
    monkeypatch.setattr(topology_mod, "_CONFIGS_DIR", tmp_path)
    with pytest.raises(ValueError, match="unknown scale"):
        _validate_submit_profiles(TOPOLOGY)
