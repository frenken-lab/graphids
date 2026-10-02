"""Fresh-process temporal trials and equal-work comparison checks."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

from graphids.exp.temporal_results import load_result


def validate_trial(result, reference=None):
    coverage = result["coverage"]
    for key in ("duplicate", "unexpected", "incorrectly_scored_warmup_or_padding",
                "duplicate_successor_targets", "unexpected_successor_targets"):
        if coverage[key]:
            raise ValueError(f"invalid sweep coverage: {key}")
    if coverage["unique"] != result["planned_scored_events"]:
        raise ValueError("trial missed scored core events")
    if coverage["observed_successor_targets"] != result["planned_successor_targets"]:
        raise ValueError("trial missed eligible successor targets")
    if coverage["warmup_events"] != result["planned_warmup_events"]:
        raise ValueError("trial missed replay events")
    if reference is not None:
        for key in ("initial_weight_sha256", "section_queue_sha256", "planned_scored_events", "planned_warmup_events", "planned_successor_targets"):
            if result[key] != reference[key]:
                raise ValueError(f"sweep trials differ in {key}")
        if result["cuda_device"] != reference["cuda_device"]:
            raise ValueError("sweep trials must use the same GPU model")


def run_trial(root, name, *, config=None, config_path=None, section_limit=None, profile=False, reference=None):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if config is not None:
        config_path = root / (name + ".yml")
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    command = [sys.executable, "-m", "graphids.exp.temporal_comparison", str(config_path), "--output", str(root / name)]
    if section_limit is not None:
        command.extend(["--section-limit", str(section_limit)])
    if profile:
        command.append("--profile")
    subprocess.run(command, check=True)
    result = load_result(root, name)
    validate_trial(result, reference)
    return result
