"""Resource profiles and adaptive retry scaling from resources.yaml."""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from pathlib import Path

import yaml

_RESOURCES_PATH = Path(__file__).resolve().parents[1] / "config" / "resources.yaml"


def _detect_cluster() -> str:
    """Detect OSC cluster from hostname, with env var override."""
    override = os.environ.get("KD_GAT_CLUSTER")
    if override:
        return override.lower()
    host = socket.gethostname().lower()
    for name in ("cardinal", "ascend", "pitzer"):
        if name in host:
            return name
    return "pitzer"


@dataclass
class ResourceSpec:
    partition: str
    time: str
    mem: str
    cpus_per_task: int
    num_workers: int
    gres: str = ""

    @property
    def mem_mb(self) -> int:
        s = self.mem.upper().rstrip("BMG")
        if self.mem.upper().endswith("G"):
            return int(s) * 1024
        return int(s)

    @property
    def time_minutes(self) -> int:
        parts = self.time.split(":")
        return int(parts[0]) * 60 + int(parts[1])


def _load() -> dict:
    return yaml.safe_load(_RESOURCES_PATH.read_text())


def get_resources(model_type: str, scale: str, stage: str) -> ResourceSpec:
    """Look up resource profile for (model_type, scale, stage).

    Resolves cluster-agnostic ``mode`` field to concrete ``partition``/``gres``
    using the ``clusters`` mapping + hostname detection.
    """
    raw = _load()
    profiles = raw["resource_profiles"]
    try:
        spec = dict(profiles[model_type][scale][stage])
    except KeyError:
        raise KeyError(
            f"No resource profile for ({model_type}, {scale}, {stage}). "
            f"Add entry to config/resources.yaml."
        ) from None

    mode = spec.pop("mode", None)
    if mode and "partition" not in spec:
        cluster = _detect_cluster()
        cluster_map = raw.get("clusters", {}).get(cluster)
        if not cluster_map:
            raise KeyError(f"No cluster config for '{cluster}' in resources.yaml")
        mode_spec = cluster_map.get(mode)
        if not mode_spec:
            raise KeyError(f"No mode '{mode}' for cluster '{cluster}' in resources.yaml")
        spec["partition"] = mode_spec["partition"]
        spec["gres"] = mode_spec.get("gres", "")

    return ResourceSpec(**spec)


def get_failure_reactions() -> dict:
    return _load().get("failure_reactions", {})


def scale_resources(spec: ResourceSpec, failure_reason: str) -> ResourceSpec:
    """Apply failure reaction scaling. Returns new ResourceSpec."""
    reactions = get_failure_reactions()
    reaction = reactions.get(failure_reason, {})
    if not reaction:
        return spec

    mem = spec.mem
    time = spec.time
    if "scale_mem" in reaction:
        new_mb = int(spec.mem_mb * reaction["scale_mem"])
        mem = f"{new_mb // 1024}G"
    if "scale_time" in reaction:
        new_min = int(spec.time_minutes * reaction["scale_time"])
        h, m = divmod(new_min, 60)
        time = f"{h:02d}:{m:02d}:00"

    return ResourceSpec(
        partition=spec.partition, time=time, mem=mem,
        cpus_per_task=spec.cpus_per_task, num_workers=spec.num_workers,
        gres=spec.gres,
    )
