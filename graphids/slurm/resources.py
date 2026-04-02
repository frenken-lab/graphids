"""Resource profiles and adaptive retry scaling from compact resources config."""

from __future__ import annotations

import dataclasses
import os
import socket
from dataclasses import dataclass
from pathlib import Path

from graphids.config.yaml_utils import read_yaml

_RESOURCES_DIR = Path(__file__).resolve().parents[1] / "config" / "resources"
_CLUSTERS_PATH = _RESOURCES_DIR / "clusters.yaml"
_PROFILES_DIR = _RESOURCES_DIR / "profiles"


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


import re

_SLURM_TIME_RE = re.compile(
    r"^(?:\d+-)?(?:\d{1,2}:)?\d{2}:\d{2}$"  # [D-]HH:MM:SS or MM:SS
)


@dataclass
class ResourceSpec:
    partition: str
    time: str
    mem: str
    cpus_per_task: int
    num_workers: int
    gres: str = ""

    def __post_init__(self) -> None:
        if not _SLURM_TIME_RE.match(self.time):
            raise ValueError(
                f"Invalid SLURM time format: {self.time!r}. "
                f"Expected H:MM:SS, HH:MM:SS, or D-HH:MM:SS."
            )

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


def _load_clusters() -> dict:
    return read_yaml(_CLUSTERS_PATH)


def _load_profile(family: str) -> dict:
    path = _PROFILES_DIR / f"{family}.yaml"
    if not path.exists():
        raise KeyError(f"No resource profile file for '{family}' at {path}")
    return read_yaml(path)


def get_resources(model_type: str, scale: str, stage: str) -> ResourceSpec:
    """Look up resource profile for (model_type, scale, stage).

    Resolves cluster-agnostic ``mode`` field to concrete ``partition``/``gres``
    using the ``clusters`` mapping + hostname detection.
    """
    # Fusion uses method-specific profiles under resources/profiles/fusion.yaml.
    family = "fusion" if model_type in {"bandit", "dqn", "mlp", "weighted_avg", "fusion"} else model_type
    raw_profile = _load_profile(family).get("resources", {})
    by_scale = raw_profile.get("by_scale", {})

    try:
        stage_spec = dict(by_scale[scale][stage])
    except KeyError:
        raise KeyError(
            f"No resource profile for ({model_type}, {scale}, {stage}). "
            f"Add entry to config/resources/profiles/{family}.yaml."
        ) from None

    if family == "fusion":
        try:
            spec = dict(stage_spec["by_method"][model_type])
        except KeyError:
            raise KeyError(
                f"No fusion resource profile for method={model_type}, scale={scale}, stage={stage}."
            ) from None
    else:
        spec = stage_spec

    mode = spec.pop("mode", None)
    mem_per_cpu = None
    if mode and "partition" not in spec:
        cluster = _detect_cluster()
        clusters = _load_clusters().get("clusters", {})
        cluster_map = clusters.get("execution_modes", {}).get(cluster)
        if not cluster_map:
            default_cluster = clusters.get("default", "")
            cluster_map = clusters.get("execution_modes", {}).get(default_cluster)
        if not cluster_map:
            raise KeyError(f"No cluster config for '{cluster}' in clusters.yaml")
        mode_spec = cluster_map.get(mode)
        if not mode_spec:
            raise KeyError(f"No mode '{mode}' for cluster '{cluster}' in clusters.yaml")
        spec["partition"] = mode_spec["partition"]
        spec["gres"] = mode_spec.get("gres", "")
        mem_per_cpu = mode_spec.get("mem_per_cpu")  # MB, from clusters.yaml

    spec["cpus_per_task"] = spec.pop("cpus")
    spec["num_workers"] = spec.pop("workers")
    result = ResourceSpec(**spec)

    # Validate memory fits within SLURM's per-CPU limit for the resolved partition.
    if mode and mem_per_cpu is not None:
        max_mem_mb = result.cpus_per_task * mem_per_cpu
        if result.mem_mb > max_mem_mb:
            raise ValueError(
                f"Requested mem={result.mem} ({result.mem_mb} MB) exceeds "
                f"cluster limit: {result.cpus_per_task} CPUs × "
                f"{mem_per_cpu} MB/CPU = {max_mem_mb} MB "
                f"({max_mem_mb // 1024}G). "
                f"Increase cpus or reduce mem."
            )

    return result


def get_failure_reactions() -> dict:
    return {
        "OUT_OF_MEMORY": {"max_retries": 2, "scale_mem": 1.5},
        "TIMEOUT": {"max_retries": 2, "scale_time": 1.5},
    }


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


def apply_resource_overrides(
    spec: ResourceSpec, overrides: dict[str, str | int],
) -> ResourceSpec:
    """Apply recipe-level resource overrides onto a ResourceSpec.

    Only known ResourceSpec fields are accepted. Unknown keys raise ValueError.
    """
    if not overrides:
        return spec
    valid = {f.name for f in dataclasses.fields(ResourceSpec)}
    unknown = set(overrides) - valid
    if unknown:
        raise ValueError(
            f"Unknown resource override keys: {sorted(unknown)}. "
            f"Valid: {sorted(valid)}"
        )
    return dataclasses.replace(spec, **overrides)
