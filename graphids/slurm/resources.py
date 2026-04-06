"""SLURM resource spec + resource profiles + adaptive retry scaling.

``ResourceSpec`` is the pure SLURM-facing data model (partition, time,
mem, cpus, workers, gres). It lives here — alongside the profile lookup
and override functions that produce and mutate it — rather than in
``graphids.config``. SLURM resources are a SLURM concern.
"""

from __future__ import annotations

import dataclasses
import os
import re
import socket
from dataclasses import dataclass

from graphids.config.constants import PROJECT_ROOT

_CLUSTERS_PATH = PROJECT_ROOT / "configs" / "resources" / "clusters.json"
_JOB_PROFILES_PATH = PROJECT_ROOT / "configs" / "resources" / "job_profiles.json"
_SUBMIT_PROFILES_PATH = PROJECT_ROOT / "configs" / "resources" / "submit_profiles.json"

_SLURM_TIME_RE = re.compile(
    r"^(?:\d+-)?(?:\d{1,2}:)?\d{2}:\d{2}$"  # [D-]HH:MM:SS or MM:SS
)


@dataclass(frozen=True)
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


def _load_clusters() -> dict:
    import json

    return json.loads(_CLUSTERS_PATH.read_text())


def _load_profiles() -> dict:
    import json

    if not _JOB_PROFILES_PATH.exists():
        raise FileNotFoundError(f"Missing job_profiles.json at {_JOB_PROFILES_PATH}")
    return json.loads(_JOB_PROFILES_PATH.read_text())


def _load_profile(family: str) -> dict:
    profiles = _load_profiles()
    profile = profiles.get(family)
    if not profile:
        raise KeyError(
            f"No resource profile for '{family}' in job_profiles.json at {_JOB_PROFILES_PATH}"
        )
    return profile


def get_resources(model_type: str, scale: str, stage: str) -> ResourceSpec:
    """Look up resource profile for (model_type, scale, stage).

    Resolves cluster-agnostic ``mode`` field to concrete ``partition``/``gres``
    using the ``clusters`` mapping + hostname detection.
    """
    # Fusion uses method-specific profiles under job_profiles.json.
    family = (
        "fusion" if model_type in {"bandit", "dqn", "mlp", "weighted_avg", "fusion"} else model_type
    )
    raw_profile = _load_profile(family).get("resources", {})
    by_scale = raw_profile.get("by_scale", {})

    try:
        stage_spec = dict(by_scale[scale][stage])
    except KeyError:
        raise KeyError(
            f"No resource profile for ({model_type}, {scale}, {stage}). "
            f"Add entry to configs/resources/job_profiles.json for '{family}'."
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
        clusters = _load_clusters()
        cluster_map = clusters.get("execution_modes", {}).get(cluster)
        if not cluster_map:
            default_cluster = clusters.get("default", "")
            cluster_map = clusters.get("execution_modes", {}).get(default_cluster)
        if not cluster_map:
            raise KeyError(f"No cluster config for '{cluster}' in clusters.json")
        mode_spec = cluster_map.get(mode)
        if not mode_spec:
            raise KeyError(f"No mode '{mode}' for cluster '{cluster}' in clusters.json")
        spec["partition"] = mode_spec["partition"]
        spec["gres"] = mode_spec.get("gres", "")
        mem_per_cpu = mode_spec.get("mem_per_cpu")  # MB, from clusters.json

    spec["cpus_per_task"] = spec.pop("cpus")
    spec["num_workers"] = spec.pop("workers")

    # Derive mem from cpus × mem_per_cpu when not declared in profile.
    # Profiles that need less (e.g. fusion with workers: 0) set mem explicitly.
    if "mem" not in spec and mem_per_cpu is not None:
        max_mem_mb = spec["cpus_per_task"] * mem_per_cpu
        spec["mem"] = f"{max_mem_mb // 1024}G"
    elif "mem" not in spec:
        raise KeyError(
            f"Profile for ({model_type}, {scale}, {stage}) has no 'mem' and "
            f"no mem_per_cpu from cluster config to derive it."
        )

    result = ResourceSpec(**spec)

    # Validate explicit mem doesn't exceed cluster limit.
    if mem_per_cpu is not None:
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
        partition=spec.partition,
        time=time,
        mem=mem,
        cpus_per_task=spec.cpus_per_task,
        num_workers=spec.num_workers,
        gres=spec.gres,
    )


def apply_resource_overrides(
    spec: ResourceSpec,
    overrides: dict[str, str | int],
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
            f"Unknown resource override keys: {sorted(unknown)}. Valid: {sorted(valid)}"
        )
    return dataclasses.replace(spec, **overrides)


# ---------------------------------------------------------------------------
# submit_profile printer — consumed by scripts/slurm/submit.sh via `read`
# ---------------------------------------------------------------------------


def _resolve_gres(mode: str) -> str:
    """Look up gres from clusters.json execution_modes for the current cluster.

    ``mode`` is the cluster-agnostic profile key (e.g. ``gpu`` / ``cpu``).
    Non-GPU modes short-circuit to ``NONE``.
    """
    if not mode.startswith("gpu"):
        return "NONE"
    cluster = _detect_cluster()
    clusters = _load_clusters()
    exec_modes = clusters.get("execution_modes", {}).get(cluster, {})
    # profile mode "gpu" maps to "gpu_train" execution mode key
    gpu_mode = exec_modes.get("gpu_train", {})
    return gpu_mode.get("gres", "NONE")


def print_submit_profile(job: str | None) -> None:
    """Print the SLURM profile line for ``scripts/slurm/submit.sh`` to stdout.

    Output (space-separated, consumed by submit.sh via ``read``):
        partition cpus mem time signal mode gres command...

    When ``job`` is ``None`` or not in ``submit_profiles.json``, prints
    the available profiles to stderr and exits with status 1.
    """
    import json
    import sys

    profiles = json.loads(_SUBMIT_PROFILES_PATH.read_text()).get("submit_profiles", {})

    if job is None:
        print("Available profiles:", ", ".join(sorted(profiles)), file=sys.stderr)
        sys.exit(1)

    if job not in profiles:
        print(f"Unknown profile: {job}", file=sys.stderr)
        print("Available:", ", ".join(sorted(profiles)), file=sys.stderr)
        sys.exit(1)

    p = profiles[job]
    gres = _resolve_gres(p["mode"])

    parts = [
        p["partition"],
        str(p["cpus"]),
        p["mem"],
        p["time"],
        p.get("signal", "") or "NONE",
        p["mode"],
        gres,
        p["command"],
    ]
    print(" ".join(parts))
