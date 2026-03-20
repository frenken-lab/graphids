"""SLURM primitives: sbatch generation, submission, polling, adaptive retry.

Low-level building blocks shared by both the Dagster Pipes client
(``pipes_slurm.py``) and the zero-daemon ``fire_and_forget()`` mode.
"""

from __future__ import annotations

import structlog
import subprocess
import time
from datetime import timedelta
from pathlib import Path

import yaml

from graphids.config import CONFIG_DIR, DEFAULT_LAKE_ROOT, PROJECT_ROOT, SLURM_ACCOUNT, SLURM_GPU_TYPE
from ..subprocess_utils import build_cli_cmd

from .job import ResourceSpec

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class SlurmJobFailed(Exception):
    """Raised when a SLURM job reaches a terminal failure state."""

    def __init__(
        self,
        reason: str,
        node: str | None = None,
        ckpt_path: str | None = None,
        metadata: dict | None = None,
    ):
        self.reason = reason
        self.node = node
        self.ckpt_path = ckpt_path
        self.metadata = metadata or {}
        super().__init__(f"SLURM job failed: {reason} (node={node})")


# ---------------------------------------------------------------------------
# Resource profiles (loaded from resources.yaml)
# ---------------------------------------------------------------------------

_RESOURCES_YAML = CONFIG_DIR / "resources.yaml"


def _load_resources_yaml() -> dict:
    """Load the full resources.yaml file."""
    return yaml.safe_load(_RESOURCES_YAML.read_text())


def _parse_resource_profiles(raw: dict) -> dict[tuple[str, str, str], ResourceSpec]:
    """Parse resource_profiles section -> dict[(model, scale, stage), ResourceSpec]."""
    profiles: dict[tuple[str, str, str], ResourceSpec] = {}
    for model, scales in raw.get("resource_profiles", {}).items():
        for scale, stages in scales.items():
            for stage, res in stages.items():
                profiles[(model, scale, stage)] = ResourceSpec.from_yaml(res)
    return profiles


# Load once at import, split into profiles + reactions
_raw_resources = _load_resources_yaml()
RESOURCE_PROFILES = _parse_resource_profiles(_raw_resources)
FAILURE_REACTIONS: dict[str, dict] = _raw_resources.get("failure_reactions", {})
del _raw_resources


def get_resources(model: str, scale: str, stage: str) -> ResourceSpec:
    """Look up resource profile for a (model, scale, stage) tuple."""
    key = (model, scale, stage)
    if key not in RESOURCE_PROFILES:
        available = sorted(RESOURCE_PROFILES.keys())
        raise KeyError(
            f"No resource profile for {key}. "
            f"Add an entry to config/resources.yaml. Available: {available}"
        )
    return RESOURCE_PROFILES[key]


# ---------------------------------------------------------------------------
# Adaptive retry: resource scaling
# ---------------------------------------------------------------------------


def scale_resources(resources: ResourceSpec, failure_reason: str) -> ResourceSpec:
    """Apply failure reaction scaling. OOM -> 2x mem, TIMEOUT -> 1.5x time."""
    reaction = FAILURE_REACTIONS.get(failure_reason, {})
    if not reaction:
        return resources

    updates: dict = {}
    if "scale_mem" in reaction:
        updates["memory_gb"] = int(resources.memory_gb * reaction["scale_mem"])
    if "scale_time" in reaction:
        total_secs = resources.walltime.total_seconds()
        updates["walltime"] = timedelta(seconds=int(total_secs * reaction["scale_time"]))

    return resources.model_copy(update=updates) if updates else resources


# ---------------------------------------------------------------------------
# Sbatch script generation
# ---------------------------------------------------------------------------


def generate_sbatch_script(
    stage: str,
    model: str,
    scale: str,
    dataset: str,
    resources: ResourceSpec,
    *,
    seed: int | None = None,
    auxiliaries: str = "none",
    ckpt_path: str | None = None,
    dependency_job_id: str | None = None,
    project_root: str | None = None,
    production: bool = True,
    extra_env: dict[str, str] | None = None,
) -> str:
    """Generate a complete sbatch script for one pipeline stage.

    Parameters
    ----------
    extra_env : dict[str, str] | None
        Additional environment variables to export (e.g. Pipes context vars).
    """
    root = project_root or str(PROJECT_ROOT)
    cli_command = " ".join(build_cli_cmd(
        stage=stage, model=model, scale=scale, dataset=dataset,
        seed=seed, auxiliaries=auxiliaries,
    ))

    lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --account={SLURM_ACCOUNT}",
        f"#SBATCH --partition={resources.partition}",
    ]
    if resources.gpus > 0:
        lines.append(f"#SBATCH --gres=gpu:{SLURM_GPU_TYPE}:{resources.gpus}")
    lines.extend([
        "#SBATCH --nodes=1 --ntasks=1",
        f"#SBATCH --cpus-per-task={resources.cpus}",
        f"#SBATCH --mem={resources.mem_slurm}",
        f"#SBATCH --time={resources.walltime_slurm}",
        f"#SBATCH --job-name=kd-gat-{stage}-{model}-{scale}",
        "#SBATCH --output=slurm_logs/dagster_%j.out",
        "#SBATCH --error=slurm_logs/dagster_%j.err",
        "#SBATCH --signal=B:USR1@180",
    ])
    if resources.exclude_nodes:
        lines.append(f"#SBATCH --exclude={resources.exclude_nodes}")
    if dependency_job_id:
        lines.append(f"#SBATCH --dependency=afterok:{dependency_job_id}")

    lines.extend(["", f'cd "{root}"', "source scripts/slurm/_preamble.sh", ""])
    if production:
        lines.append("export KD_GAT_PRODUCTION=1")
    if ckpt_path:
        lines.append(f'export KD_GAT_CKPT_PATH="{ckpt_path}"')
    if extra_env:
        for k, v in sorted(extra_env.items()):
            lines.append(f'export {k}="{v}"')
    lines.append("")
    if resources.gpus == 0:
        lines.extend(["export SKIP_CUDA_CONF=1", ""])

    lines.extend([
        f"{cli_command} &",
        "_KD_CHILD_PID=$!",
        "wait $_KD_CHILD_PID",
        "EXIT_CODE=$?",
        "",
        "source scripts/slurm/_epilog.sh",
        "exit $EXIT_CODE",
    ])
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# SLURM submission and polling
# ---------------------------------------------------------------------------

TERMINAL_STATES = frozenset({
    "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT",
    "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED",
})


def submit_sbatch(script_path: Path | str, *, cwd: str | None = None) -> str:
    """Submit an sbatch script, return the job ID."""
    result = subprocess.run(
        ["sbatch", "--parsable", str(script_path)],
        capture_output=True, text=True, check=True,
        cwd=cwd or str(PROJECT_ROOT),
    )
    return result.stdout.strip().split(";")[0]


def sacct_query(job_id: str) -> tuple[str, str, str]:
    """Query sacct for (state, reason, node_name)."""
    result = subprocess.run(
        ["sacct", "-j", job_id, "-X", "--parsable2", "--noheader", "-o", "State,Reason,NodeList"],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return "PENDING", "", ""
    parts = result.stdout.strip().split("\n")[0].split("|")
    state = parts[0].split()[0] if parts else "UNKNOWN"
    return state, parts[1] if len(parts) > 1 else "", parts[2] if len(parts) > 2 else ""


def poll_until_done(job_id: str, *, poll_interval: int = 30) -> tuple[str, str, str]:
    """Poll sacct until job reaches a terminal state."""
    while True:
        state, reason, node = sacct_query(job_id)
        if state in TERMINAL_STATES:
            log.info("job_terminal", job_id=job_id, state=state, reason=reason)
            return state, reason, node
        log.debug("job_polling", job_id=job_id, state=state, poll_interval_s=poll_interval)
        time.sleep(poll_interval)


def write_script_file(
    script: str,
    scripts_dir: Path,
    model: str,
    scale: str,
    stage: str,
    auxiliaries: str = "none",
) -> Path:
    """Write an sbatch script to disk, return the path."""
    aux_tag = f"_{auxiliaries}" if auxiliaries != "none" else ""
    script_path = scripts_dir / f"dagster_{model}_{scale}_{stage}{aux_tag}.sbatch"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(script)
    return script_path
