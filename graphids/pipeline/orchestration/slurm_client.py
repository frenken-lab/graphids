"""SLURM job management: sbatch generation, submission, polling, adaptive retry.

Provides both low-level primitives (submit_sbatch, sacct_query) and the
high-level PipesSlurmClient (submit + poll + validate artifacts + checkpoint
discovery). Used by Dagster assets and fire-and-forget mode.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from datetime import timedelta
from pathlib import Path

import yaml

from graphids.config import PROJECT_ROOT, SLURM_ACCOUNT, SLURM_GPU_TYPE
from graphids.pipeline.subprocess_utils import build_cli_cmd

from .job import ResourceSpec

log = logging.getLogger(__name__)


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

_RESOURCES_YAML = Path(__file__).resolve().parents[2] / "config" / "resources.yaml"


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
# Adaptive retry: resource scaling + state persistence
# ---------------------------------------------------------------------------

_RETRY_STATE_DIR = Path(PROJECT_ROOT) / "slurm_logs" / "dagster_retry"


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


def save_retry_state(
    asset_key: str, reason: str, node: str | None = None, ckpt_path: str | None = None
) -> None:
    """Write retry metadata to slurm_logs/dagster_retry/{asset_key}.json."""
    _RETRY_STATE_DIR.mkdir(parents=True, exist_ok=True)
    state = {"reason": reason, "node": node, "ckpt_path": ckpt_path}
    (_RETRY_STATE_DIR / f"{asset_key}.json").write_text(json.dumps(state, indent=2))


def load_retry_state(asset_key: str) -> dict | None:
    """Read retry metadata from previous attempt, if any."""
    path = _RETRY_STATE_DIR / f"{asset_key}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def clear_retry_state(asset_key: str) -> None:
    """Remove retry state after successful completion."""
    path = _RETRY_STATE_DIR / f"{asset_key}.json"
    if path.exists():
        path.unlink()


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
) -> str:
    """Generate a complete sbatch script for one pipeline stage."""
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
            log.info("Job %s reached state: %s (reason: %s)", job_id, state, reason)
            return state, reason, node
        log.debug("Job %s state: %s, polling in %ds...", job_id, state, poll_interval)
        time.sleep(poll_interval)


# ---------------------------------------------------------------------------
# PipesSlurmClient: high-level submit + poll + validate
# ---------------------------------------------------------------------------


class PipesSlurmClient:
    """Submit sbatch jobs and poll via sacct. Validates artifacts on completion."""

    def __init__(
        self,
        project_root: str | None = None,
        poll_interval: int = 30,
        dry_run: bool = False,
    ):
        self.project_root = project_root or str(PROJECT_ROOT)
        self.poll_interval = poll_interval
        self.dry_run = dry_run
        self._scripts_dir = Path(self.project_root) / "slurm_logs"

    def _write_script(
        self,
        stage: str, model: str, scale: str, dataset: str, resources: ResourceSpec,
        *, seed: int | None = None, auxiliaries: str = "none",
        ckpt_path: str | None = None, dependency_job_id: str | None = None,
    ) -> tuple[str, Path]:
        """Generate sbatch script, write to disk, return (content, path)."""
        script = generate_sbatch_script(
            stage=stage, model=model, scale=scale, dataset=dataset, resources=resources,
            seed=seed, auxiliaries=auxiliaries, ckpt_path=ckpt_path,
            dependency_job_id=dependency_job_id, project_root=self.project_root,
        )
        aux_tag = f"_{auxiliaries}" if auxiliaries != "none" else ""
        script_path = self._scripts_dir / f"dagster_{model}_{scale}_{stage}{aux_tag}.sbatch"
        self._scripts_dir.mkdir(parents=True, exist_ok=True)
        script_path.write_text(script)
        return script, script_path

    def run(
        self,
        stage: str, model: str, scale: str, dataset: str, resources: ResourceSpec,
        *, seed: int | None = None, auxiliaries: str = "none",
        ckpt_path: str | None = None, dependency_job_id: str | None = None,
    ) -> dict:
        """Submit a SLURM job and poll until completion.

        Returns metadata dict. Raises SlurmJobFailed on failure.
        """
        script, script_path = self._write_script(
            stage, model, scale, dataset, resources,
            seed=seed, auxiliaries=auxiliaries,
            ckpt_path=ckpt_path, dependency_job_id=dependency_job_id,
        )
        if self.dry_run:
            log.info("[DRY RUN] Would submit:\n%s", script)
            return {"job_id": "dry-run", "state": "DRY_RUN", "script_path": str(script_path)}

        job_id = submit_sbatch(script_path, cwd=self.project_root)
        log.info("Submitted SLURM job %s for %s/%s/%s", job_id, model, scale, stage)

        t0 = time.monotonic()
        state, reason, node = poll_until_done(job_id, poll_interval=self.poll_interval)
        elapsed = time.monotonic() - t0

        metadata = {
            "job_id": job_id, "state": state, "reason": reason, "node": node,
            "elapsed_seconds": round(elapsed, 1), "script_path": str(script_path),
        }

        if state == "COMPLETED":
            metadata["artifacts_valid"] = self._validate_artifacts(
                dataset, model, scale, stage, auxiliaries,
            )
            return metadata

        ckpt = self._find_checkpoint(dataset, model, scale, stage, auxiliaries)
        raise SlurmJobFailed(reason=state, node=node, ckpt_path=ckpt, metadata=metadata)

    def submit_no_poll(
        self,
        stage: str, model: str, scale: str, dataset: str, resources: ResourceSpec,
        *, seed: int | None = None, auxiliaries: str = "none",
        dependency_job_id: str | None = None,
    ) -> str:
        """Submit a job without polling. Returns job ID."""
        _script, script_path = self._write_script(
            stage, model, scale, dataset, resources,
            seed=seed, auxiliaries=auxiliaries, dependency_job_id=dependency_job_id,
        )
        if self.dry_run:
            log.info("[DRY RUN] Would submit: %s", script_path)
            return "dry-run"
        job_id = submit_sbatch(script_path, cwd=self.project_root)
        log.info("Submitted (no-poll) SLURM job %s for %s/%s/%s", job_id, model, scale, stage)
        return job_id

    def _validate_artifacts(
        self, dataset: str, model: str, scale: str, stage: str,
        auxiliaries: str, seed: int = 42,
    ) -> bool:
        """Check that expected artifacts exist after stage completion."""
        from pydantic import ValidationError

        from graphids.config import EvaluationArtifact, TrainingArtifact, lake_run_dir

        if stage == "preprocess":
            return True
        aux = auxiliaries if auxiliaries != "none" else ""
        stage_path = lake_run_dir(
            lake_root=os.environ.get("KD_GAT_LAKE_ROOT", "experimentruns"),
            dataset=dataset, model_type=model, scale=scale,
            stage=stage, aux=aux, seed=seed, production=True,
        )
        if not stage_path.exists():
            return False
        contract = EvaluationArtifact if stage == "evaluation" else TrainingArtifact
        try:
            contract.from_stage_dir(stage_path)
            return True
        except (ValidationError, ValueError):
            return False

    def _find_checkpoint(
        self, dataset: str, model: str, scale: str, stage: str,
        auxiliaries: str, seed: int = 42,
    ) -> str | None:
        """Look for a Lightning auto-save checkpoint after TIMEOUT."""
        from graphids.config import lake_run_dir

        aux = auxiliaries if auxiliaries != "none" else ""
        stage_path = lake_run_dir(
            lake_root=os.environ.get("KD_GAT_LAKE_ROOT", "experimentruns"),
            dataset=dataset, model_type=model, scale=scale,
            stage=stage, aux=aux, seed=seed, production=True,
        )
        ckpt = stage_path / ".pl_auto_save.ckpt"
        return str(ckpt) if ckpt.exists() else None
