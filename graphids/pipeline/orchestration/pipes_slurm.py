"""SLURM sbatch/sacct wrapper for Dagster Pipes orchestration.

Generates sbatch scripts, submits jobs, polls via sacct, and validates
artifacts. Resource profiles and failure reactions loaded from
``graphids/config/resources.yaml``.

Reuses:
- ``ResourceSpec`` from ``job.py`` (single resource model)
- ``build_cli_cmd()`` from ``subprocess_utils.py``
- ``stage_dir()`` / ``checkpoint_path()`` from ``paths.py``
- ``SLURM_ACCOUNT`` / ``SLURM_GPU_TYPE`` from ``constants.py``
"""

from __future__ import annotations

import logging
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


_RESOURCES_YAML = Path(__file__).resolve().parents[2] / "config" / "resources.yaml"


# ---------------------------------------------------------------------------
# YAML loading (same pattern as resolver.py)
# ---------------------------------------------------------------------------


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
    """Look up resource profile for a (model, scale, stage) tuple.

    Raises KeyError with a helpful message if not found.
    """
    key = (model, scale, stage)
    if key not in RESOURCE_PROFILES:
        available = sorted(RESOURCE_PROFILES.keys())
        raise KeyError(
            f"No resource profile for {key}. "
            f"Add an entry to config/resources.yaml. Available: {available}"
        )
    return RESOURCE_PROFILES[key]


# ---------------------------------------------------------------------------
# Resource scaling for adaptive retry
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
        scaled_secs = int(total_secs * reaction["scale_time"])
        updates["walltime"] = timedelta(seconds=scaled_secs)

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
    pipes_context_path: str | None = None,
    pipes_messages_path: str | None = None,
    project_root: str | None = None,
) -> str:
    """Generate a complete sbatch script for one pipeline stage.

    Sources _preamble.sh for env setup and _epilog.sh for cleanup.
    Uses build_cli_cmd() for the stage command.
    """
    root = project_root or str(PROJECT_ROOT)

    # Build the CLI command string
    cmd_list = build_cli_cmd(
        stage=stage,
        model=model,
        scale=scale,
        dataset=dataset,
        seed=seed,
        auxiliaries=auxiliaries,
        ckpt_path=ckpt_path,
    )
    cli_command = " ".join(cmd_list)

    # Build SBATCH directives
    lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --account={SLURM_ACCOUNT}",
        f"#SBATCH --partition={resources.partition}",
    ]

    if resources.gpus > 0:
        lines.append(f"#SBATCH --gres=gpu:{SLURM_GPU_TYPE}:{resources.gpus}")

    lines.extend(
        [
            "#SBATCH --nodes=1 --ntasks=1",
            f"#SBATCH --cpus-per-task={resources.cpus}",
            f"#SBATCH --mem={resources.mem_slurm}",
            f"#SBATCH --time={resources.walltime_slurm}",
            f"#SBATCH --job-name=kd-gat-{stage}-{model}-{scale}",
            "#SBATCH --output=slurm_logs/dagster_%j.out",
            "#SBATCH --error=slurm_logs/dagster_%j.err",
            "#SBATCH --signal=B:USR1@180",
        ]
    )

    if resources.exclude_nodes:
        lines.append(f"#SBATCH --exclude={resources.exclude_nodes}")

    if dependency_job_id:
        lines.append(f"#SBATCH --dependency=afterok:{dependency_job_id}")

    # Script body
    lines.extend(
        [
            "",
            f'cd "{root}"',
            "source scripts/slurm/_preamble.sh",
            "",
        ]
    )

    # Dagster-orchestrated jobs write to production/ in the data lake
    lines.append("export KD_GAT_PRODUCTION=1")
    lines.append("")

    # Dagster Pipes env vars (NFS temp file transport)
    if pipes_context_path:
        lines.append(f'export DAGSTER_PIPES_CONTEXT="{pipes_context_path}"')
    if pipes_messages_path:
        lines.append(f'export DAGSTER_PIPES_MESSAGES="{pipes_messages_path}"')
    if pipes_context_path or pipes_messages_path:
        lines.append("")

    # For CPU-only jobs, skip CUDA config
    if resources.gpus == 0:
        lines.append("export SKIP_CUDA_CONF=1")
        lines.append("")

    lines.extend(
        [
            "# Stage command (generated by build_cli_cmd)",
            f"{cli_command} &",
            "_KD_CHILD_PID=$!",
            "wait $_KD_CHILD_PID",
            "EXIT_CODE=$?",
            "",
            "source scripts/slurm/_epilog.sh",
            "exit $EXIT_CODE",
        ]
    )

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# SLURM job submission and polling
# ---------------------------------------------------------------------------


class PipesSlurmClient:
    """Submit sbatch jobs and poll via sacct.

    In dry_run mode, logs the generated script without submitting.
    """

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
    ) -> tuple[str, Path]:
        """Generate sbatch script, write to disk, return (script_content, path)."""
        script = generate_sbatch_script(
            stage=stage,
            model=model,
            scale=scale,
            dataset=dataset,
            resources=resources,
            seed=seed,
            auxiliaries=auxiliaries,
            ckpt_path=ckpt_path,
            dependency_job_id=dependency_job_id,
            project_root=self.project_root,
        )

        aux_tag = f"_{auxiliaries}" if auxiliaries != "none" else ""
        script_name = f"dagster_{model}_{scale}_{stage}{aux_tag}.sbatch"
        script_path = self._scripts_dir / script_name
        self._scripts_dir.mkdir(parents=True, exist_ok=True)
        script_path.write_text(script)
        log.info("Wrote sbatch script: %s", script_path)
        return script, script_path

    def run(
        self,
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
    ) -> dict:
        """Submit a SLURM job and poll until completion.

        Returns a metadata dict with job_id, state, elapsed, etc.
        Raises SlurmJobFailed on terminal failure states.
        """
        script, script_path = self._write_script(
            stage,
            model,
            scale,
            dataset,
            resources,
            seed=seed,
            auxiliaries=auxiliaries,
            ckpt_path=ckpt_path,
            dependency_job_id=dependency_job_id,
        )

        if self.dry_run:
            log.info("[DRY RUN] Would submit:\n%s", script)
            return {
                "job_id": "dry-run",
                "state": "DRY_RUN",
                "script_path": str(script_path),
            }

        # Submit
        job_id = self._submit(script_path)
        log.info("Submitted SLURM job %s for %s/%s/%s", job_id, model, scale, stage)

        # Poll
        t0 = time.monotonic()
        state, reason, node = self._poll_until_done(job_id)
        elapsed = time.monotonic() - t0

        metadata = {
            "job_id": job_id,
            "state": state,
            "reason": reason,
            "node": node,
            "elapsed_seconds": round(elapsed, 1),
            "script_path": str(script_path),
        }

        if state == "COMPLETED":
            # Validate artifacts
            artifacts_ok = self._validate_artifacts(dataset, model, scale, stage, auxiliaries)
            metadata["artifacts_valid"] = artifacts_ok
            if not artifacts_ok:
                log.warning(
                    "Job %s COMPLETED but artifacts missing for %s/%s/%s",
                    job_id,
                    model,
                    scale,
                    stage,
                )
            return metadata

        # Terminal failure
        ckpt = self._find_checkpoint(dataset, model, scale, stage, auxiliaries)
        raise SlurmJobFailed(
            reason=state,
            node=node,
            ckpt_path=ckpt,
            metadata=metadata,
        )

    def _submit(self, script_path: Path) -> str:
        """Submit sbatch script, return job ID."""
        result = subprocess.run(
            ["sbatch", "--parsable", str(script_path)],
            capture_output=True,
            text=True,
            check=True,
            cwd=self.project_root,
        )
        job_id = result.stdout.strip().split(";")[0]  # parsable may include cluster
        return job_id

    def _poll_until_done(self, job_id: str) -> tuple[str, str, str]:
        """Poll sacct until job reaches terminal state.

        Returns (state, reason, node_name).
        """
        terminal_states = {
            "COMPLETED",
            "FAILED",
            "CANCELLED",
            "TIMEOUT",
            "OUT_OF_MEMORY",
            "NODE_FAIL",
            "PREEMPTED",
        }

        while True:
            state, reason, node = self._sacct_query(job_id)

            if state in terminal_states:
                log.info("Job %s reached state: %s (reason: %s)", job_id, state, reason)
                return state, reason, node

            log.debug(
                "Job %s state: %s, polling in %ds...",
                job_id,
                state,
                self.poll_interval,
            )
            time.sleep(self.poll_interval)

    def _sacct_query(self, job_id: str) -> tuple[str, str, str]:
        """Query sacct for job state.

        Returns (state, reason, node_name).
        """
        result = subprocess.run(
            [
                "sacct",
                "-j",
                job_id,
                "-X",  # no sub-steps
                "--parsable2",
                "--noheader",
                "-o",
                "State,Reason,NodeList",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return "PENDING", "", ""

        # Take first line (the main job, not steps)
        parts = result.stdout.strip().split("\n")[0].split("|")
        state = parts[0].split()[0] if parts else "UNKNOWN"  # strip trailing modifiers
        reason = parts[1] if len(parts) > 1 else ""
        node = parts[2] if len(parts) > 2 else ""
        return state, reason, node

    def _validate_artifacts(
        self,
        dataset: str,
        model: str,
        scale: str,
        stage: str,
        auxiliaries: str,
        seed: int = 42,
    ) -> bool:
        """Check that expected artifacts exist after stage completion.

        Checks seed subdirectory first, then falls back to flat layout.
        """
        from pydantic import ValidationError

        from graphids.config import (
            EXPERIMENT_ROOT,
            EvaluationArtifact,
            TrainingArtifact,
            run_id_str,
        )

        if stage == "preprocess":
            return True

        aux = auxiliaries if auxiliaries != "none" else ""
        rid = run_id_str(dataset, model, scale, stage, aux)
        base_path = Path(EXPERIMENT_ROOT) / rid

        # Check seed subdirectory first, then flat
        for stage_path in [base_path / f"seed_{seed}", base_path]:
            if not stage_path.exists():
                continue
            contract = EvaluationArtifact if stage == "evaluation" else TrainingArtifact
            try:
                contract.from_stage_dir(stage_path)
                return True
            except (ValidationError, ValueError):
                continue

        log.warning("Artifact validation failed for %s (seed=%d)", base_path, seed)
        return False

    def _find_checkpoint(
        self,
        dataset: str,
        model: str,
        scale: str,
        stage: str,
        auxiliaries: str,
        seed: int = 42,
    ) -> str | None:
        """Look for a Lightning auto-save checkpoint after TIMEOUT."""
        from graphids.config import EXPERIMENT_ROOT, run_id_str

        aux = auxiliaries if auxiliaries != "none" else ""
        rid = run_id_str(dataset, model, scale, stage, aux)
        # Check seed subdir first, then flat
        for subdir in [f"seed_{seed}", ""]:
            ckpt = Path(EXPERIMENT_ROOT) / rid / subdir / ".pl_auto_save.ckpt"
            if ckpt.exists():
                return str(ckpt)
        return None

    def submit_no_poll(
        self,
        stage: str,
        model: str,
        scale: str,
        dataset: str,
        resources: ResourceSpec,
        *,
        seed: int | None = None,
        auxiliaries: str = "none",
        dependency_job_id: str | None = None,
    ) -> str:
        """Submit a job without polling. Returns job ID.

        Used for fire-and-forget mode with ``--dependency=afterok`` chains.
        """
        _script, script_path = self._write_script(
            stage,
            model,
            scale,
            dataset,
            resources,
            seed=seed,
            auxiliaries=auxiliaries,
            dependency_job_id=dependency_job_id,
        )

        if self.dry_run:
            log.info("[DRY RUN] Would submit: %s", script_path)
            return "dry-run"

        job_id = self._submit(script_path)
        log.info("Submitted (no-poll) SLURM job %s for %s/%s/%s", job_id, model, scale, stage)
        return job_id
