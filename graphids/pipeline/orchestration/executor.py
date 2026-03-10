"""Scheduler-agnostic job executor interface with SLURM and Flux backends.

The executor handles submission, status polling, and cancellation.
It maps JobSpec resources to scheduler-specific directives.
No domain logic lives here — it only knows about jobs and schedulers.
"""

from __future__ import annotations

import logging
import os
import subprocess
from abc import ABC, abstractmethod
from typing import Any

from .job import JobSpec, JobState

log = logging.getLogger(__name__)


class ExecutorError(Exception):
    """Raised when job submission or status query fails."""


class JobExecutor(ABC):
    """Abstract job executor. Subclass for each scheduler."""

    @abstractmethod
    def submit(
        self,
        job: JobSpec,
        dependency_ids: list[str] | None = None,
        extra_flags: list[str] | None = None,
    ) -> str:
        """Submit a job. Returns the native scheduler job ID as a string.

        Parameters
        ----------
        job : JobSpec
            The job to submit.
        dependency_ids : list[str] | None
            Native IDs of jobs that must complete successfully first.
        extra_flags : list[str] | None
            Additional scheduler-specific flags.
        """

    @abstractmethod
    def poll(self, native_id: str) -> tuple[JobState, dict[str, Any]]:
        """Poll job status. Returns (state, metadata).

        Metadata may include: failure_reason, hostname, resources_used, etc.
        """

    @abstractmethod
    def cancel(self, native_id: str) -> None:
        """Cancel a running or queued job."""

    @classmethod
    def create(cls, backend: str | None = None) -> JobExecutor:
        """Factory: create executor from backend name or environment.

        Backend is auto-detected from ORCHESTRATOR_BACKEND env var if not specified.
        Falls back to SLURM.
        """
        backend = backend or os.getenv("ORCHESTRATOR_BACKEND", "slurm")
        if backend == "slurm":
            return SlurmExecutor()
        if backend == "flux":
            return FluxExecutor()
        if backend == "local":
            return LocalExecutor()
        if backend == "dry_run":
            return DryRunExecutor()
        raise ValueError(f"Unknown executor backend: {backend}")


class SlurmExecutor(JobExecutor):
    """SLURM executor using sbatch/sacct/scancel."""

    def __init__(
        self,
        account: str | None = None,
        partition_gpu: str = "gpu",
        partition_cpu: str = "cpu",
        gpu_type: str | None = None,
        workdir: str | None = None,
        log_dir: str = "slurm_logs",
        preamble_script: str | None = None,
    ):
        self.account = account or os.getenv("KD_GAT_SLURM_ACCOUNT", "PAS1266")
        self.partition_gpu = partition_gpu
        self.partition_cpu = partition_cpu
        self.gpu_type = gpu_type or os.getenv("KD_GAT_GPU_TYPE", "v100")
        self.workdir = workdir or os.getcwd()
        self.log_dir = log_dir
        self.preamble_script = preamble_script or "scripts/slurm/_preamble.sh"

    def submit(
        self,
        job: JobSpec,
        dependency_ids: list[str] | None = None,
        extra_flags: list[str] | None = None,
    ) -> str:
        res = job.resources
        partition = self.partition_gpu if res.gpus > 0 else self.partition_cpu

        # Build the command to run
        if job.executable:
            py_cmd = " ".join([job.executable, *job.arguments])
        else:
            py_cmd = " ".join(job.arguments)

        # Wrap with preamble for SIGUSR1 trap support
        wrap_cmd = (
            f"bash -c 'source {self.preamble_script} && {py_cmd} & "
            f"_KD_CHILD_PID=$!; wait $_KD_CHILD_PID'"
        )

        safe_name = job.name.replace("/", "_")[:40]
        cmd = [
            "sbatch",
            "--parsable",
            f"--account={self.account}",
            f"--chdir={self.workdir}",
            f"--partition={partition}",
            f"--mem={res.memory_gb}G",
            f"--time={res.walltime_str}",
            f"--cpus-per-task={res.cpus}",
            f"--job-name=kd-{safe_name}",
            f"--output={self.log_dir}/%j-{safe_name}.out",
            f"--error={self.log_dir}/%j-{safe_name}.err",
            "--signal=B:USR1@180",
        ]

        if res.gpus > 0:
            cmd.append(f"--gres=gpu:{self.gpu_type}:{res.gpus}")

        if dependency_ids:
            dep_str = ":".join(dependency_ids)
            cmd.append(f"--dependency=afterok:{dep_str}")

        if extra_flags:
            cmd.extend(extra_flags)

        # Environment variables
        if job.environment:
            exports = ",".join(f"{k}={v}" for k, v in job.environment.items())
            cmd.append(f"--export=ALL,{exports}")

        cmd.extend(["--wrap", wrap_cmd])

        log.info("sbatch: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.workdir)
        if result.returncode != 0:
            raise ExecutorError(f"sbatch failed: {result.stderr.strip()}")

        native_id = result.stdout.strip().split(";")[0]
        log.info("Submitted %s → SLURM job %s", job.name, native_id)
        return native_id

    def poll(self, native_id: str) -> tuple[JobState, dict[str, Any]]:
        result = subprocess.run(
            [
                "sacct",
                "-j",
                native_id,
                "--format=State,ExitCode,NodeList,MaxRSS,Elapsed",
                "--noheader",
                "-P",
                "--parsable2",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            log.warning("sacct failed for job %s: %s", native_id, result.stderr.strip())
            return JobState.QUEUED, {}

        # Parse first non-empty line (skip batch/extern steps)
        meta: dict[str, Any] = {}
        for line in result.stdout.strip().splitlines():
            parts = line.strip().split("|")
            if len(parts) >= 5 and parts[0].strip():
                slurm_state = parts[0].strip().split(" ")[0]  # "CANCELLED by UID" -> "CANCELLED"
                meta["exit_code"] = parts[1].strip()
                meta["hostname"] = parts[2].strip()
                meta["max_rss"] = parts[3].strip()
                meta["elapsed"] = parts[4].strip()
                meta["failure_reason"] = slurm_state

                state_map = {
                    "PENDING": JobState.QUEUED,
                    "RUNNING": JobState.RUNNING,
                    "COMPLETED": JobState.COMPLETED,
                    "FAILED": JobState.FAILED,
                    "OUT_OF_MEMORY": JobState.FAILED,
                    "TIMEOUT": JobState.FAILED,
                    "NODE_FAIL": JobState.FAILED,
                    "CANCELLED": JobState.CANCELED,
                    "PREEMPTED": JobState.FAILED,
                }
                return state_map.get(slurm_state, JobState.QUEUED), meta

        return JobState.QUEUED, {}

    def cancel(self, native_id: str) -> None:
        subprocess.run(["scancel", native_id], capture_output=True, text=True)
        log.info("Cancelled SLURM job %s", native_id)


class FluxExecutor(JobExecutor):
    """Flux executor using flux CLI commands.

    Placeholder for LLNL internship — implements the same interface.
    """

    def submit(
        self,
        job: JobSpec,
        dependency_ids: list[str] | None = None,
        extra_flags: list[str] | None = None,
    ) -> str:
        res = job.resources
        if job.executable:
            run_cmd = [job.executable, *job.arguments]
        else:
            run_cmd = list(job.arguments)

        cmd = [
            "flux",
            "batch",
            f"--nslots={res.nodes}",
            f"--cores-per-slot={res.cpus}",
            f"--time={res.walltime_str}",
            f"--job-name={job.name.replace('/', '_')[:40]}",
        ]
        if res.gpus > 0:
            cmd.append(f"--gpus-per-slot={res.gpus}")

        if extra_flags:
            cmd.extend(extra_flags)

        cmd.extend(["--wrap", " ".join(run_cmd)])

        log.info("flux batch: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise ExecutorError(f"flux batch failed: {result.stderr.strip()}")

        native_id = result.stdout.strip()
        log.info("Submitted %s → Flux job %s", job.name, native_id)
        return native_id

    def poll(self, native_id: str) -> tuple[JobState, dict[str, Any]]:
        result = subprocess.run(
            ["flux", "jobs", "--no-header", "-o", "{status_abbrev}", native_id],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return JobState.QUEUED, {}

        status = result.stdout.strip()
        state_map = {
            "PD": JobState.QUEUED,
            "R": JobState.RUNNING,
            "CD": JobState.COMPLETED,
            "F": JobState.FAILED,
            "CA": JobState.CANCELED,
        }
        return state_map.get(status, JobState.QUEUED), {}

    def cancel(self, native_id: str) -> None:
        subprocess.run(["flux", "cancel", native_id], capture_output=True, text=True)
        log.info("Cancelled Flux job %s", native_id)


class LocalExecutor(JobExecutor):
    """Executor that runs jobs as local subprocesses. For development without a scheduler."""

    _counter = 0

    def __init__(self, workdir: str | None = None):
        self.workdir = workdir or os.getcwd()
        self._processes: dict[str, subprocess.Popen] = {}

    def submit(
        self,
        job: JobSpec,
        dependency_ids: list[str] | None = None,
        extra_flags: list[str] | None = None,
    ) -> str:
        LocalExecutor._counter += 1
        native_id = f"LOCAL_{LocalExecutor._counter}"

        cmd = [job.executable, *job.arguments] if job.executable else list(job.arguments)
        env = {**os.environ, **(job.environment or {})}

        log.info("[LOCAL] Running: %s", " ".join(cmd))
        proc = subprocess.Popen(
            cmd,
            cwd=self.workdir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._processes[native_id] = proc
        log.info("Submitted %s → local process %s (pid %d)", job.name, native_id, proc.pid)
        return native_id

    def poll(self, native_id: str) -> tuple[JobState, dict[str, Any]]:
        proc = self._processes.get(native_id)
        if proc is None:
            return JobState.COMPLETED, {}

        returncode = proc.poll()
        if returncode is None:
            return JobState.RUNNING, {"pid": proc.pid}

        meta: dict[str, Any] = {"pid": proc.pid, "returncode": returncode}
        if returncode != 0:
            stderr = proc.stderr.read().decode() if proc.stderr else ""
            meta["failure_reason"] = stderr[-500:] if stderr else f"exit code {returncode}"
            return JobState.FAILED, meta

        return JobState.COMPLETED, meta

    def cancel(self, native_id: str) -> None:
        proc = self._processes.get(native_id)
        if proc and proc.poll() is None:
            proc.terminate()
            log.info("[LOCAL] Terminated process %s (pid %d)", native_id, proc.pid)


class DryRunExecutor(JobExecutor):
    """Executor that prints commands without submitting. For testing."""

    _counter = 0

    def submit(
        self,
        job: JobSpec,
        dependency_ids: list[str] | None = None,
        extra_flags: list[str] | None = None,
    ) -> str:
        DryRunExecutor._counter += 1
        native_id = f"DRY_{DryRunExecutor._counter}"
        deps_str = f" (after {dependency_ids})" if dependency_ids else ""
        log.info(
            "[DRY RUN] Would submit: %s | resources: %dGPU %dCPU %dGB %s%s",
            job.name,
            job.resources.gpus,
            job.resources.cpus,
            job.resources.memory_gb,
            job.resources.walltime_str,
            deps_str,
        )
        return native_id

    def poll(self, native_id: str) -> tuple[JobState, dict[str, Any]]:
        return JobState.COMPLETED, {}

    def cancel(self, native_id: str) -> None:
        log.info("[DRY RUN] Would cancel: %s", native_id)
