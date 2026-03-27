"""SLURM job launcher — adapted from test-tube SlurmCluster (Lightning AI, 2019)."""

from __future__ import annotations

import subprocess
from pathlib import Path


class SlurmCluster:
    """Generate and submit sbatch scripts with dependency chaining."""

    def __init__(self, log_path: str | Path):
        self.log_path = Path(log_path)
        self.job_time = "04:00:00"
        self.per_experiment_nb_gpus = 1
        self.per_experiment_nb_nodes = 1
        self.per_experiment_nb_cpus = 1
        self.memory_mb_per_node = 16000
        self.gpu_type: str | None = None
        self.partition = "gpu"
        self.account: str | None = None
        self.signal_seconds = 300
        self.modules: list[str] = []
        self.commands: list[str] = []
        self.extra_sbatch: list[tuple[str, str]] = []

    def add_command(self, cmd: str): self.commands.append(cmd)
    def load_modules(self, modules: list[str]): self.modules = modules
    def add_slurm_cmd(self, cmd: str, value: str): self.extra_sbatch.append((cmd, value))

    def schedule(self, run_cmd: str, job_name: str, *, depend_on: int | list[int] | None = None) -> int:
        """Build sbatch script, submit, return SLURM job ID."""
        for d in ("scripts", "out", "err"):
            (self.log_path / d).mkdir(parents=True, exist_ok=True)
        path = self.log_path / "scripts" / f"{job_name}.sh"
        path.write_text(self._build(run_cmd, job_name))
        cmd = ["sbatch"]
        if depend_on is not None:
            ids = [depend_on] if isinstance(depend_on, int) else depend_on
            cmd.append(f"--dependency=afterany:{':'.join(str(d) for d in ids)}")
        cmd.append(str(path))
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        job_id = int(result.stdout.strip().split()[-1])
        print(f"{job_name} -> {job_id}")
        return job_id

    def optimize_parallel(self, run_cmds: list[str], job_name: str, *, depend_on=None) -> list[int]:
        """Submit multiple jobs in parallel (grid search)."""
        return [self.schedule(c, f"{job_name}_{i}", depend_on=depend_on) for i, c in enumerate(run_cmds)]

    def _build(self, run_cmd: str, job_name: str) -> str:
        S = "#SBATCH"  # noqa: N806
        o, e = self.log_path / "out", self.log_path / "err"
        h = [
            "#!/bin/bash", f"{S} --job-name={job_name}",
            f"{S} --output={o}/{job_name}_%j.out", f"{S} --error={e}/{job_name}_%j.err",
            f"{S} --time={self.job_time}", f"{S} --nodes={self.per_experiment_nb_nodes}",
            f"{S} --mem={self.memory_mb_per_node}", f"{S} --cpus-per-task={self.per_experiment_nb_cpus}",
            f"{S} --signal=USR1@{self.signal_seconds}",
        ]
        if self.partition:
            h.append(f"{S} --partition={self.partition}")
        if self.account:
            h.append(f"{S} --account={self.account}")
        if self.per_experiment_nb_gpus > 0:
            g = f"{self.gpu_type}:{self.per_experiment_nb_gpus}" if self.gpu_type else self.per_experiment_nb_gpus
            h.append(f"{S} --gres=gpu:{g}")
        h.extend(f"{S} --{c}={v}" for c, v in self.extra_sbatch)
        h.append("")
        h.extend(f"module load {m}" for m in self.modules)
        h.extend(self.commands)
        h.extend(["", f"srun {run_cmd}"])
        return "\n".join(h) + "\n"
