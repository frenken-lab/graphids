"""SLURM configuration helpers for Ray on OSC."""

from __future__ import annotations

import os

from graphids.config.constants import SLURM_ACCOUNT, SLURM_GPU_TYPE, SLURM_PARTITION

SCRATCH_ROOT = os.getenv("KD_GAT_SCRATCH", "/fs/scratch/PAS1266")
RAY_SCRATCH = f"{SCRATCH_ROOT}/.ray"


def ensure_dirs() -> None:
    """Create scratch directories for Ray temp files."""
    os.makedirs(RAY_SCRATCH, exist_ok=True)


def ray_init_kwargs(num_gpus: int = 1) -> dict:
    """Kwargs for ray.init() on a SLURM node.

    On a compute node (SLURM_JOB_ID set), initializes Ray with local resources.
    On a login node, uses Ray local mode for testing.
    """
    ensure_dirs()
    on_compute = bool(os.environ.get("SLURM_JOB_ID"))

    kwargs: dict = {
        "_temp_dir": RAY_SCRATCH,
        "logging_level": "info",
    }

    if on_compute:
        # Let Ray auto-detect resources from SLURM allocation
        kwargs["num_gpus"] = num_gpus
    else:
        # Login node: local mode for quick testing
        kwargs["num_gpus"] = 0

    return kwargs


def sbatch_header(
    job_name: str = "kd-gat-ray",
    nodes: int = 1,
    gpus_per_node: int = 1,
    cpus_per_task: int = 4,
    mem: str = "50G",
    time: str = "08:00:00",
) -> str:
    """Generate SBATCH header lines for a Ray job."""
    lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --account={SLURM_ACCOUNT}",
        f"#SBATCH --partition={SLURM_PARTITION}",
        f"#SBATCH --gres=gpu:{SLURM_GPU_TYPE}:{gpus_per_node}",
        f"#SBATCH --nodes={nodes}",
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={cpus_per_task}",
        f"#SBATCH --mem={mem}",
        f"#SBATCH --time={time}",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --output=slurm_logs/{job_name}_%j.out",
        f"#SBATCH --error=slurm_logs/{job_name}_%j.err",
    ]
    return "\n".join(lines)
