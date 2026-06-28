"""SLURM submission for experiment YAML configs."""

from __future__ import annotations

import math
import os
import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from graphids.exp.config import ExperimentConfig
from graphids.paths import PROJECT_ROOT


@dataclass(frozen=True)
class SlurmSubmitResult:
    command: tuple[str, ...]
    script_path: Path
    script: str
    job_id: str | None = None
    stdout: str = ""


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return slug[:120] or "graphids"


def _log_dir() -> Path:
    if override := os.environ.get("GRAPHIDS_SLURM_LOG_DIR"):
        return Path(override)
    dotenv = _dotenv_values()
    if override := dotenv.get("GRAPHIDS_SLURM_LOG_DIR"):
        return Path(override)
    if lake_root := os.environ.get("GRAPHIDS_LAKE_ROOT"):
        return Path(lake_root) / "slurm"
    if lake_root := dotenv.get("GRAPHIDS_LAKE_ROOT"):
        return Path(lake_root) / "slurm"
    return PROJECT_ROOT / "slurm"


def _script_dir() -> Path:
    return _log_dir() / "scripts"


def _dotenv_values() -> dict[str, str]:
    path = PROJECT_ROOT / ".env"
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


def _gpus(cfg: ExperimentConfig) -> int:
    requested = int(math.ceil(float(cfg.resources.gpus_per_worker)))
    if cfg.resources.accelerator == "gpu":
        return max(1, requested)
    return max(0, requested)


def _sbatch_directives(
    cfg: ExperimentConfig,
    *,
    job_name: str,
    cluster: str | None,
    partition: str | None,
    time_limit: str | None,
    gres: str | None,
) -> list[str]:
    resources = cfg.resources
    gpus = _gpus(cfg)
    resolved_partition = partition or resources.partition or ("gpu" if gpus else "cpu")
    resolved_time = time_limit or resources.time_limit or "01:00:00"
    resolved_gres = gres or resources.gres or (f"gpu:{gpus}" if gpus else None)
    dotenv = _dotenv_values()
    resolved_account = (
        resources.account
        or os.environ.get("GRAPHIDS_SLURM_ACCOUNT")
        or dotenv.get("GRAPHIDS_SLURM_ACCOUNT")
    )

    lines = [
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --partition={resolved_partition}",
        f"#SBATCH --time={resolved_time}",
        f"#SBATCH --cpus-per-task={max(1, int(resources.cpus_per_worker))}",
    ]
    if cluster or resources.cluster:
        lines.append(f"#SBATCH --clusters={cluster or resources.cluster}")
    if resolved_gres:
        lines.append(f"#SBATCH --gres={resolved_gres}")
    if resources.memory_gb is not None:
        lines.append(f"#SBATCH --mem={int(resources.memory_gb)}G")
    if resolved_account:
        lines.append(f"#SBATCH --account={resolved_account.lower()}")
    return lines


def build_slurm_script(
    cfg: ExperimentConfig,
    yaml_path: str | Path,
    *,
    cluster: str | None = None,
    partition: str | None = None,
    time_limit: str | None = None,
    gres: str | None = None,
    nodes: int = 1,
    ray_port: int = 6379,
) -> tuple[Path, str]:
    """Build a SLURM script that starts Ray inside one allocation."""
    yaml_abs = Path(yaml_path).resolve()
    if not yaml_abs.is_file():
        raise FileNotFoundError(f"experiment YAML not found: {yaml_abs}")

    job_name = _slug(f"ray-{cfg.experiment_name}")
    log_dir = _log_dir()
    script_dir = _script_dir()
    script_path = script_dir / f"{job_name}.sbatch"
    stdout_path = log_dir / f"{job_name}_%j.out"
    stderr_path = log_dir / f"{job_name}_%j.err"
    directives = _sbatch_directives(
        cfg,
        job_name=job_name,
        cluster=cluster,
        partition=partition,
        time_limit=time_limit,
        gres=gres,
    )
    directives[0] = f"#SBATCH --nodes={max(1, int(nodes))}"
    directives[1] = "#SBATCH --ntasks-per-node=1"
    directives.extend(
        [
            f"#SBATCH --output={stdout_path}",
            f"#SBATCH --error={stderr_path}",
        ]
    )
    gpus = _gpus(cfg)
    cpus = max(1, int(cfg.resources.cpus_per_worker))
    script = "\n".join(
        [
            "#!/usr/bin/env bash",
            *directives,
            "",
            "set -euo pipefail",
            f"cd {PROJECT_ROOT}",
            "source scripts/slurm/_preamble.sh",
            "",
            f"RAY_PORT=${{GRAPHIDS_RAY_PORT:-{int(ray_port)}}}",
            f"RAY_LOG_DIR={log_dir}/ray_${{SLURM_JOB_ID}}",
            'mkdir -p "${RAY_LOG_DIR}"',
            'mapfile -t RAY_NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")',
            'HEAD_NODE="${RAY_NODES[0]}"',
            'HEAD_IP=$(srun --nodes=1 --ntasks=1 -w "${HEAD_NODE}" hostname --ip-address | awk \'{print $1}\')',
            'RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"',
            "cleanup_ray() {",
            '  for NODE in "${RAY_NODES[@]}"; do',
            '    srun --nodes=1 --ntasks=1 -w "${NODE}" ray stop --force >/dev/null 2>&1 || true &',
            "  done",
            "  wait || true",
            "}",
            "trap cleanup_ray EXIT",
            "",
            "cleanup_ray",
            'srun --nodes=1 --ntasks=1 -w "${HEAD_NODE}" \\',
            '  ray start --head --node-ip-address="${HEAD_IP}" --port="${RAY_PORT}" \\',
            f'    --num-cpus={cpus} --num-gpus={gpus} --temp-dir="${{RAY_LOG_DIR}}/head" \\',
            '    --block >"${RAY_LOG_DIR}/head.log" 2>&1 &',
            "sleep 10",
            "",
            'for NODE in "${RAY_NODES[@]:1}"; do',
            '  srun --nodes=1 --ntasks=1 -w "${NODE}" \\',
            '    ray start --address="${RAY_ADDRESS}" \\',
            f'      --num-cpus={cpus} --num-gpus={gpus} --temp-dir="${{RAY_LOG_DIR}}/${{NODE}}" \\',
            '      --block >"${RAY_LOG_DIR}/${NODE}.log" 2>&1 &',
            "done",
            "sleep 10",
            "",
            f'python -m graphids exp launch {yaml_abs} --address "${{RAY_ADDRESS}}"',
            "source scripts/slurm/_epilog.sh",
            "",
        ]
    )
    return script_path, script


def submit_experiment(
    cfg: ExperimentConfig,
    yaml_path: str | Path,
    *,
    cluster: str | None = None,
    partition: str | None = None,
    time_limit: str | None = None,
    gres: str | None = None,
    nodes: int = 1,
    dry_run: bool = False,
    sbatch: str = "sbatch",
) -> SlurmSubmitResult:
    cfg.build_run(name=cfg.experiment_name, stage=cfg.stage, config=cfg.config)
    script_path, script = build_slurm_script(
        cfg,
        yaml_path,
        cluster=cluster,
        partition=partition,
        time_limit=time_limit,
        gres=gres,
        nodes=nodes,
    )
    command: Sequence[str] = (sbatch, str(script_path))
    if dry_run:
        return SlurmSubmitResult(command=tuple(command), script_path=script_path, script=script)

    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(script)
    completed = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    match = re.search(r"Submitted batch job\s+(\S+)", completed.stdout)
    return SlurmSubmitResult(
        command=tuple(command),
        script_path=script_path,
        script=script,
        job_id=match.group(1) if match else None,
        stdout=completed.stdout,
    )
