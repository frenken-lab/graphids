"""SLURM submission for experiment YAML configs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
import shlex
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from graphids.exp.config import ExperimentConfig
from graphids.paths import PROJECT_ROOT, env_value, slurm_log_dir


@dataclass(frozen=True)
class SlurmSubmitResult:
    command: tuple[str, ...]
    script_path: Path
    script: str
    job_id: str | None = None
    stdout: str = ""


@dataclass(frozen=True)
class SlurmJob:
    script_path: Path
    script: str
    config_path: Path
    config_text: str


@dataclass(frozen=True)
class SourceSnapshot:
    root: Path
    git_sha: str
    hashes: dict[str, str]
    diff: str


def dependency_versions() -> dict[str, str]:
    return {
        dist.metadata["Name"]: dist.version
        for dist in importlib.metadata.distributions()
    }


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def snapshot_source(
    destination: Path, *, source_root: Path = PROJECT_ROOT
) -> SourceSnapshot:
    """Copy runtime code, tests, docs and configs, including uncommitted files."""
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"source snapshot already exists: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    selected = {"graphids", "scripts", "tests", "configs", "docs"}
    selected_files = {
        "pyproject.toml",
        "uv.lock",
        "MODEL_IMPROVEMENT_PLAN.md",
        "mkdocs.yml",
        "requirements-docs.txt",
    }
    tracked = (
        subprocess.check_output(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            cwd=source_root,
        )
        .decode()
        .split("\0")
    )
    hashes = {}
    for name in sorted(set(tracked) - {""}):
        relative = Path(name)
        if relative.parts[0] not in selected and name not in selected_files:
            continue
        source = source_root / relative
        if not source.is_file():  # Preserve deletions in the working tree.
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[name] = file_hash(target)
    (destination / ".venv").symlink_to(source_root / ".venv", target_is_directory=True)
    git_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source_root, text=True
    ).strip()
    diff = subprocess.check_output(
        ["git", "diff", "HEAD", "--binary"], cwd=source_root, text=True
    )
    provenance = destination / "source-provenance.json"
    provenance.write_text(
        json.dumps(
            dict(
                git_head=git_sha,
                files=sorted(hashes),
                working_tree_diff=diff,
            ),
            indent=2,
        )
        + "\n"
    )
    hashes[provenance.name] = file_hash(provenance)
    return SourceSnapshot(destination, git_sha, hashes, diff)


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return slug[:120] or "graphids"


def _submission_yaml(cfg: ExperimentConfig, *, directory: Path) -> tuple[Path, str]:
    expanded = yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False)
    digest = hashlib.sha256(expanded.encode()).hexdigest()[:16]
    path = directory / f"{_slug(cfg.experiment_name)}-{digest}.yml"
    return path, expanded


def _cpus(cfg: ExperimentConfig) -> int:
    return max(1, int(cfg.resources.cpus_per_worker))


def _gpus(cfg: ExperimentConfig) -> int:
    return int(cfg.resources.accelerator == "gpu" or cfg.resources.gpus_per_worker > 0)


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
    resolved_account = resources.account or env_value("GRAPHIDS_SLURM_ACCOUNT")

    lines = [
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --partition={resolved_partition}",
        f"#SBATCH --time={resolved_time}",
        f"#SBATCH --cpus-per-task={_cpus(cfg)}",
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


def build_experiment_job(
    cfg: ExperimentConfig,
    *,
    cluster: str | None = None,
    partition: str | None = None,
    time_limit: str | None = None,
    gres: str | None = None,
    source_root: str | Path | None = None,
    log_dir: str | Path | None = None,
    preflight: Sequence[Sequence[str]] = (),
    modules: Sequence[str] = (),
) -> SlurmJob:
    """Validate and render a complete single-node job without side effects."""
    cfg.build_run(name=cfg.experiment_name, stage=cfg.stage)

    job_name = _slug(f"ray-{cfg.experiment_name}")
    log_dir = Path(log_dir) if log_dir else slurm_log_dir()
    script_dir = log_dir / "scripts"
    yaml_abs, expanded = _submission_yaml(cfg, directory=script_dir)
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
    directives.extend(
        [
            f"#SBATCH --output={stdout_path}",
            f"#SBATCH --error={stderr_path}",
        ]
    )
    gpus = _gpus(cfg)
    cpus = _cpus(cfg)
    script = "\n".join(
        [
            "#!/usr/bin/env bash",
            *directives,
            "",
            "set -euo pipefail",
            f"cd {shlex.quote(str(Path(source_root).resolve() if source_root else PROJECT_ROOT))}",
            "source scripts/slurm/_preamble.sh",
            'export PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1',
            *([shlex.join(["module", "load", *modules])] if modules else []),
            *(shlex.join(command) for command in preflight),
            "",
            f"export SLURM_CPUS_PER_TASK={cpus}",
            "RAY_BASE_PORT=${GRAPHIDS_RAY_BASE_PORT:-$((20000 + (SLURM_JOB_ID % 200) * 200))}",
            "RAY_PORT=${GRAPHIDS_RAY_PORT:-${RAY_BASE_PORT}}",
            'RAY_OBJECT_MANAGER_PORT="$((RAY_BASE_PORT + 1))"',
            'RAY_NODE_MANAGER_PORT="$((RAY_BASE_PORT + 2))"',
            'RAY_DASHBOARD_AGENT_PORT="$((RAY_BASE_PORT + 4))"',
            'RAY_DASHBOARD_AGENT_GRPC_PORT="$((RAY_BASE_PORT + 5))"',
            'RAY_RUNTIME_ENV_AGENT_PORT="$((RAY_BASE_PORT + 6))"',
            'RAY_METRICS_EXPORT_PORT="$((RAY_BASE_PORT + 7))"',
            'RAY_MIN_WORKER_PORT="$((RAY_BASE_PORT + 20))"',
            'RAY_MAX_WORKER_PORT="$((RAY_BASE_PORT + 119))"',
            f"RAY_LOG_DIR={shlex.quote(str(log_dir))}/ray_${{SLURM_JOB_ID}}",
            'RAY_TMP_DIR="${TMPDIR:-/tmp}/graphids-ray-${SLURM_JOB_ID}"',
            'mkdir -p "${RAY_LOG_DIR}" "${RAY_TMP_DIR}"',
            "RAY_START_PID=",
            "HEAD_IP=$(hostname --ip-address | awk '{print $1}')",
            'RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"',
            "export RAY_ADDRESS",
            "cleanup_ray() {",
            '  if [[ -n "${RAY_START_PID}" ]]; then',
            '    kill "${RAY_START_PID}" >/dev/null 2>&1 || true',
            '    wait "${RAY_START_PID}" >/dev/null 2>&1 || true',
            "  fi",
            '  rm -rf "${RAY_TMP_DIR}"',
            "}",
            "trap cleanup_ray EXIT",
            "",
            'mkdir -p "${RAY_LOG_DIR}" "${RAY_TMP_DIR}"',
            "srun --nodes=1 --ntasks=1 \\",
            '  ray start --head --node-ip-address="${HEAD_IP}" --port="${RAY_PORT}" \\',
            "    --include-dashboard=false \\",
            '    --object-manager-port="${RAY_OBJECT_MANAGER_PORT}" \\',
            '    --node-manager-port="${RAY_NODE_MANAGER_PORT}" \\',
            '    --dashboard-agent-listen-port="${RAY_DASHBOARD_AGENT_PORT}" \\',
            '    --dashboard-agent-grpc-port="${RAY_DASHBOARD_AGENT_GRPC_PORT}" \\',
            '    --runtime-env-agent-port="${RAY_RUNTIME_ENV_AGENT_PORT}" \\',
            '    --metrics-export-port="${RAY_METRICS_EXPORT_PORT}" \\',
            '    --min-worker-port="${RAY_MIN_WORKER_PORT}" \\',
            '    --max-worker-port="${RAY_MAX_WORKER_PORT}" \\',
            f'    --num-cpus={cpus} --num-gpus={gpus} --temp-dir="${{RAY_TMP_DIR}}/head" \\',
            '    --block >"${RAY_LOG_DIR}/head.log" 2>&1 &',
            "RAY_START_PID=$!",
            "sleep 10",
            "",
            f'python -m graphids exp launch {shlex.quote(str(yaml_abs))} --address "${{RAY_ADDRESS}}"',
            "source scripts/slurm/_epilog.sh",
            "",
        ]
    )
    return SlurmJob(script_path, script, yaml_abs, expanded)


def write_job(job: SlurmJob) -> tuple[Path, Path]:
    job.script_path.parent.mkdir(parents=True, exist_ok=True)
    job.config_path.write_text(job.config_text)
    job.script_path.write_text(job.script)
    return job.script_path, job.config_path


def submit_script(
    path: str | Path,
    *,
    dependency: str | None = None,
    sbatch: str = "sbatch",
    dry_run: bool = False,
) -> SlurmSubmitResult:
    """Submit any prepared script and retain the exact dependency command."""
    path = Path(path)
    command = [sbatch, "--parsable", "--kill-on-invalid-dep=yes"]
    if dependency:
        command.append(f"--dependency={dependency}")
    command.append(str(path))
    if dry_run:
        return SlurmSubmitResult(tuple(command), path, "")
    script = path.read_text()
    output = subprocess.check_output(command, text=True, timeout=60).strip()
    job_id = output.split(";")[0]
    if not job_id.isdigit():
        raise ValueError(f"unexpected sbatch result: {output!r}")
    return SlurmSubmitResult(tuple(command), path, script, job_id, output)


def submit_experiment(
    cfg: ExperimentConfig,
    *,
    dependency: str | None = None,
    dry_run: bool = False,
    sbatch: str = "sbatch",
    **job_options,
) -> SlurmSubmitResult:
    job = build_experiment_job(cfg, **job_options)
    if not dry_run:
        write_job(job)
    result = submit_script(
        job.script_path, dependency=dependency, dry_run=dry_run, sbatch=sbatch
    )
    return SlurmSubmitResult(
        result.command, job.script_path, job.script, result.job_id, result.stdout
    )
