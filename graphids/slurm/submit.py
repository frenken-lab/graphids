"""Submit one blueprint row as a SLURM job via :class:`parsl.providers.SlurmProvider`.

Single primitive — no DFK, no executor pool, no monitoring DB. Reads cluster
directives from ``configs/resources/submit_profiles.json``, instantiates a
fresh ``SlurmProvider`` per call, submits a literal bash command, returns
the job id. The calling process exits.

Bash command shape:
    python -m graphids exec --row '<json>' [--ckpt-path <X>]

Per ``.claude/rules/single-submission-primitive.md``: this is the ONLY place
that calls ``SlurmProvider.submit``. Pipelines walk the JSON blueprint and
invoke ``submit_row`` per row externally — no Python pipeline driver here.
"""

from __future__ import annotations

import json
import os
import shlex
from pathlib import Path
from typing import Any

from parsl.launchers import SrunLauncher
from parsl.providers import SlurmProvider

from graphids.blueprint import Row, TrainRow

_PROFILES = Path(__file__).resolve().parents[2] / "configs" / "resources" / "submit_profiles.json"


def _profile(mode: str, cluster: str, length: str) -> dict[str, Any]:
    """Look up the SLURM kwargs for ``[mode][cluster][length]``."""
    profiles = json.loads(_PROFILES.read_text())
    try:
        return profiles[mode][cluster][length]
    except KeyError as e:
        raise ValueError(
            f"no profile for mode={mode!r} cluster={cluster!r} length={length!r}"
        ) from e


def _to_walltime(timeout_min: int) -> str:
    """Convert minutes (submitit/profile shape) → ``HH:MM:SS`` (parsl shape)."""
    h, m = divmod(int(timeout_min), 60)
    return f"{h:02d}:{m:02d}:00"


def _build_provider(
    profile: dict[str, Any],
    *,
    account: str,
    venv_path: str,
    extra_directives: list[str] | None = None,
) -> SlurmProvider:
    """Translate a profile dict into a configured ``SlurmProvider``.

    Profile keys map to Parsl kwargs:
        slurm_partition       → partition
        cpus_per_task         → cores_per_node
        mem_gb                → mem_per_node
        timeout_min           → walltime ("HH:MM:SS")
        gpus_per_node         → gpus_per_node
        slurm_signal_delay_s  → scheduler_options "#SBATCH --signal=USR2@N"
    """
    directives = list(extra_directives or [])
    if (delay := profile.get("slurm_signal_delay_s")) is not None:
        directives.append(f"#SBATCH --signal=USR2@{int(delay)}")
    return SlurmProvider(
        partition=profile["slurm_partition"],
        account=account,
        cores_per_node=profile["cpus_per_task"],
        mem_per_node=profile["mem_gb"],
        gpus_per_node=profile.get("gpus_per_node"),
        walltime=_to_walltime(profile["timeout_min"]),
        scheduler_options="\n".join(directives),
        worker_init=f"source {venv_path}/bin/activate",
        launcher=SrunLauncher(),
        exclusive=False,
    )


def _build_command(row: Row, ckpt_path: str | None) -> str:
    """Bash command the SLURM job runs: in-process row execution via the CLI."""
    row_json = row.model_dump_json()
    parts = ["python", "-m", "graphids", "exec", "--row", shlex.quote(row_json)]
    if ckpt_path:
        parts += ["--ckpt-path", shlex.quote(ckpt_path)]
    return " ".join(parts)


def _jobname(row: Row) -> str:
    """SLURM job name. TrainRow carries a structured ``identity.jobname``;
    ExtractRow / CmdRow only carry ``name`` (no MLflow run identity)."""
    if isinstance(row, TrainRow):
        return row.identity.jobname
    return row.name


def submit_row(
    row: Row,
    *,
    cluster: str,
    length: str = "long",
    ckpt_path: str | None = None,
    depends_on_afterok: str | None = None,
    depends_on_afterany: str | None = None,
    account: str | None = None,
    venv_path: str | None = None,
) -> str:
    """Submit ``row`` as a SLURM job. Return the job id.

    ``depends_on_*`` adds ``--dependency=after{ok,any}:<jid>`` to the sbatch
    directives. Use ``afterany`` for preempt-resume (where the prior job may
    have failed cleanly with ckpt) and ``afterok`` for chained data deps.
    ``account`` defaults to ``$GRAPHIDS_SLURM_ACCOUNT``; ``venv_path`` to the
    project ``.venv`` next to ``pyproject.toml``.
    """
    account = account or os.environ.get("GRAPHIDS_SLURM_ACCOUNT", "")
    if not account:
        raise RuntimeError("SLURM account unset — pass account= or set GRAPHIDS_SLURM_ACCOUNT")
    venv_path = venv_path or str(Path(__file__).resolve().parents[2] / ".venv")
    profile = _profile(row.resources.mode, cluster, length)
    extras: list[str] = []
    if depends_on_afterok:
        extras.append(f"#SBATCH --dependency=afterok:{depends_on_afterok}")
    if depends_on_afterany:
        extras.append(f"#SBATCH --dependency=afterany:{depends_on_afterany}")
    provider = _build_provider(
        profile, account=account, venv_path=venv_path, extra_directives=extras
    )
    command = _build_command(row, ckpt_path)
    return provider.submit(command, tasks_per_node=1, job_name=_jobname(row))
