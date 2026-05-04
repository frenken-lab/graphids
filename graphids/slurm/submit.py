"""Submit one blueprint row as a SLURM job via :class:`parsl.providers.SlurmProvider`.

Single primitive — no DFK, no executor pool. Reads cluster directives from
``configs/resources/submit_profiles.json`` (Parsl-shaped kwargs), instantiates
a fresh ``SlurmProvider`` per call, submits a literal bash command, returns
the job id. The calling process exits.

Bash command shape:
    python -m graphids exec --row '<json>' [--ckpt-path <X>]

Per ``.claude/rules/single-submission-primitive.md``: the ONLY caller of
``SlurmProvider.submit``. Pipelines walk the JSON blueprint and invoke
``submit_row`` per row externally — no Python pipeline driver here.
"""

from __future__ import annotations

import json
import os
import shlex
from pathlib import Path

from parsl.launchers import SrunLauncher
from parsl.providers import SlurmProvider

from graphids.blueprint import Row, TrainRow

_PROFILES = Path(__file__).resolve().parents[2] / "configs" / "resources" / "submit_profiles.json"


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

    ``depends_on_*`` adds ``--dependency=after{ok,any}:<jid>``. Use ``afterany``
    for preempt-resume (prior job may have failed cleanly with ckpt) and
    ``afterok`` for chained data deps. ``account`` defaults to
    ``$GRAPHIDS_SLURM_ACCOUNT``; ``venv_path`` to the project ``.venv``.

    Profile JSON keys map directly to ``SlurmProvider`` kwargs — the only
    extra is ``signal_delay_s``, which becomes ``#SBATCH --signal=USR2@N``
    so ``runtime.py``'s SIGUSR2 trap can resubmit before walltime.
    """
    account = account or os.environ.get("GRAPHIDS_SLURM_ACCOUNT", "")
    if not account:
        raise RuntimeError("SLURM account unset — pass account= or set GRAPHIDS_SLURM_ACCOUNT")
    venv_path = venv_path or str(Path(__file__).resolve().parents[2] / ".venv")

    profile = dict(json.loads(_PROFILES.read_text())[row.resources.mode][cluster][length])
    directives: list[str] = []
    if (delay := profile.pop("signal_delay_s", None)) is not None:
        directives.append(f"#SBATCH --signal=USR2@{int(delay)}")
    if depends_on_afterok:
        directives.append(f"#SBATCH --dependency=afterok:{depends_on_afterok}")
    if depends_on_afterany:
        directives.append(f"#SBATCH --dependency=afterany:{depends_on_afterany}")

    if isinstance(row, TrainRow):
        script_dir = Path(row.identity.run_dir) / ".parsl_scripts"
        job_name = row.identity.jobname
    else:
        lake = os.environ.get("GRAPHIDS_LAKE_ROOT", str(Path.cwd()))
        script_dir = Path(lake) / "slurm" / "scripts" / row.name
        job_name = row.name
    script_dir.mkdir(parents=True, exist_ok=True)

    provider = SlurmProvider(
        **profile,
        account=account,
        scheduler_options="\n".join(directives),
        worker_init=(
            f'if [ -f "{venv_path}/../.env" ]; then source "{venv_path}/../.env"; fi && '
            f"source {venv_path}/bin/activate"
        ),
        launcher=SrunLauncher(),
        exclusive=False,
    )
    provider.script_dir = str(script_dir)

    row_json = row.model_dump_json()
    parts = ["python", "-m", "graphids", "exec", "--row", shlex.quote(row_json)]
    if ckpt_path:
        parts += ["--ckpt-path", shlex.quote(ckpt_path)]
    return provider.submit(" ".join(parts), tasks_per_node=1, job_name=job_name)
