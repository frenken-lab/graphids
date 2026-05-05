"""Submit one blueprint row as a SLURM job via ``parsl.providers.SlurmProvider``.

Single primitive. Reads cluster directives from
``configs/resources/submit_profiles.json``, instantiates a fresh
``SlurmProvider``, submits a literal bash command, returns the job id.

The bash command shape, with v3 runtime expectations baked into ``worker_init``:

    source <venv>/../.env  (if present)
    source <venv>/bin/activate
    export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
    export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
    export MLFLOW_TRACKING_URI="sqlite:///<lake_root>/mlflow.db"
    export MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=true
    python -m graphids exec --row '<json>' [--ckpt-path X]

The exports replace the ``runtime.py`` Python calls that v3 dropped;
``SLURMEnvironment(auto_requeue=True, requeue_signal=SIGUSR2)`` in
``orchestrate_v3._make_trainer`` handles preempt-resume natively.

Per ``single-submission-primitive.md``: ONLY caller of ``SlurmProvider.submit``.
"""

from __future__ import annotations

import json
import os
import shlex
from pathlib import Path

from parsl.launchers import SrunLauncher
from parsl.providers import SlurmProvider

from graphids.plan.blueprint import Row, TrainRow

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROFILES = _REPO_ROOT / "configs" / "resources" / "submit_profiles.json"


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
    """Submit ``row``; return the job id.

    ``depends_on_afterok``: chained data deps. ``depends_on_afterany``:
    legacy preempt-resume (kept for compat — v3 uses Lightning's
    ``SLURMEnvironment.auto_requeue`` so this rarely fires).
    """
    account = account or os.environ.get("GRAPHIDS_SLURM_ACCOUNT", "")
    if not account:
        raise RuntimeError("SLURM account unset — pass account= or set GRAPHIDS_SLURM_ACCOUNT")
    venv = venv_path or str(_REPO_ROOT / ".venv")

    from graphids.paths import lake_root

    profile = dict(json.loads(_PROFILES.read_text())[row.resources.mode][cluster][length])
    delay = profile.pop("signal_delay_s", None)

    directives = []
    if delay is not None:
        directives.append(f"#SBATCH --signal=USR2@{int(delay)}")
    if depends_on_afterok:
        directives.append(f"#SBATCH --dependency=afterok:{depends_on_afterok}")
    if depends_on_afterany:
        directives.append(f"#SBATCH --dependency=afterany:{depends_on_afterany}")

    if isinstance(row, TrainRow):
        script_dir, job_name = Path(row.identity.run_dir) / ".parsl_scripts", row.identity.jobname
    else:
        lake_env = os.environ.get("GRAPHIDS_LAKE_ROOT", str(Path.cwd()))
        script_dir, job_name = Path(lake_env) / "slurm" / "scripts" / row.name, row.name
    script_dir.mkdir(parents=True, exist_ok=True)

    worker_init = " && ".join([
        f'if [ -f "{venv}/../.env" ]; then source "{venv}/../.env"; fi',
        f"source {venv}/bin/activate",
        "export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK",
        "export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK",
        f'export MLFLOW_TRACKING_URI="sqlite:///{lake_root().rstrip("/")}/mlflow.db"',
        "export MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=true",
    ])

    provider = SlurmProvider(
        **profile,
        account=account,
        scheduler_options="\n".join(directives),
        worker_init=worker_init,
        launcher=SrunLauncher(),
        exclusive=False,
    )
    provider.script_dir = str(script_dir)

    cmd_parts = ["python", "-m", "graphids", "exec", "--row", row.model_dump_json()]
    if ckpt_path:
        cmd_parts += ["--ckpt-path", ckpt_path]
    return provider.submit(shlex.join(cmd_parts), tasks_per_node=1, job_name=job_name)
