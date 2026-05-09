"""Submit one blueprint row as a SLURM job via direct ``sbatch``.

Single primitive. Profiles are typed Python constants in ``_PROFILES``
(keyed ``[mode][cluster][length]``), replacing the external
``submit_profiles.json``. ``submit_row()`` writes an sbatch script to
``{script_dir}/{job_name}.sh``, calls ``sbatch --parsable``, returns the
job id.

Script body shape:

    source <venv>/../.env  (if present)
    source <venv>/bin/activate
    export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK   # gpu; cpu pins to 1
    export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
    export MLFLOW_TRACKING_URI="sqlite:///<lake_root>/mlflow.db"
    export MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=true
    srun --ntasks=1 python -m graphids exec --row '<json>' [--ckpt-path X]

Per ``chassis-invariants.md`` (drift resistance): ONLY caller of sbatch.
Both ``cli.commands.submit_cli`` (single row) and ``cli.plans.plans_submit``
(multi-row) ultimately invoke this.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class _Profile:
    partition: str
    cores_per_node: int
    mem_per_node: int  # GiB
    walltime: str
    gpus_per_node: int = 0
    signal_delay_s: int | None = None


_PROFILES: dict[str, dict[str, dict[str, _Profile]]] = {
    "gpu": {
        "pitzer": {
            "short": _Profile("gpudebug", 8, 48, "01:00:00", 1, 300),
            "long": _Profile("gpu", 8, 48, "03:00:00", 1, 300),
        },
        "cardinal": {
            "short": _Profile("debug", 8, 48, "01:00:00", 1, 300),
            "long": _Profile("gpu", 8, 48, "01:30:00", 1, 300),
        },
        "ascend": {
            "short": _Profile("debug", 8, 48, "01:00:00", 1, 300),
            "long": _Profile("gpu", 8, 48, "01:45:00", 1, 300),
        },
    },
    "cpu": {
        "pitzer": {
            "short": _Profile("debug-cpu", 16, 64, "01:00:00"),
            "long": _Profile("cpu", 16, 64, "04:00:00"),
        },
        "cardinal": {
            "short": _Profile("debug", 16, 64, "01:00:00"),
            "long": _Profile("cpu", 16, 64, "03:00:00"),
        },
        "ascend": {
            "short": _Profile("debug", 16, 64, "01:00:00"),
            "long": _Profile("cpu", 16, 64, "04:00:00"),
        },
    },
}


def _lookup(mode: str, cluster: str, length: str) -> _Profile:
    try:
        return _PROFILES[mode][cluster][length]
    except KeyError:
        valid = sorted(
            f"{m}/{c}/{l}" for m, cv in _PROFILES.items() for c, lv in cv.items() for l in lv
        )
        raise KeyError(
            f"No profile for mode={mode!r} cluster={cluster!r} length={length!r}. Valid: {valid}"
        ) from None


_SCRIPT = """\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={log_out}
#SBATCH --error={log_err}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time={walltime}
#SBATCH --partition={partition}
#SBATCH --cpus-per-task={cores_per_node}
#SBATCH --mem={mem_per_node}g
#SBATCH --account={account}
#SBATCH --comment=graphids.plan_id={plan_id}
{extra_directives}
{worker_init}

srun --ntasks=1 {command}
"""


def submit_row(
    row,
    *,
    cluster: str,
    length: str = "long",
    ckpt_path: str | None = None,
    depends_on_afterok: str | None = None,
    depends_on_afterany: str | None = None,
    account: str | None = None,
    venv_path: str | None = None,
) -> str:
    """Submit ``row``; return the job id."""
    from graphids.paths import lake_root
    from graphids.plan.rows import TrainRow

    account = account or os.environ.get("GRAPHIDS_SLURM_ACCOUNT", "")
    if not account:
        raise RuntimeError("SLURM account unset — pass account= or set GRAPHIDS_SLURM_ACCOUNT")
    venv = venv_path or str(_REPO_ROOT / ".venv")

    p = _lookup(row.resources.mode, cluster, length)

    if isinstance(row, TrainRow):
        script_dir = Path(row.identity.run_dir) / ".slurm_scripts"
        job_name = row.identity.jobname
    else:
        lake_env = os.environ.get("GRAPHIDS_LAKE_ROOT", str(Path.cwd()))
        script_dir = Path(lake_env) / "slurm" / "scripts" / row.name
        job_name = row.name
    script_dir.mkdir(parents=True, exist_ok=True)

    # CPU-mode: pin thread pools to 1 (OpenMP/MKL spawn overhead on 16-core alloc).
    if row.resources.mode == "cpu":
        thread_exports = [
            "export OMP_NUM_THREADS=1",
            "export MKL_NUM_THREADS=1",
            "export PYTORCH_NUM_THREADS=1",
        ]
    else:
        thread_exports = [
            "export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK",
            "export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK",
        ]

    worker_init = "\n".join(
        [
            f'if [ -f "{venv}/../.env" ]; then source "{venv}/../.env"; fi',
            f"source {venv}/bin/activate",
            *thread_exports,
            f'export MLFLOW_TRACKING_URI="sqlite:///{lake_root().rstrip("/")}/mlflow.db"',
            "export MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=true",
        ]
    )

    # Auto-resolve test ckpt_path from the deterministic fit output path.
    if ckpt_path is None and isinstance(row, TrainRow) and row.action == "test":
        ckpt_path = f"{row.identity.run_dir}/checkpoints/best_model.ckpt"

    cmd_parts = ["python", "-m", "graphids", "exec", "--row", row.model_dump_json()]
    if ckpt_path:
        cmd_parts += ["--ckpt-path", ckpt_path]

    extra: list[str] = []
    if p.gpus_per_node:
        extra.append(f"#SBATCH --gpus-per-node={p.gpus_per_node}")
    if p.signal_delay_s is not None:
        extra.append(f"#SBATCH --signal=USR2@{p.signal_delay_s}")
    if depends_on_afterok:
        extra.append(f"#SBATCH --dependency=afterok:{depends_on_afterok}")
    if depends_on_afterany:
        extra.append(f"#SBATCH --dependency=afterany:{depends_on_afterany}")

    script = _SCRIPT.format(
        job_name=job_name,
        log_out=script_dir / f"{job_name}.stdout",
        log_err=script_dir / f"{job_name}.stderr",
        walltime=p.walltime,
        partition=p.partition,
        cores_per_node=p.cores_per_node,
        mem_per_node=p.mem_per_node,
        account=account,
        plan_id=row.plan_id,
        extra_directives="\n".join(extra),
        worker_init=worker_init,
        command=shlex.join(cmd_parts),
    )

    script_path = script_dir / f"{job_name}.sh"
    script_path.write_text(script)

    result = subprocess.run(
        ["sbatch", "--parsable", str(script_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"sbatch failed (rc={result.returncode}): {result.stderr.strip()}")
    # --parsable emits "<jobid>" or "<jobid>;<cluster>" for federated clusters
    return result.stdout.strip().split(";")[0]
