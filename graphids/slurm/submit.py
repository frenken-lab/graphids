"""SLURM submitter — thin wrapper over ``submitit.AutoExecutor``.

Two shapes, one function:

* **preset mode** ships a ``_TrainingJob`` (Python callable) that imports
  and invokes ``graphids.cli.training.{fit,test}`` directly on the compute
  node. Exceptions come back as real Python tracebacks via submitit's
  pickled result/error files; ``job.result()`` returns whatever the
  training command returns. ``_TrainingJob.checkpoint()`` makes SLURM
  preemption auto-requeue with ``ckpt_path`` pointing at
  ``{run_dir}/checkpoints/last.ckpt``. The profile sets
  ``slurm_signal_delay_s=300`` so sbatch sends SIGUSR2 five minutes
  before walltime; submitit's handler catches it and sbatch-queues the
  resumed job via afterany. No manual resubmit loop.

* **ops mode** (``--command``) ships ``submitit.helpers.CommandFunction``
  — shell strings genuinely need a shell.

Profile JSON (``configs/resources/submit_profiles.json``) stores raw
submitit AutoExecutor kwargs keyed ``[mode][cluster][length]``; there is
nothing to parse. ``slurm_setup`` sources ``scripts/slurm/_preamble.sh``
inside the sbatch shell so module-load / venv-activate runs before
submitit's ``srun python -u -m submitit.core._submit``.
"""

from __future__ import annotations

import json
import math
import os
import shlex
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PROFILES: dict[str, Any] = json.loads(
    (_PROJECT_ROOT / "configs" / "resources" / "submit_profiles.json").read_text()
)
_PREAMBLE_SH = _PROJECT_ROOT / "scripts" / "slurm" / "_preamble.sh"


def _align_cpus_to_mem(params: dict[str, Any], cluster: str) -> None:
    """Bump ``cpus_per_task`` so SLURM doesn't have to. Mutates in place.

    The profile's ``cpus_per_task`` is the floor (parallelism need);
    ``ceil(mem_gb / mem_per_cpu_gb)`` is the SLURM-policy minimum. Take the max.
    Cluster ratio data is in ``configs/resources/submit_profiles.json``
    under ``cluster_policy``.
    """
    ratio = int(_PROFILES["cluster_policy"][cluster]["mem_per_cpu_gb"])
    floor = int(params.get("cpus_per_task", 1))
    mem_gb = int(params["mem_gb"])
    params["cpus_per_task"] = max(floor, math.ceil(mem_gb / ratio))


def ensure_env_loaded() -> None:
    """Populate ``os.environ`` from ``.env`` via ``python-dotenv`` (transitive via pydantic-settings).

    Distinct from ``GraphIDSSettings``'s pydantic-settings load: that one
    builds a typed Settings object but does NOT mutate ``os.environ``, and
    ``submit()`` reads vars (``SLURM_ACCOUNT``, ``LAKE_ROOT``, ...) via
    ``os.environ.get`` so the SLURM env composes with the inherited shell.
    """
    from dotenv import load_dotenv

    load_dotenv(_PROJECT_ROOT / ".env", override=False)


# --------------------------------------------------------------------------
# The submittable work unit. submitit pickles this; the compute node runs
# `python -m submitit.core._submit {folder}` which unpickles + calls.
# --------------------------------------------------------------------------


@dataclass
class _TrainingJob:
    """Pickle-safe callable that invokes ``graphids.cli.training.{fit,test}``.

    ``checkpoint()`` is called by submitit's SIGUSR2 handler when SLURM
    sends the preemption signal (``slurm_signal_delay_s`` seconds before
    walltime; see profile). We look up ``{run_dir}/checkpoints/last.ckpt``
    and return a ``DelayedSubmission`` so submitit sbatch-queues the resumed
    job with that ckpt as the resume source — free preemption recovery.

    ``run_dir`` is rendered once at submit time (pure function of preset +
    tlas + RUN_ROOT) and stamped into the pickled payload, so the compute
    node never re-renders jsonnet during preemption recovery.
    """

    action: str  # "fit" | "test"
    config: str
    tlas: list[tuple[str, Any]] = field(default_factory=list)
    sets: list[tuple[str, Any]] = field(default_factory=list)
    ckpt_path: str | None = None
    run_dir: str | None = None

    def __call__(self) -> None:
        from graphids.cli.training import fit, test

        fn = fit if self.action == "fit" else test
        fn(
            config=Path(self.config),
            tla=list(self.tlas) or None,
            set_=list(self.sets) or None,
            ckpt_path=Path(self.ckpt_path) if self.ckpt_path else None,
        )

    def checkpoint(self, *args: Any, **kwargs: Any):  # noqa: ANN401 — submitit API
        import submitit

        resume = self._last_ckpt() or self.ckpt_path
        return submitit.helpers.DelayedSubmission(replace(self, ckpt_path=resume))

    def _last_ckpt(self) -> str | None:
        if not self.run_dir:
            return None
        last = Path(self.run_dir) / "checkpoints" / "last.ckpt"
        return str(last) if last.exists() else None


# --------------------------------------------------------------------------
# The one real entrypoint
# --------------------------------------------------------------------------


def submit(  # noqa: PLR0913 — every flag is a real public surface
    *,
    preset: Path | None = None,
    command: str | None = None,
    action: str = "fit",
    mode: str | None = None,
    length: str = "long",
    cluster: str | None = None,
    tlas: Sequence[tuple[str, Any]] = (),
    sets: Sequence[tuple[str, Any]] = (),
    ckpt_path: str | None = None,
    mem_gb: int | None = None,
    timeout_min: int | None = None,
    time_from_history: bool = False,
    dep_jids: Sequence[int] = (),
    dry_run: bool = False,
) -> int | None:
    """Submit one SLURM job via submitit. Returns the jid, or ``None`` for dry-run.

    ``tlas`` / ``sets`` are passed verbatim to ``_TrainingJob``; callers
    (CLI / dag.py) build the list themselves — no flat-flag→TLA sugar
    layer here. ``ckpt_path`` is the fit/test ``--ckpt-path`` passthrough,
    distinct from any ``ckpt_path`` TLA the caller may include in ``tlas``.

    ``dep_jids`` are afterok dependency jids; non-positive values
    (typically ``0`` from a skipped upstream's ``--skip-if-finished``) are
    filtered before the dependency string is composed. Real SLURM jids are
    always positive.
    """
    ensure_env_loaded()

    if preset is None and not command:
        raise ValueError("submit() needs preset= or command=")
    if mode is None:
        if preset is None:
            raise ValueError("--command requires mode='gpu' or 'cpu'")
        mode = "gpu"
    cluster = cluster or os.environ.get("GRAPHIDS_CLUSTER", "pitzer")

    params: dict[str, Any] = dict(_PROFILES[mode][cluster][length])
    if mem_gb is not None:
        params["mem_gb"] = mem_gb
    if timeout_min is not None:
        params["timeout_min"] = timeout_min
    if time_from_history and timeout_min is None and preset and length == "long":
        from graphids.slurm.sizing import estimate_walltime_minutes

        dataset = next((v for k, v in tlas if k == "dataset"), None)
        if dataset:
            mins = estimate_walltime_minutes(cluster, preset.parent.name, dataset)
            if mins:
                params["timeout_min"] = mins
    _align_cpus_to_mem(params, cluster)

    # --- Build the work unit -----------------------------------------------
    import submitit
    from submitit.helpers import CommandFunction

    payload: Any
    if preset and not command:
        # Render once on the login node so the SIGUSR2 handler on the compute
        # node doesn't have to re-evaluate jsonnet to find last.ckpt.
        from graphids.config.jsonnet import render

        rendered = render(preset, tla=dict(tlas) or None)
        rendered_run_dir = (rendered.get("trainer") or {}).get("default_root_dir") or None
        payload = _TrainingJob(
            action=action,
            config=str(preset),
            tlas=list(tlas),
            sets=list(sets),
            ckpt_path=ckpt_path,
            run_dir=rendered_run_dir,
        )
        jobname = f"graphids-{action}-{preset.stem}"
    else:
        assert command is not None
        payload = CommandFunction(["bash", "-c", command])
        jobname = f"graphids-{_jobname_for_cmd(command)}"

    # --- Assemble additional SLURM params (cluster, dep, signal) -----------
    additional = dict(params.pop("slurm_additional_parameters", {}))
    additional["clusters"] = cluster
    # afterok dep jids come from --depends-on resolution (RUNNING upstream
    # contributes its slurm.slurm_job_id MLflow tag). Real SLURM jids are
    # always positive; defensively filter to be safe.
    live_deps = [str(j) for j in dep_jids if j > 0]
    if live_deps:
        additional["dependency"] = f"afterok:{':'.join(live_deps)}"

    setup = [f"source {_PREAMBLE_SH}"]
    if mode == "cpu":
        setup.insert(0, "export SKIP_CUDA_CONF=1")

    log_dir = os.environ.get("GRAPHIDS_SLURM_LOG_DIR")
    if not log_dir:
        lake = os.environ.get("GRAPHIDS_LAKE_ROOT")
        if not lake:
            raise RuntimeError(
                "GRAPHIDS_LAKE_ROOT unset; source .env or set GRAPHIDS_SLURM_LOG_DIR"
            )
        log_dir = f"{lake}/slurm"

    executor = submitit.AutoExecutor(folder=log_dir)
    executor.update_parameters(
        slurm_account=os.environ.get("GRAPHIDS_SLURM_ACCOUNT") or _raise_account(),
        slurm_job_name=jobname,
        slurm_setup=setup,
        slurm_additional_parameters=additional,
        **params,
    )

    if dry_run:
        _print_dry_run(executor, payload, jobname)
        return None

    job = executor.submit(payload)
    print(f"Submitted job {job.job_id} on cluster {cluster}", file=sys.stderr)
    return int(job.job_id.split("_", 1)[0])


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def _raise_account() -> str:
    raise RuntimeError("GRAPHIDS_SLURM_ACCOUNT must be set (source .env)")


def _jobname_for_cmd(command: str) -> str:
    skip = {"python", "graphids", "-m"}
    return next(
        (t for t in shlex.split(command) if t and t[0].isalpha() and t not in skip),
        "cmd",
    )


def _print_dry_run(executor: Any, payload: Any, jobname: str) -> None:  # noqa: ANN401
    """Print the sbatch script submitit would generate. Uses internal API
    (``_make_submission_file_text``) — stable since submitit 1.0."""
    try:
        script = executor._executor._make_submission_file_text(  # noqa: SLF001
            command="<stub>", uid="dry"
        )
    except Exception as exc:
        print(f"# {jobname}: dry-run script generation failed ({exc})", file=sys.stderr)
        return
    payload_desc = _describe_payload(payload)
    print(f"# === {jobname} (dry-run) ===", file=sys.stderr)
    print(f"# payload: {payload_desc}", file=sys.stderr)
    for line in script.splitlines():
        print(f"# {line}" if line else "#", file=sys.stderr)


def _describe_payload(payload: Any) -> str:  # noqa: ANN401
    if isinstance(payload, _TrainingJob):
        bits = [f"{payload.action} config={payload.config}"]
        if payload.tlas:
            bits.append(f"tlas={payload.tlas}")
        if payload.sets:
            bits.append(f"sets={payload.sets}")
        if payload.ckpt_path:
            bits.append(f"ckpt={payload.ckpt_path}")
        return " ".join(bits)
    return shlex.join(getattr(payload, "command", []) or ["<opaque>"])
