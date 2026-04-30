"""SLURM submitter — one function, two shapes.

:func:`submit` is the single entrypoint for both the Typer ``submit``
command and any programmatic caller. It accepts the full CLI flag
surface (``preset`` / ``--dataset`` / ``--seed`` / ``--depends-on`` /
``--skip-if-finished`` / ...), resolves dependencies + flag→TLA mapping
inline, and dispatches to ``submitit.AutoExecutor``.

Two payload shapes via ``preset`` XOR ``command``:

* **preset mode** — ``submit`` renders the jsonnet on the login node and
  ships a pickled :class:`_TrainingJob` that calls
  ``graphids.cli.training.run_rendered`` on the compute node (no jsonnet
  re-evaluation, no submission/execution config drift).
  ``_TrainingJob.checkpoint()`` makes SLURM preemption auto-requeue with
  ``ckpt_path`` pointing at ``{run_dir}/checkpoints/last.ckpt``. The
  profile sets ``slurm_signal_delay_s=300`` so sbatch sends SIGUSR2 five
  minutes before walltime; submitit's handler catches it and
  sbatch-queues the resumed job via afterany.

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
    """Pickle-safe callable that invokes ``graphids.cli.training.run_rendered``.

    The rendered config dict is produced ONCE by ``submit()`` on the login
    node (with TLAs + ``--set`` overrides already baked in) and pickled
    into this payload. The compute node never re-evaluates jsonnet —
    submission-time and execution-time configs cannot drift while the job
    sits in the queue. ``run_dir`` is read from
    ``rendered['trainer']['default_root_dir']`` so there is no separate
    field to keep in sync.

    ``checkpoint()`` is called by submitit's SIGUSR2 handler when SLURM
    sends the preemption signal (``slurm_signal_delay_s`` seconds before
    walltime; see profile). We look up ``{run_dir}/checkpoints/last.ckpt``
    and return a ``DelayedSubmission`` so submitit sbatch-queues the
    resumed job with that ckpt as the resume source.
    """

    action: str  # "fit" | "test"
    rendered: dict[str, Any]
    stage_name: str
    tla_log: list[tuple[str, Any]] = field(default_factory=list)
    set_log: list[tuple[str, Any]] = field(default_factory=list)
    ckpt_path: str | None = None

    def __call__(self) -> None:
        from graphids.cli.training import run_rendered

        run_rendered(
            action=self.action,
            rendered=self.rendered,
            stage_name=self.stage_name,
            tla_log=self.tla_log,
            set_log=self.set_log,
            ckpt_path=self.ckpt_path,
        )

    def checkpoint(self, *args: Any, **kwargs: Any):  # noqa: ANN401 — submitit API
        import submitit

        resume = self._last_ckpt() or self.ckpt_path
        return submitit.helpers.DelayedSubmission(replace(self, ckpt_path=resume))

    def _last_ckpt(self) -> str | None:
        run_dir = (self.rendered.get("trainer") or {}).get("default_root_dir")
        if not run_dir:
            return None
        last = Path(run_dir) / "checkpoints" / "last.ckpt"
        return str(last) if last.exists() else None


# --------------------------------------------------------------------------
# The one entrypoint
# --------------------------------------------------------------------------


def _infer_group_variant(preset: Path, name: str | None) -> tuple[str, str]:
    """Resolve ``(group, variant)`` from ``--name`` or the preset path convention.

    Convention: ``configs/ablations/<group>/<variant>.jsonnet``. ``--name``
    overrides the convention as ``"group/variant"``. Raises
    :class:`typer.BadParameter` when neither resolves — used by
    ``--skip-if-finished`` to feed the MLflow filter.
    """
    import typer  # noqa: PLC0415

    if name:
        if "/" not in name:
            raise typer.BadParameter(f"--name must be 'group/variant' (got {name!r})")
        group, _, variant = name.partition("/")
        return group, variant
    parts = preset.parts
    if "ablations" in parts:
        idx = parts.index("ablations")
        if idx + 2 < len(parts):
            return parts[idx + 1], preset.stem
    raise typer.BadParameter(
        f"--skip-if-finished cannot infer group/variant from {preset}. "
        "Pass --name <group>/<variant> explicitly."
    )


def submit(  # noqa: PLR0913 — every flag is a real CLI surface
    *,
    preset: Path | None = None,
    command: str | None = None,
    action: str = "fit",
    mode: str | None = None,
    length: str = "long",
    smoke: bool = False,
    cpu: bool = False,
    cluster: str | None = None,
    dataset: str | None = None,
    seed: int | None = None,
    scale: str | None = None,
    ckpt_tla: str | None = None,
    ckpt_path: str | None = None,
    lake_root: str | None = None,
    mem_gb: int | None = None,
    timeout_min: int | None = None,
    time_from_history: bool = False,
    tla: Sequence[tuple[str, Any]] | None = None,
    set_: Sequence[tuple[str, Any]] | None = None,
    depends_on: str | None = None,
    name: str | None = None,
    skip_if_finished: bool = False,
    dry_run: bool = False,
) -> int | None:
    """Submit one SLURM job via submitit.

    Single entrypoint for both the Typer ``submit`` command and any
    programmatic callers. Returns the jid on success, ``None`` when the
    run was skipped (``--skip-if-finished`` hit a FINISHED upstream) or
    when ``--dry-run`` was set. The Typer wrapper prints ``0`` on
    ``None`` (non-chaining sentinel for ``afterok:$jid``).

    Two shapes via ``preset`` XOR ``command``: training mode renders the
    preset on the login node and ships a pickled ``_TrainingJob``;
    ops mode ships ``submitit.helpers.CommandFunction(['bash','-c',...])``.

    Raises :class:`typer.BadParameter` on invalid flag combinations or
    unresolved dependencies.
    """
    import typer  # noqa: PLC0415

    if preset is None and not command:
        raise typer.BadParameter('supply a preset path or --command "..."')
    # --depends-on (MLflow-resolved upstream ckpts) and --ckpt-path
    # (resume the current preset) look similar but mean different things.
    if depends_on and ckpt_path:
        raise typer.BadParameter(
            "--ckpt-path resumes the *current* preset; --depends-on injects "
            "upstream teacher ckpts. Different semantics — pass them on "
            "separate invocations."
        )

    # --- High-level flags → TLA pairs --------------------------------------
    # --ckpt-tla writes the ``ckpt_path`` TLA (jsonnet field), distinct from
    # --ckpt-path (fit/test passthrough).
    flag_tlas: list[tuple[str, Any]] = []
    for key, val in (
        ("dataset", dataset),
        ("scale", scale),
        ("ckpt_path", ckpt_tla),
        ("lake_root", lake_root),
    ):
        if val:
            flag_tlas.append((key, val))
    if seed is not None:
        flag_tlas.append(("seed", seed))

    # Resolve --depends-on BEFORE user --tla so explicit --tla overrides
    # (last-wins on flag_tlas). Hard error on resolution failure.
    afterok_jids: list[int] = []
    if depends_on:
        from graphids.slurm.dependencies import (  # noqa: PLC0415
            DependencyResolutionError,
            parse_depends_on,
            resolve_all,
        )

        if not dataset:
            raise typer.BadParameter("--depends-on requires --dataset")
        try:
            specs = parse_depends_on(depends_on, default_seed=seed)
            dep_tlas, afterok_jids = resolve_all(specs, dataset)
            flag_tlas.extend(dep_tlas)
        except DependencyResolutionError as exc:
            raise typer.BadParameter(str(exc)) from exc

    flag_tlas.extend(tla or ())

    if skip_if_finished:
        if preset is None:
            raise typer.BadParameter("--skip-if-finished requires a preset (no --command form)")
        group, variant = _infer_group_variant(preset, name)
        if not dataset or seed is None:
            raise typer.BadParameter(
                "--skip-if-finished needs both --dataset and --seed so the MLflow lookup is unambiguous"
            )
        from graphids._mlflow import is_finished  # noqa: PLC0415

        phase = "test" if action == "test" else "fit"
        if is_finished(dataset=dataset, group=group, variant=variant, seed=seed, phase=phase):
            return None

    # --- Resolve mode / cluster / profile params ---------------------------
    ensure_env_loaded()
    effective_mode = "cpu" if cpu else (mode or ("gpu" if preset else None))
    if effective_mode is None:
        raise typer.BadParameter("--command requires --mode gpu|cpu")
    effective_length = "short" if smoke else length
    cluster = cluster or os.environ.get("GRAPHIDS_CLUSTER", "pitzer")

    params: dict[str, Any] = dict(_PROFILES[effective_mode][cluster][effective_length])
    if mem_gb is not None:
        params["mem_gb"] = mem_gb
    if timeout_min is not None:
        params["timeout_min"] = timeout_min
    if (
        time_from_history
        and timeout_min is None
        and preset
        and effective_length == "long"
        and dataset
    ):
        from graphids.slurm.sizing import estimate_walltime_minutes  # noqa: PLC0415

        mins = estimate_walltime_minutes(cluster, preset.parent.name, dataset)
        if mins:
            params["timeout_min"] = mins
    _align_cpus_to_mem(params, cluster)

    # --- Build the work unit -----------------------------------------------
    import submitit  # noqa: PLC0415
    from submitit.helpers import CommandFunction  # noqa: PLC0415

    payload: Any
    if preset and not command:
        # Render ONCE on the login node — TLAs + --set overrides are baked
        # into the rendered dict that ships in the pickled payload. The
        # compute node never re-evaluates jsonnet, so the config can't
        # drift between submission and execution.
        from graphids.config.jsonnet import render_with_flags  # noqa: PLC0415

        rendered = render_with_flags(preset, flag_tlas, set_)
        payload = _TrainingJob(
            action=action,
            rendered=rendered,
            stage_name=preset.stem,
            tla_log=list(flag_tlas),
            set_log=list(set_ or []),
            ckpt_path=ckpt_path,
        )
        jobname = f"graphids-{action}-{preset.stem}"
    else:
        assert command is not None
        payload = CommandFunction(["bash", "-c", command])
        jobname = f"graphids-{_jobname_for_cmd(command)}"

    # --- Assemble additional SLURM params (cluster, signal) ----------------
    additional = dict(params.pop("slurm_additional_parameters", {}))
    additional["clusters"] = cluster
    # Real SLURM jids are always positive; filter out the 0 sentinel.
    live_deps = [str(j) for j in afterok_jids if j > 0]

    setup = [f"source {_PREAMBLE_SH}"]
    if effective_mode == "cpu":
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
    if live_deps:
        executor.update_parameters(slurm_dependency=f"afterok:{':'.join(live_deps)}")

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
        bits = [f"{payload.action} stage={payload.stage_name}"]
        if payload.tla_log:
            bits.append(f"tlas={payload.tla_log}")
        if payload.set_log:
            bits.append(f"sets={payload.set_log}")
        if payload.ckpt_path:
            bits.append(f"ckpt={payload.ckpt_path}")
        return " ".join(bits)
    return shlex.join(getattr(payload, "command", []) or ["<opaque>"])
