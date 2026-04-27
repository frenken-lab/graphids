"""Training commands: fit, test.

CLI path renders the jsonnet preset, then hands the rendered dict to
:func:`run_rendered`. The submitit compute-node path
(``slurm.submit._TrainingJob.__call__``) calls :func:`run_rendered`
directly with a dict that was rendered ONCE on the login node — so the
compute node never re-evaluates jsonnet, and the
submission-time/execution-time config can't drift while the job is queued.
"""

from __future__ import annotations

import json
import multiprocessing
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

from graphids.cli.app import CkptPath, ConfigPath, SetList, TlaList, app
from graphids.orchestrate import ResolvedConfig

_SPAWN_SET = False
_THREADS_SET = False


def _ensure_spawn() -> None:
    """Set mp start method to ``spawn`` + tensor IPC to ``file_system``.

    Must run before any CUDA-touching DataLoader worker is spawned —
    fork + CUDA is a silent segfault (see critical-constraints.md).
    Idempotent.
    """
    global _SPAWN_SET  # noqa: PLW0603
    if _SPAWN_SET:
        return
    import torch.multiprocessing  # noqa: PLC0415

    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    torch.multiprocessing.set_sharing_strategy("file_system")
    _SPAWN_SET = True


def _configure_cpu_threads() -> None:
    """Pin torch + OMP thread counts to the SLURM CPU quota. Idempotent.

    PyTorch defaults intra-op threads to ``os.cpu_count()`` (node-wide,
    ignoring SLURM cgroup affinity). On a node with ``--cpus-per-task=16``,
    torch will spawn 64+ threads across NUMA domains and BLAS backends will
    fight for cores. We pin intra-op to the allocation, interop to 1
    (cross-op fork-join double-subscribes once intra-op fills the quota),
    and mirror to ``OMP_NUM_THREADS`` / ``MKL_NUM_THREADS`` so BLAS backends
    that read env (not torch) stay in sync.
    """
    global _THREADS_SET  # noqa: PLW0603
    if _THREADS_SET:
        return
    import torch  # noqa: PLC0415

    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    n = (int(slurm) if slurm and slurm.isdigit() else None) or os.cpu_count() or 1
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # Some torch op already launched — log and continue with whatever
        # interop count is locked in. Intra-op is still applied, and that's
        # where CPU-bound ops get most of their wins.
        from graphids._otel import get_logger  # noqa: PLC0415

        get_logger(__name__).warning("cpu_threads_interop_locked", intra_op=n)
    _THREADS_SET = True


def _setup_runtime() -> None:
    """Idempotent boot: providers + spawn + threads + MLflow URI.

    Runs from both the CLI path (Typer's root callback set up providers
    already; this re-call is a no-op) and the submitit compute-node path
    (callback was never invoked; this is the only place these get set).
    """
    from graphids._mlflow import ensure_tracking_uri
    from graphids._otel import init_providers

    init_providers(
        "graphids",
        wandb_entity=os.environ.get("WANDB_ENTITY", ""),
        wandb_project=os.environ.get("WANDB_PROJECT", "graphids"),
    )
    _ensure_spawn()
    _configure_cpu_threads()
    ensure_tracking_uri()


def run_rendered(
    *,
    action: str,
    rendered: dict[str, Any],
    stage_name: str,
    tla_log: list[tuple[str, Any]] | None = None,
    set_log: list[tuple[str, Any]] | None = None,
    ckpt_path: str | None = None,
) -> None:
    """Run fit/test from a pre-rendered config dict.

    The single compute-node entrypoint shared by the CLI ``fit``/``test``
    commands and ``slurm.submit._TrainingJob.__call__``. ``rendered`` was
    produced by :func:`graphids.config.jsonnet.render` (with TLAs +
    ``--set`` overrides already baked in); we never re-evaluate jsonnet
    here. ``tla_log`` / ``set_log`` are the original flag lists, kept
    only for the ``overrides.json`` provenance sidecar — the rendered
    dict is the source of truth for execution.
    """
    from graphids._otel import wire_file_exporters
    from graphids.orchestrate import build_run, evaluate, train

    _setup_runtime()
    resolved = ResolvedConfig.from_rendered(rendered, stage_name=stage_name)
    if resolved.run_dir is not None:
        wire_file_exporters(resolved.run_dir)
        resolved.run_dir.joinpath("resolved.json").write_text(
            resolved.validated.model_dump_json(indent=2)
        )
        resolved.run_dir.joinpath("overrides.json").write_text(
            json.dumps({"tla": dict(tla_log or []), "set": list(set_log or [])}, indent=2)
        )
    artifacts = build_run(resolved.rendered, validated=resolved.validated, reset_gpu=True)
    if action == "fit":
        train(artifacts, resolved, resume_from=ckpt_path)
        return
    if ckpt_path:
        resolved = replace(resolved, ckpt_file=Path(ckpt_path))
    evaluate(artifacts, resolved)


def _dispatch(
    action: str,
    config: Path,
    tla: list[Any] | None,
    set_: list[Any] | None,
    ckpt_path: str | None,
) -> None:
    """Render the preset, then call :func:`run_rendered` — shared fit/test body."""
    from graphids.config.jsonnet import render_with_flags

    run_rendered(
        action=action,
        rendered=render_with_flags(config, tla, set_),
        stage_name=config.stem,
        tla_log=list(tla or []),
        set_log=list(set_ or []),
        ckpt_path=ckpt_path,
    )


@app.command(rich_help_panel="Training")
def fit(
    config: ConfigPath,
    tla: TlaList = None,
    set_: SetList = None,
    ckpt_path: CkptPath = None,
) -> None:
    """Train a model from a jsonnet stage config."""
    _dispatch("fit", config, tla, set_, ckpt_path)


@app.command(rich_help_panel="Training")
def test(
    config: ConfigPath,
    tla: TlaList = None,
    set_: SetList = None,
    ckpt_path: CkptPath = None,
) -> None:
    """Evaluate a trained model on the test set."""
    _dispatch("test", config, tla, set_, ckpt_path)
