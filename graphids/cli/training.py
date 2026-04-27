"""Training commands: fit, test.

Both share the same prelude (render → validate → build) with the
pipeline driver (``orchestrate/run.py``), then dispatch through
``orchestrate.stage.train`` / ``orchestrate.stage.evaluate`` so the
CLI and the pipeline loop produce identical markers, OTel wiring,
and GPU-reset semantics.
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


def _prepare(
    config: Path,
    tla: list[Any] | None,
    overrides: list[Any] | None,
) -> tuple[ResolvedConfig, object]:
    """Shared prelude: render → resolve → wire OTel → build.

    Returns ``(resolved, artifacts)``. ``--set`` overrides flow into render
    as ``std.extVar('overrides')`` and are applied by ``std.mergePatch`` at
    each ablation preset's apex (one mechanism, replaces the prior in-place
    Python mutator + jsonnet ``apply_dotted`` pair). Heavy imports live
    inside the function so the app stays login-node-safe.
    """
    from graphids._mlflow import ensure_tracking_uri
    from graphids._otel import init_providers, wire_file_exporters
    from graphids.cli.app import dotted_to_nested
    from graphids.config.jsonnet import render
    from graphids.orchestrate import build

    # Idempotent — Typer's root callback calls this on the CLI path. On the
    # submitit compute-node path, _TrainingJob.__call__ bypasses the callback,
    # so we initialise here too. Guarded by _providers global; second call no-ops.
    init_providers(
        "graphids",
        wandb_entity=os.environ.get("WANDB_ENTITY", ""),
        wandb_project=os.environ.get("WANDB_PROJECT", "graphids"),
    )
    _ensure_spawn()
    _configure_cpu_threads()
    ensure_tracking_uri()

    rendered = render(
        config,
        tla=dict(tla or []) or None,
        set_overrides=dotted_to_nested(overrides),
    )
    resolved = ResolvedConfig.from_rendered(rendered, stage_name=config.stem)
    if resolved.run_dir is not None:
        wire_file_exporters(resolved.run_dir)
        resolved.run_dir.joinpath("resolved.json").write_text(
            resolved.validated.model_dump_json(indent=2)
        )
        resolved.run_dir.joinpath("overrides.json").write_text(
            json.dumps({"tla": dict(tla or []), "set": list(overrides or [])}, indent=2)
        )
    artifacts = build(resolved)
    return resolved, artifacts


@app.command(rich_help_panel="Training")
def fit(
    config: ConfigPath,
    tla: TlaList = None,
    set_: SetList = None,
    ckpt_path: CkptPath = None,
) -> None:
    """Train a model from a jsonnet stage config."""
    from graphids.orchestrate import train

    resolved, artifacts = _prepare(config, tla, set_)
    train(artifacts, resolved, resume_from=ckpt_path)


@app.command(rich_help_panel="Training")
def test(
    config: ConfigPath,
    tla: TlaList = None,
    set_: SetList = None,
    ckpt_path: CkptPath = None,
) -> None:
    """Evaluate a trained model on the test set."""
    from graphids.orchestrate import evaluate

    resolved, artifacts = _prepare(config, tla, set_)
    # When --ckpt-path is explicit, it overrides the resolved ckpt_file.
    if ckpt_path:
        resolved = replace(resolved, ckpt_file=Path(ckpt_path))
    evaluate(artifacts, resolved)
