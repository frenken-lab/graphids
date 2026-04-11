"""Single-stage primitives — Layer 4 of the orchestrate stack.

``build`` / ``train`` / ``evaluate`` are the atomic verbs between a
``ResolvedConfig`` and a running ``Trainer``. Each takes the resolved
config directly so callers (CLI + pipeline driver) don't have to
unpack ``rendered`` / ``validated`` / ``run_dir`` / ``ckpt_file``
into positional arguments at every call site.

``wire_file_exporters`` is called once by the caller per stage, not by
these primitives — callers that run ``build → train → evaluate`` wire
it once, not twice (one per primitive call).

When ``resolved.run_dir`` is ``None`` (CLI smoke with no
``default_root_dir``), the primitives skip all filesystem side effects:
no markers, no file exporters, no ckpt_path hand-off to ``.test()``.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import torch

from graphids._fs import touch_marker
from graphids._otel import get_logger
from graphids.config.constants import COMPLETE_MARKER, PHASE_MARKERS
from graphids.orchestrate.config import InstantiatedRun, ResolvedConfig
from graphids.orchestrate.instantiate import build_run

log = get_logger(__name__)


def build(resolved: ResolvedConfig) -> InstantiatedRun:
    """Instantiate trainer + model + datamodule from a resolved config.

    GPU state is reset first so a prior stage's VRAM / compiled kernels
    don't leak into this one.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    torch.compiler.reset()
    return build_run(resolved.rendered, validated=resolved.validated)


def train(
    artifacts: InstantiatedRun,
    resolved: ResolvedConfig,
    *,
    resume_from: str | None = None,
) -> Path | None:
    """Fit the model and return the canonical checkpoint path.

    Touches the train phase marker on success. Caller is expected to
    have wired OTel file exporters for this stage's run_dir already.
    """
    stage_name = resolved.stage_name
    run_dir = resolved.run_dir
    ckpt_file = resolved.ckpt_file
    log.info("stage_train", stage=stage_name, run_dir=str(run_dir) if run_dir else "")
    artifacts.trainer.fit(
        artifacts.model,
        datamodule=artifacts.datamodule,
        ckpt_path=resume_from,
    )
    if run_dir is not None:
        touch_marker(run_dir / PHASE_MARKERS["train"])
    log.info(
        "stage_train_complete",
        stage=stage_name,
        ckpt=str(ckpt_file) if ckpt_file else "",
    )
    return ckpt_file


def evaluate(
    artifacts: InstantiatedRun,
    resolved: ResolvedConfig,
) -> dict[str, Any]:
    """Run the test phase and return metrics. Lenient on failure.

    Touches the ``test`` phase marker on success and the ``complete``
    marker unconditionally (so the chain's resume check sees the stage
    as finished even if test itself crashed).
    """
    stage_name = resolved.stage_name
    run_dir = resolved.run_dir
    ckpt_file = resolved.ckpt_file
    try:
        log.info("stage_test", stage=stage_name)
        metrics = artifacts.trainer.test(
            artifacts.model,
            datamodule=artifacts.datamodule,
            ckpt_path=str(ckpt_file) if ckpt_file is not None else None,
        )
        if run_dir is not None:
            touch_marker(run_dir / PHASE_MARKERS["test"])
        result: dict[str, Any] = metrics or {}
    except Exception as exc:
        log.warning("stage_test_failed", stage=stage_name, error=str(exc))
        result = {}
    if run_dir is not None:
        touch_marker(run_dir / COMPLETE_MARKER)
    log.info("stage_complete", stage=stage_name)
    return result
