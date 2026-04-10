"""Single-stage primitives — build, train, evaluate, run_stage.

Atomic layer between ``ResolvedConfig`` (pure data) and the chain
driver. Each primitive owns one verb at one level; composition happens
in ``run_stage`` only.

``analyze`` is *not* called here — per the orchestrate refactor plan
it's a pipeline-level concern (one call over all checkpoints after the
chain finishes), not a per-stage side effect.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from graphids._otel import get_logger
from graphids.orchestrate._setup import touch_marker

if TYPE_CHECKING:
    from graphids.orchestrate.resolve import ResolvedConfig

log = get_logger(__name__)


@dataclass(frozen=True)
class StageResult:
    """Outcome of a single stage run."""

    ckpt: Path
    metrics: dict[str, Any]


# ---------------------------------------------------------------------------
# Primitives — each owns one verb at one level
# ---------------------------------------------------------------------------


def build(resolved: ResolvedConfig) -> Any:
    """Instantiate trainer + model + datamodule from a ``ResolvedConfig``.

    GPU state is reset first so a prior stage's VRAM / compiled kernels
    don't leak into this one. Returns an ``InstantiatedRun`` from
    ``graphids.instantiate``.
    """
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    torch.compiler.reset()

    from graphids.instantiate import instantiate

    return instantiate(resolved.rendered, validated=resolved.validated)


def train(artifacts: Any, resolved: ResolvedConfig) -> Path:
    """Fit the model and return the checkpoint path.

    Wires the run_dir's file-backed OTel exporters before starting, so
    traces + metrics land alongside the checkpoint. Touches the
    ``train`` phase marker on success.
    """
    from graphids._otel import wire_file_exporters
    from graphids.config.constants import PHASE_MARKERS

    run_dir = Path(str(resolved.paths.run_dir))
    ckpt_path = Path(str(resolved.paths.ckpt_file))
    stage = resolved.paths.stage

    wire_file_exporters(run_dir)
    log.info("stage_train", stage=stage, run_dir=str(run_dir))
    artifacts.trainer.fit(artifacts.model, datamodule=artifacts.datamodule)
    touch_marker(run_dir / PHASE_MARKERS["train"])
    log.info("stage_train_complete", stage=stage, ckpt=str(ckpt_path))
    return ckpt_path


def evaluate(artifacts: Any, resolved: ResolvedConfig, ckpt: Path) -> dict[str, Any]:
    """Run the test phase and return metrics. Lenient on failure.

    Touches the ``test`` phase marker on success. Swallows exceptions
    and returns an empty dict on failure — test failures must not kill
    the chain because downstream stages may still be resumable.
    """
    from graphids.config.constants import PHASE_MARKERS

    run_dir = Path(str(resolved.paths.run_dir))
    stage = resolved.paths.stage
    try:
        log.info("stage_test", stage=stage)
        metrics = artifacts.trainer.test(
            artifacts.model, datamodule=artifacts.datamodule, ckpt_path=str(ckpt),
        )
        touch_marker(run_dir / PHASE_MARKERS["test"])
        return metrics or {}
    except Exception as exc:
        log.warning("stage_test_failed", stage=stage, error=str(exc))
        return {}


# ---------------------------------------------------------------------------
# Composition — driver for a single stage
# ---------------------------------------------------------------------------


def run_stage(resolved: ResolvedConfig, *, force_retrain: bool = False) -> StageResult:
    """Single-stage driver: ``build → train → evaluate``.

    Skipped if ``force_retrain=False`` and the stage's checkpoint +
    ``complete`` marker already exist — returns the cached checkpoint
    path with empty metrics. Analyze is deliberately *not* called here.
    """
    ckpt_file = Path(str(resolved.paths.ckpt_file))
    run_dir = Path(str(resolved.paths.run_dir))
    stage = resolved.paths.stage

    if (
        not force_retrain
        and ckpt_file.exists()
        and resolved.paths.complete_marker.exists()
    ):
        log.info("stage_skip_complete", stage=stage, run_dir=str(run_dir))
        return StageResult(ckpt=ckpt_file, metrics={})

    artifacts = build(resolved)
    ckpt = train(artifacts, resolved)
    metrics = evaluate(artifacts, resolved, ckpt)
    touch_marker(resolved.paths.complete_marker)
    log.info("stage_complete", stage=stage)
    return StageResult(ckpt=ckpt, metrics=metrics)
