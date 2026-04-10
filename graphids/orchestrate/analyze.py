"""Pipeline-level analyze — runs *once* after ``run_chain`` returns.

Per the orchestrate refactor design (plans/kd-gat-orchestrate-refactor.md,
decision #2), analysis is not a per-stage side effect: it operates over
the full dict of trained checkpoints. Internal iteration (one at a
time or batched) is this module's concern.

Two public functions:

``analyze(actor, stages, chain, …)``
    Pipeline-level driver. Iterates over analyzable stages and
    dispatches each one to the actor's ``analyze_stage`` endpoint so
    the work runs on the compute node where torch + dataset cache are
    already warm.

``run_single_analysis(spec)``
    Single-checkpoint implementation. Builds the ``Analyzer``, runs
    it, and writes an ``analysis_manifest.json`` sidecar. Called only
    by ``actors.PipelineActor.analyze_stage``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from graphids._otel import get_logger

if TYPE_CHECKING:
    from graphids.core.analysis.schemas import AnalysisSpec
    from graphids.orchestrate.chain import ChainResult
    from graphids.orchestrate.planning import StageConfig

log = get_logger(__name__)

# model_types that actually have analyzers; fusion has its own path
_ANALYZABLE_MODEL_TYPES = frozenset({"vgae", "dgi", "gat"})

ANALYSIS_MANIFEST_NAME = "analysis_manifest.json"


def analyze(
    actor,
    stages: list["StageConfig"],
    chain: "ChainResult",
    *,
    dataset: str,
    seed: int,
) -> list[str]:
    """Run analysis on every analyzable stage in ``chain``.

    Iterates over stages, filtering to those whose ``model_type`` is
    analyzable (vgae/dgi/gat) and whose checkpoint is present in the
    chain result. Each call dispatches to the actor's
    ``analyze_stage`` endpoint. Failures are lenient and logged.

    Returns the list of ``asset_name``s that were successfully analyzed.
    """
    analyzed: list[str] = []
    for cfg in stages:
        if cfg.model_type not in _ANALYZABLE_MODEL_TYPES:
            continue
        ckpt = chain.checkpoints.get(cfg.asset_name)
        if not ckpt:
            continue
        try:
            actor.analyze_stage.call_one(
                stage_config=cfg.model_dump(),
                dataset=dataset,
                seed=seed,
                ckpt_path=ckpt,
            ).get()
            analyzed.append(cfg.asset_name)
        except Exception as exc:
            log.warning("analyze_failed", stage=cfg.stage, error=str(exc))
    return analyzed


def run_single_analysis(spec: "AnalysisSpec") -> None:
    """Run the analyzer for one checkpoint and write a manifest sidecar.

    Called by the Monarch actor's ``analyze_stage`` endpoint.
    """
    from graphids.core.analysis.analyzer import Analyzer
    from graphids.core.analysis.schemas import expected_outputs

    Analyzer(**spec.model_dump(exclude={"metadata"})).run()

    output_dir = Path(spec.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    expected = expected_outputs(spec)
    existing = [name for name in expected if (output_dir / name).exists()]
    manifest = {
        "contract": spec.CONTRACT_NAME,
        "version": spec.CONTRACT_VERSION,
        "dataset": spec.dataset,
        "checkpoint_path": spec.ckpt_path,
        "output_dir": str(output_dir),
        "expected_outputs": list(expected),
        "existing_outputs": existing,
    }
    (output_dir / ANALYSIS_MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
