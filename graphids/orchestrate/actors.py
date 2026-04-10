"""Pipeline actor — thin Monarch endpoint wrapper around stage primitives.

The real work lives in ``graphids.orchestrate.stage``; this module
exists only because Monarch requires its endpoints to live on an
``Actor`` subclass. When the orchestrate refactor finishes and Monarch
can dispatch to free functions via ``call_one``, this file goes away.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from graphids._otel import get_logger
from graphids.orchestrate._setup import ensure_spawn, touch_marker
from graphids.orchestrate.stage import build, evaluate, train

if TYPE_CHECKING:
    from graphids.orchestrate.resolve import ResolvedConfig

log = get_logger(__name__)

try:
    from monarch.actor import Actor, endpoint  # type: ignore[import-not-found]
except ImportError:

    class Actor:  # type: ignore[no-redef]
        pass

    def endpoint(fn):  # type: ignore[no-redef]
        return fn


class PipelineActor(Actor):
    """Runs all pipeline stages in one allocation.

    Dataset reuse is handled by the process-level ``get_or_build``
    cache inside the datamodule — no actor-side dataset state.
    """

    def __init__(self, lake_root: str, user: str = "") -> None:
        ensure_spawn()
        self.lake_root = lake_root
        self.user = user or os.environ.get("USER", "unknown")
        from graphids._otel import init_providers

        init_providers("graphids.monarch")

    # -- resolve ---------------------------------------------------------------

    def _resolve(
        self,
        stage_config: dict[str, Any],
        dataset: str,
        seed: int,
        upstream_ckpts: dict[str, str],
    ) -> "ResolvedConfig":
        from graphids.orchestrate.planning import StageConfig
        from graphids.orchestrate.resolve import ResolvedConfig

        cfg = StageConfig.model_validate(stage_config)
        return ResolvedConfig.resolve(
            cfg,
            lake_root=self.lake_root,
            user=self.user,
            dataset=dataset,
            seed=seed,
            upstream_ckpts=upstream_ckpts,
        )

    # -- stage endpoints -------------------------------------------------------

    @endpoint
    def train_stage(
        self,
        stage_config: dict[str, Any],
        dataset: str,
        seed: int,
        upstream_ckpts: dict[str, str] | None = None,
    ) -> str:
        """Train a single stage. Returns checkpoint path. Idempotent."""
        resolved = self._resolve(stage_config, dataset, seed, upstream_ckpts or {})
        ckpt_file = Path(str(resolved.paths.ckpt_file))

        if ckpt_file.exists() and resolved.paths.complete_marker.exists():
            log.info(
                "stage_skip_complete",
                stage=stage_config.get("stage"),
                run_dir=str(resolved.paths.run_dir),
            )
            return str(ckpt_file)

        artifacts = build(resolved)
        ckpt = train(artifacts, resolved)
        return str(ckpt)

    @endpoint
    def eval_stage(
        self,
        stage_config: dict[str, Any],
        dataset: str,
        seed: int,
        upstream_ckpts: dict[str, str] | None = None,
    ) -> None:
        """Run test + finalize for a completed stage. Lenient.

        Analyze is **not** run here — it's a pipeline-level concern
        dispatched by ``graphids.orchestrate.analyze`` via the
        ``analyze_stage`` endpoint below.
        """
        resolved = self._resolve(stage_config, dataset, seed, upstream_ckpts or {})
        ckpt_file = Path(str(resolved.paths.ckpt_file))

        artifacts = build(resolved)
        evaluate(artifacts, resolved, ckpt_file)

        touch_marker(resolved.paths.complete_marker)
        log.info("stage_eval_complete", stage=stage_config.get("stage"))

    @endpoint
    def analyze_stage(
        self,
        stage_config: dict[str, Any],
        dataset: str,
        seed: int,
        ckpt_path: str,
    ) -> None:
        """Run the analyzer for one stage's checkpoint.

        Dispatched by the pipeline-level ``analyze`` driver once
        ``run_chain`` finishes. Lenient on failure — a bad analyzer
        run shouldn't kill the chain.
        """
        from graphids.config.constants import PHASE_MARKERS
        from graphids.core.analysis.schemas import AnalysisSpec
        from graphids.orchestrate.analyze import run_single_analysis

        resolved = self._resolve(stage_config, dataset, seed, {})
        model_type = stage_config.get("model_type", "")
        ckpt_file = Path(ckpt_path)

        try:
            spec = AnalysisSpec(
                ckpt_path=str(ckpt_file),
                dataset=dataset,
                model_type=model_type,
                output_dir=str(ckpt_file.resolve().parent.parent / "artifacts"),
                seed=seed,
            )
            log.info("stage_analyze", model_type=model_type)
            run_single_analysis(spec)
            touch_marker(Path(str(resolved.paths.run_dir)) / PHASE_MARKERS["analyze"])
        except Exception as exc:
            log.warning(
                "stage_analyze_failed",
                stage=stage_config.get("stage"),
                error=str(exc),
            )

    # -- fault tolerance -------------------------------------------------------

    def __supervise__(self, failure: Any) -> bool:
        """Monarch supervision hook — absorb child mesh failures."""
        report = failure.report() if hasattr(failure, "report") else str(failure)
        log.error("actor_supervision", report=report)
        return True
