"""Pipeline driver — Layer 5 of the orchestrate stack.

``run_pipeline`` loops over the planner's ``StageConfig`` list,
resolves each, then runs ``build → train → evaluate → analyze`` on
the resolved config. Upstream checkpoints flow through via the
asset-name graph from the planner. The driver runs in-process inside
whatever SLURM allocation ``submit.sh`` hands it — no cross-node
plumbing.

Retry semantics: only the ``resolve → build → train → evaluate``
segment is retried on failure. ``analyze`` is lenient by design — its
failures log a warning but don't trigger a full retrain (analyze can
be re-run standalone from the surviving checkpoint).
"""

from __future__ import annotations

import os
from typing import Callable

from graphids._otel import get_logger
from graphids.orchestrate.config import PipelineConfig, PipelineResult, StageConfig

log = get_logger(__name__)


def run_pipeline(config: PipelineConfig) -> PipelineResult:
    """Run the pipeline end-to-end in the current process.

    Loops over stages, passing upstream asset checkpoints through to
    each stage's ``resolve_config`` call. Retries the train+eval
    segment of each stage up to ``config.max_retries`` times on
    exception; analyze always runs once and swallows failures.
    """
    from graphids._fs import touch_marker
    from graphids._otel import wire_file_exporters
    from graphids._spawn import ensure_spawn
    from graphids.config.constants import COMPLETE_MARKER, LAKE_ROOT, PHASE_MARKERS
    from graphids.core.analysis.runner import (
        ANALYZABLE_MODEL_TYPES,
        analysis_spec_for,
        run_single_analysis,
    )
    from graphids.orchestrate.planning import build_pipeline_stages
    from graphids.orchestrate.resolve import resolve_config
    from graphids.orchestrate.stage import build, evaluate, train

    ensure_spawn()
    stages = build_pipeline_stages(config)
    lake_root = config.lake_root or LAKE_ROOT
    user = os.environ.get("USER", "unknown")

    checkpoints: dict[str, str] = {}
    analyzed: list[str] = []
    stage_to_asset = {cfg.stage: cfg.asset_name for cfg in stages}

    def train_and_eval(cfg: StageConfig, upstream_ckpts: dict[str, str]) -> tuple[str, bool]:
        """Resolve → skip-check → build → train → evaluate → analyze. Called per retry attempt."""
        resolved = resolve_config(
            cfg,
            lake_root=lake_root,
            user=user,
            dataset=config.dataset,
            seed=config.seed,
            upstream_ckpts=upstream_ckpts,
        )
        assert resolved.run_dir is not None and resolved.ckpt_file is not None
        run_dir, ckpt_file = resolved.run_dir, resolved.ckpt_file

        if ckpt_file.exists() and (run_dir / COMPLETE_MARKER).exists():
            log.info("stage_skip_complete", stage=cfg.stage, run_dir=str(run_dir))
            return str(ckpt_file), (run_dir / PHASE_MARKERS["analyze"]).exists()

        wire_file_exporters(run_dir)
        artifacts = build(resolved)
        train(artifacts, resolved)
        evaluate(artifacts, resolved)

        did_analyze = False
        if cfg.model_type in ANALYZABLE_MODEL_TYPES:
            try:
                log.info("stage_analyze", stage=cfg.stage, model_type=cfg.model_type)
                run_single_analysis(analysis_spec_for(
                    ckpt_file, dataset=config.dataset,
                    model_type=cfg.model_type, seed=config.seed,
                    upstream_ckpts=upstream_ckpts,
                    upstream_families=cfg.upstream_model_families,
                ))
                touch_marker(run_dir / PHASE_MARKERS["analyze"])
                did_analyze = True
            except Exception as exc:
                log.warning("stage_analyze_failed", stage=cfg.stage, error=str(exc))

        return str(ckpt_file), did_analyze

    for cfg in stages:
        upstream = {n: checkpoints[n] for n in cfg.upstream_asset_names if n in checkpoints}
        ckpt, did_analyze = _retry(
            lambda: train_and_eval(cfg, upstream),
            stage=cfg.stage,
            max_retries=config.max_retries,
        )
        checkpoints[cfg.asset_name] = ckpt
        if did_analyze:
            analyzed.append(cfg.asset_name)

    return PipelineResult(
        checkpoints=checkpoints,
        analyzed_assets=analyzed,
        stage_to_asset=stage_to_asset,
    )


def _retry(
    fn: Callable[[], tuple[str, bool]],
    *,
    stage: str,
    max_retries: int,
) -> tuple[str, bool]:
    """Run ``fn`` up to ``max_retries + 1`` times; raise on final failure."""
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            log.error("stage_failed", stage=stage, attempt=attempt, error=str(exc))
    raise RuntimeError(
        f"{stage} failed after {max_retries + 1} attempts",
    ) from last_exc
