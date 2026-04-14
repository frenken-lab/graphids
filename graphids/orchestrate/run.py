"""Pipeline driver — Layer 5 of the orchestrate stack.

``run_pipeline`` loops over the planner's ``StageConfig`` list,
resolves each, then runs ``build → train → evaluate`` on the resolved
config. Upstream checkpoints flow through via the asset-name graph
from the planner. The driver runs in-process inside whatever SLURM
allocation ``submit.sh`` hands it — no cross-node plumbing.

Analysis is intentionally out of scope: run ``python -m graphids analyze
--ckpt-path <path>`` after the pipeline. Folding it in turned a lenient
failure mode into a silent "stage marked done but artifacts missing"
trap; decoupling gives researchers explicit control over when analyzers
run (and lets them re-run one without re-training anything).

Resume semantics: a stage is skipped when its ``best_model.ckpt`` is on
disk. That file is the last artifact any successful training writes, so
its presence means "training finished at least one best epoch and saved
it." A mid-epoch crash leaves only ``last.ckpt`` (if that) — no skip,
and the trainer's own resume-from-last path takes over on the next run.
"""

from __future__ import annotations

import os
from collections.abc import Callable

from graphids._otel import get_logger
from graphids.orchestrate.config import PipelineConfig, PipelineResult, StageConfig

log = get_logger(__name__)


def run_pipeline(config: PipelineConfig) -> PipelineResult:
    """Run train+evaluate for each stage in the configured chain."""
    from graphids._otel import wire_file_exporters
    from graphids._spawn import ensure_spawn
    from graphids.config.constants import LAKE_ROOT
    from graphids.orchestrate.planning import build_pipeline_stages
    from graphids.orchestrate.resolve import resolve_config
    from graphids.orchestrate.stage import build, evaluate, train

    ensure_spawn()
    stages = build_pipeline_stages(config)
    lake_root = config.lake_root or LAKE_ROOT
    user = os.environ.get("USER", "unknown")

    checkpoints: dict[str, str] = {}
    stage_to_asset = {cfg.stage: cfg.asset_name for cfg in stages}

    def train_and_eval(cfg: StageConfig, upstream_ckpts: dict[str, str]) -> str:
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

        if ckpt_file.exists():
            log.info("stage_skip_complete", stage=cfg.stage, run_dir=str(run_dir))
            return str(ckpt_file)

        wire_file_exporters(run_dir)
        artifacts = build(resolved)
        train(artifacts, resolved)
        evaluate(artifacts, resolved)
        return str(ckpt_file)

    for cfg in stages:
        upstream = {n: checkpoints[n] for n in cfg.upstream_asset_names if n in checkpoints}
        ckpt = _retry(
            lambda: train_and_eval(cfg, upstream),
            stage=cfg.stage,
            max_retries=config.max_retries,
        )
        checkpoints[cfg.asset_name] = ckpt

    return PipelineResult(checkpoints=checkpoints, stage_to_asset=stage_to_asset)


def _retry(
    fn: Callable[[], str],
    *,
    stage: str,
    max_retries: int,
) -> str:
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
