"""Pipeline controller -- sequences stages via Monarch actor endpoints.

Three entry points:

- ``run_pipeline(PipelineConfig)`` — single explicit pipeline (``monarch-run``).
  Builds StageConfigs via the planner, constructs one chain, delegates to
  ``run_chain``.
- ``run_chain(ChainSpec, MonarchJobSpec)`` — one Monarch allocation for a
  chain of stages. Creates SlurmJob, spawns actor, sequences train/eval
  with retry and idempotency.
- ``run_sweep(SweepConfig)`` — recipe-driven sweep (``monarch-sweep``).
  Plans chains via ``plan_chains``, runs them in parallel via
  ``ThreadPoolExecutor``.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from graphids.config.constants import PIPELINE_DEFAULTS
from graphids.log import get_logger

log = get_logger(__name__)

_D = PIPELINE_DEFAULTS


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """What to run in a single Monarch allocation (monarch-run CLI)."""

    dataset: str = _D.get("dataset", "hcrl_ch")
    seed: int = _D.get("seed", 42)
    scale: str = _D.get("scale", "small")
    lake_root: str = ""
    fusion_method: str = _D.get("fusion_method", "bandit")
    stages: list[str] = field(
        default_factory=lambda: list(_D.get("stages", ["autoencoder", "supervised", "fusion"])),
    )
    conv_type: str = _D.get("conv_type", "gatv2")
    variational: bool = _D.get("variational", True)
    loss_fn: str = _D.get("loss_fn", "focal")
    tla_overrides: dict[str, Any] = field(default_factory=dict)
    max_retries: int = 2


@dataclass
class SweepConfig:
    """Full sweep run configuration (monarch-sweep CLI)."""

    recipe_path: str
    datasets: list[str] = field(default_factory=lambda: [_D.get("dataset", "hcrl_ch")])
    seeds: list[int] = field(default_factory=lambda: [_D.get("seed", 42)])
    lake_root: str = ""
    max_retries: int = 2
    max_concurrent: int = 0  # 0 = all parallel


@dataclass
class ChainResult:
    """Result of executing one chain."""

    ok: bool
    checkpoints: dict[str, str] = field(default_factory=dict)
    error: str = ""


@dataclass
class SweepResult:
    """Aggregate result of a sweep."""

    results: dict[str, ChainResult] = field(default_factory=dict)
    num_chains: int = 0


# ---------------------------------------------------------------------------
# Build StageConfigs from PipelineConfig (single-pipeline mode)
# ---------------------------------------------------------------------------


def build_pipeline_stages(config: PipelineConfig) -> list[Any]:
    """Build StageConfigs for a single pipeline via the planner.

    Constructs a synthetic expanded recipe from ``PipelineConfig`` fields
    and passes it through ``enumerate_assets`` — the same path used by
    recipe-driven sweeps. This ensures identity hashes, override wiring,
    and upstream deps are always computed by the planner, never duplicated.

    Returns StageConfigs in topological order (matching ``config.stages``).
    """
    from graphids.config.topology import PIPELINE_TOPOLOGY
    from graphids.monarch.sweep import build_single_recipe
    from graphids.orchestrate.planning import enumerate_assets

    recipe = build_single_recipe(
        stages=config.stages,
        scale=config.scale,
        conv_type=config.conv_type,
        variational=config.variational,
        loss_fn=config.loss_fn,
        fusion_method=config.fusion_method,
        trainer_overrides=config.tla_overrides,
    )

    configs = enumerate_assets(PIPELINE_TOPOLOGY, recipe)

    # Order by the user's requested stage order
    stage_order = {s: i for i, s in enumerate(config.stages)}
    configs.sort(key=lambda c: stage_order.get(c.stage, 99))

    return configs


# ---------------------------------------------------------------------------
# run_chain — one Monarch allocation for one chain
# ---------------------------------------------------------------------------


def run_chain(
    chain: Any,
    max_retries: int = 2,
    lake_root: str = "",
    tla_overrides: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Run one chain of stages in a single SLURM allocation.

    Creates a Monarch SlurmJob, spawns ``PipelineActor``, and sequences
    train → eval for each stage. Checkpoints thread between stages via
    return values. Idempotent stages are skipped by the actor.

    Returns dict mapping stage name to checkpoint path.
    """
    from monarch.config import configure  # type: ignore[import-not-found]

    from graphids.config.constants import LAKE_ROOT
    from graphids.monarch.job import chain_job_spec, create_slurm_job

    lake_root = lake_root or LAKE_ROOT
    spec = chain_job_spec(
        chain.stages, job_name=f"graphids-{chain.chain_id}", dataset=chain.dataset
    )

    configure(
        enable_log_forwarding=True,
        process_exit_timeout="60s",
        cleanup_timeout="30s",
        mesh_terminate_timeout="30s",
        host_spawn_ready_timeout="120s",
    )

    log.info(
        "chain_start",
        chain_id=chain.chain_id,
        partition=spec.partition,
        time=spec.time,
        mem=spec.mem,
        stages=[s.stage for s in chain.stages],
    )

    job = create_slurm_job(spec)
    state = job.state()
    host_mesh = state.pipeline

    from graphids.monarch._setup import bootstrap_staging
    from graphids.monarch.actors import PipelineActor

    proc_mesh = host_mesh.spawn_procs(
        per_host={"gpus": spec.gpus_per_node},
        bootstrap=lambda: bootstrap_staging(chain.dataset),
    )

    actor = proc_mesh.spawn("pipeline", PipelineActor, lake_root=lake_root)

    # Train all stages (critical path, sequential deps)
    checkpoints: dict[str, str] = {}
    for cfg in chain.stages:
        upstream = _upstream_ckpts(cfg, checkpoints)
        ckpt = _run_with_retry(
            lambda c=cfg, u=upstream: actor.train_stage.call_one(
                stage_config=c.to_dict(),
                dataset=chain.dataset,
                seed=chain.seed,
                upstream_ckpts=u,
            ),
            f"train:{cfg.stage}",
            max_retries,
        )
        checkpoints[cfg.asset_name] = ckpt

    log.info("chain_trains_complete", chain_id=chain.chain_id, checkpoints=checkpoints)

    # Eval all stages (lenient)
    for cfg in chain.stages:
        upstream = _upstream_ckpts(cfg, checkpoints)
        try:
            actor.eval_stage.call_one(
                stage_config=cfg.to_dict(),
                dataset=chain.dataset,
                seed=chain.seed,
                upstream_ckpts=upstream,
            ).get()
        except Exception as exc:
            log.warning("chain_eval_failed", stage=cfg.stage, error=str(exc))

    log.info("chain_complete", chain_id=chain.chain_id)

    try:
        job.kill()
    except Exception:
        log.warning("monarch_job_kill_failed", chain_id=chain.chain_id)

    return {cfg.stage: checkpoints.get(cfg.asset_name, "") for cfg in chain.stages}


def _upstream_ckpts(cfg: Any, checkpoints: dict[str, str]) -> dict[str, str]:
    """Map a StageConfig's upstream_asset_names to checkpoint paths."""
    return {name: checkpoints[name] for name in cfg.upstream_asset_names if name in checkpoints}


# ---------------------------------------------------------------------------
# run_sweep — recipe-driven parallel chains
# ---------------------------------------------------------------------------


def run_sweep(config: SweepConfig) -> SweepResult:
    """Plan and execute all chains for a recipe.

    Expands the recipe into chains via ``plan_chains``, then runs each
    chain in a separate Monarch allocation. Chains run in parallel up
    to ``max_concurrent`` (0 = all at once).
    """
    from graphids.monarch.sweep import plan_chains

    chains = plan_chains(config.recipe_path, config.datasets, config.seeds)
    log.info("sweep_start", num_chains=len(chains))

    max_workers = config.max_concurrent or len(chains) or 1
    results: dict[str, ChainResult] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                run_chain,
                chain,
                max_retries=config.max_retries,
                lake_root=config.lake_root,
            ): chain
            for chain in chains
        }
        for future in as_completed(futures):
            chain = futures[future]
            try:
                checkpoints = future.result()
                results[chain.chain_id] = ChainResult(ok=True, checkpoints=checkpoints)
                log.info("sweep_chain_ok", chain_id=chain.chain_id)
            except Exception as exc:
                results[chain.chain_id] = ChainResult(ok=False, error=str(exc))
                log.error("sweep_chain_failed", chain_id=chain.chain_id, error=str(exc))

    ok_count = sum(1 for r in results.values() if r.ok)
    log.info("sweep_complete", ok=ok_count, failed=len(results) - ok_count)

    return SweepResult(results=results, num_chains=len(chains))


# ---------------------------------------------------------------------------
# run_pipeline — backward-compatible single-pipeline entry point
# ---------------------------------------------------------------------------


def run_pipeline(config: PipelineConfig) -> dict[str, str]:
    """Run a single explicit pipeline in one SLURM allocation.

    Backward-compatible entry point for ``monarch-run``. Builds
    StageConfigs via the planner and delegates to ``run_chain``.
    """
    from graphids.monarch.sweep import ChainSpec

    stages = build_pipeline_stages(config)
    chain = ChainSpec(
        chain_id=f"pipeline_{config.dataset}_s{config.seed}",
        stages=stages,
        dataset=config.dataset,
        seed=config.seed,
    )
    return run_chain(
        chain,
        max_retries=config.max_retries,
        lake_root=config.lake_root,
        tla_overrides=config.tla_overrides,
    )


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------


def _run_with_retry(fn: Any, stage_name: str, max_retries: int) -> str:
    """Run a stage endpoint call with retry on failure.

    Monarch's supervision tree absorbs actor failures but does NOT
    auto-restart. Retry logic lives here in the controller.
    """
    for attempt in range(max_retries + 1):
        try:
            log.info("stage_start", stage=stage_name, attempt=attempt)
            future = fn()
            ckpt_path = future.get()
            log.info("stage_complete", stage=stage_name, ckpt=ckpt_path)
            return ckpt_path
        except Exception as exc:
            log.error(
                "stage_failed",
                stage=stage_name,
                attempt=attempt,
                error=str(exc),
            )
            if attempt >= max_retries:
                raise RuntimeError(
                    f"Stage {stage_name!r} failed after {max_retries + 1} attempts"
                ) from exc
    raise AssertionError("unreachable")  # pragma: no cover
