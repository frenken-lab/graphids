"""Pipeline execution via Monarch actor endpoints."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from graphids.log import get_logger

log = get_logger(__name__)


def build_pipeline_stages(config: Any) -> list[Any]:
    """PipelineConfig → StageConfigs via the planner. Also used by CLI dry-run."""
    from graphids.orchestrate.planning import enumerate_assets

    recipe = {
        "defaults": config.to_training_run().model_dump(),
        "configs": {"default": {}},
        "sweep": {},
        "trainer_overrides": dict(config.tla_overrides),
        "stage_overrides": {},
        "resource_overrides": {},
    }
    configs = enumerate_assets(recipe)
    stage_order = {s: i for i, s in enumerate(config.stages)}
    configs.sort(key=lambda c: stage_order.get(c.stage, 99))
    return configs


def run_chain(
    chain: Any, max_retries: int = 2, lake_root: str = "", job_spec_override: Any = None
) -> dict[str, str]:
    """Run one chain in a single SLURM allocation. Returns {stage: ckpt_path}."""
    from monarch.config import configure  # type: ignore[import-not-found]

    from graphids.config.constants import LAKE_ROOT
    from graphids.orchestrate._setup import bootstrap_staging
    from graphids.orchestrate.actors import PipelineActor
    from graphids.orchestrate.job import chain_job_spec

    lake_root = lake_root or LAKE_ROOT
    spec = job_spec_override or chain_job_spec(
        chain.stages, job_name=f"graphids-{chain.chain_id}", dataset=chain.dataset
    )
    configure(
        enable_log_forwarding=True,
        process_exit_timeout="60s",
        cleanup_timeout="30s",
        mesh_terminate_timeout="30s",
        host_spawn_ready_timeout="120s",
    )

    job = spec.create_job()
    try:
        proc_mesh = job.state().pipeline.spawn_procs(
            per_host={"gpus": spec.gpus_per_node},
            bootstrap=lambda: bootstrap_staging(chain.dataset),
        )
        actor = proc_mesh.spawn("pipeline", PipelineActor, lake_root=lake_root)

        checkpoints: dict[str, str] = {}
        for cfg in chain.stages:
            upstream = {n: checkpoints[n] for n in cfg.upstream_asset_names if n in checkpoints}
            call = lambda c=cfg, u=upstream: actor.train_stage.call_one(  # noqa: E731
                stage_config=c.model_dump(),
                dataset=chain.dataset,
                seed=chain.seed,
                upstream_ckpts=u,
            )
            for attempt in range(max_retries + 1):
                try:
                    checkpoints[cfg.asset_name] = call().get()
                    break
                except Exception as exc:
                    log.error("stage_failed", stage=cfg.stage, attempt=attempt, error=str(exc))
                    if attempt >= max_retries:
                        raise RuntimeError(
                            f"{cfg.stage} failed after {max_retries + 1} attempts"
                        ) from exc

        for cfg in chain.stages:
            upstream = {n: checkpoints[n] for n in cfg.upstream_asset_names if n in checkpoints}
            try:
                actor.eval_stage.call_one(
                    stage_config=cfg.model_dump(),
                    dataset=chain.dataset,
                    seed=chain.seed,
                    upstream_ckpts=upstream,
                ).get()
            except Exception as exc:
                log.warning("eval_failed", stage=cfg.stage, error=str(exc))

        return {cfg.stage: checkpoints.get(cfg.asset_name, "") for cfg in chain.stages}
    finally:
        try:
            job.kill()
        except Exception:
            pass


def run_sweep(config: Any) -> dict[str, dict[str, str] | str]:
    """Run all recipe chains in parallel. Returns {chain_id: checkpoints | error_str}."""
    from graphids.orchestrate.sweep import plan_chains

    chains = plan_chains(config.recipe_path, config.datasets, config.seeds)
    results: dict[str, dict[str, str] | str] = {}

    with ThreadPoolExecutor(max_workers=config.max_concurrent or len(chains) or 1) as pool:
        futures = {
            pool.submit(run_chain, c, max_retries=config.max_retries, lake_root=config.lake_root): c
            for c in chains
        }
        for future in as_completed(futures):
            chain = futures[future]
            try:
                results[chain.chain_id] = future.result()
            except Exception as exc:
                results[chain.chain_id] = str(exc)

    return results
