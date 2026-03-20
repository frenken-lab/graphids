"""Programmatic API facade for KD-GAT pipeline.

Thin wrapper over execute_stage(), designed for notebook usage.

Usage:
    from graphids.api import train, evaluate, orchestrate

    # Train a single stage (full guarantees: validation, manifest, logging)
    result = train("vgae", "large", "hcrl_sa")

    # Evaluate all models
    result = evaluate("hcrl_sa")

    # Submit full pipeline to SLURM
    job_ids = orchestrate("hcrl_sa", seeds=[42, 123, 456])
"""

from __future__ import annotations

from graphids.config import DEFAULT_DATASET, resolve
from graphids.pipeline.executor import StageResult, execute_stage


def train(
    model_type: str,
    scale: str,
    dataset: str = DEFAULT_DATASET,
    stage: str = "autoencoder",
    seed: int = 42,
    **overrides,
) -> StageResult:
    """Train a model. Returns StageResult with metrics, checkpoint path, manifest."""
    cfg = resolve(model_type, scale, dataset=dataset, seed=seed, **overrides)
    return execute_stage(cfg, stage)


def evaluate(
    dataset: str = DEFAULT_DATASET,
    scale: str = "large",
    seed: int = 42,
    **overrides,
) -> StageResult:
    """Evaluate all trained models. Returns StageResult."""
    cfg = resolve("vgae", scale, dataset=dataset, seed=seed, **overrides)
    return execute_stage(cfg, "evaluation")


def orchestrate(
    dataset: str = DEFAULT_DATASET,
    seeds: list[int] | None = None,
    dry_run: bool = False,
) -> dict:
    """Fire-and-forget pipeline submission. Returns {asset: Future}."""
    from graphids.pipeline.orchestration.dag import build_dag_topology, run_dag
    from graphids.pipeline.orchestration.slurm import make_slurm_executor

    seed_list = seeds or [resolve("vgae", "large").seed]
    dag = build_dag_topology()
    return run_dag(
        executor_factory=lambda r, deps: make_slurm_executor(r, dep_futures=deps),
        dag=dag, dataset=dataset, seeds=seed_list, dry_run=dry_run,
    )
