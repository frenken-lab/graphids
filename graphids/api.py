"""Programmatic API facade for KD-GAT pipeline.

Thin wrapper over the same resolve→dispatch logic as the CLI, designed
for notebook usage and Dagster integration (no argparse).

Usage:
    from graphids.api import train, evaluate, orchestrate

    # Train a single stage
    ckpt = train("vgae", "large", "hcrl_sa")

    # Evaluate all models
    metrics = evaluate("hcrl_sa")

    # Submit full pipeline to SLURM
    job_ids = orchestrate("hcrl_sa", seeds=[42, 123, 456])
"""

from __future__ import annotations

from pathlib import Path

from graphids.config import checkpoint_path, resolve
from graphids.config.constants import DEFAULT_DATASET


def train(
    model_type: str,
    scale: str,
    dataset: str = DEFAULT_DATASET,
    stage: str = "autoencoder",
    seed: int = 42,
    **overrides,
) -> Path:
    """Train a model. Returns checkpoint path."""
    from graphids.pipeline.stages import STAGE_FNS

    cfg = resolve(model_type, scale, dataset=dataset, seed=seed, **overrides)
    STAGE_FNS[stage](cfg)
    return checkpoint_path(cfg, stage)


def evaluate(
    dataset: str = DEFAULT_DATASET,
    scale: str = "large",
    seed: int = 42,
    **overrides,
) -> dict:
    """Evaluate all trained models. Returns stage result dict."""
    from graphids.pipeline.stages import STAGE_FNS

    cfg = resolve("vgae", scale, dataset=dataset, seed=seed, **overrides)
    return STAGE_FNS["evaluation"](cfg)


def orchestrate(
    dataset: str = DEFAULT_DATASET,
    seeds: list[int] | None = None,
    dry_run: bool = False,
) -> dict[str, str]:
    """Fire-and-forget pipeline submission. Returns {asset: job_id}."""
    from graphids.pipeline.orchestration.dagster_defs import fire_and_forget

    return fire_and_forget(dataset=dataset, seeds=seeds, dry_run=dry_run)
