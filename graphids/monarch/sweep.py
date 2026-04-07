"""Sweep controller — bridges the planner to the Monarch execution model.

Reads a recipe, expands it via ``enumerate_assets``, decomposes the
resulting DAG into maximal root-to-leaf chains, and crosses with
datasets x seeds to produce ``ChainSpec`` objects that each map to
one Monarch SLURM allocation.

Shared upstream stages (e.g. one autoencoder feeding multiple supervised
configs) appear in multiple chains but are idempotent — trained once,
skipped by subsequent chains via ``.complete`` marker checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from graphids.log import get_logger
from graphids.orchestrate.planning import StageConfig

log = get_logger(__name__)


@dataclass
class ChainSpec:
    """One maximal DAG path — becomes one Monarch allocation."""

    chain_id: str
    stages: list[StageConfig]
    dataset: str
    seed: int


def plan_chains(
    recipe_path: str | Path,
    datasets: list[str],
    seeds: list[int],
) -> list[ChainSpec]:
    """Expand a recipe into Monarch chain specs.

    1. Render the recipe jsonnet → raw dict
    2. ``expand_recipe_configs`` → expanded recipe with configs dict
    3. ``enumerate_assets`` → deduplicated ``StageConfig`` list
    4. ``decompose_dag`` → maximal root-to-leaf chains
    5. Cross product with datasets × seeds → ``ChainSpec`` list
    """
    from graphids.config.jsonnet import render
    from graphids.config.topology import PIPELINE_TOPOLOGY
    from graphids.orchestrate.planning import enumerate_assets, expand_recipe_configs

    raw = render(Path(recipe_path))
    expanded = expand_recipe_configs(raw)
    configs = enumerate_assets(PIPELINE_TOPOLOGY, expanded)

    log.info("sweep_plan", num_assets=len(configs), datasets=datasets, seeds=seeds)

    chains = decompose_dag(configs)

    specs: list[ChainSpec] = []
    for idx, chain in enumerate(chains):
        leaf = chain[-1]
        label = f"{leaf.stage}_{leaf.identity or leaf.asset_name}"
        for ds in datasets:
            for seed in seeds:
                specs.append(
                    ChainSpec(
                        chain_id=f"chain_{idx}_{label}_{ds}_s{seed}",
                        stages=chain,
                        dataset=ds,
                        seed=seed,
                    )
                )

    log.info("sweep_chains", num_chains=len(specs), num_unique_dags=len(chains))
    return specs


def decompose_dag(configs: list[StageConfig]) -> list[list[StageConfig]]:
    """Decompose a DAG of StageConfigs into maximal root-to-leaf chains.

    Algorithm:
    - Build lookup and adjacency from ``upstream_asset_names``
    - Identify leaves (nodes with no downstream dependents)
    - For each leaf, walk upstream to root → one chain (topologically ordered)
    - Chains sharing upstream nodes produce duplicates; idempotency
      ensures the shared node is only executed once.

    Returns a list of chains, each a list of StageConfigs in execution order
    (root first, leaf last).
    """
    by_name: dict[str, StageConfig] = {c.asset_name: c for c in configs}
    has_downstream: set[str] = set()
    for cfg in configs:
        for upstream in cfg.upstream_asset_names:
            has_downstream.add(upstream)

    leaves = [c for c in configs if c.asset_name not in has_downstream]

    # If no leaves found (isolated nodes), each config is its own chain
    if not leaves:
        return [[c] for c in configs]

    chains: list[list[StageConfig]] = []
    for leaf in leaves:
        chain = _walk_to_root(leaf, by_name)
        chains.append(chain)

    return chains


def _walk_to_root(
    leaf: StageConfig,
    by_name: dict[str, StageConfig],
) -> list[StageConfig]:
    """Walk upstream from a leaf to all roots, returning topologically ordered chain."""
    visited: set[str] = set()
    order: list[StageConfig] = []

    def _visit(cfg: StageConfig) -> None:
        if cfg.asset_name in visited:
            return
        visited.add(cfg.asset_name)
        for upstream_name in cfg.upstream_asset_names:
            upstream = by_name.get(upstream_name)
            if upstream is not None:
                _visit(upstream)
        order.append(cfg)

    _visit(leaf)
    return order


def build_single_recipe(
    stages: list[str],
    scale: str,
    conv_type: str,
    variational: bool,
    loss_fn: str,
    fusion_method: str,
    trainer_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build a synthetic expanded recipe for a single pipeline config.

    Produces the same shape as ``expand_recipe_configs`` output so
    ``enumerate_assets`` can consume it. Used by ``monarch-run`` to
    go through the planner instead of duplicating its logic.
    """
    defaults: dict[str, Any] = {
        "stages": list(stages),
        "scale": scale,
        "conv_type": conv_type,
        "variational": variational,
        "loss_fn": loss_fn,
        "fusion_method": fusion_method,
    }

    return {
        "defaults": defaults,
        "configs": {"default": {}},
        "sweep": {},
        "trainer_overrides": dict(trainer_overrides or {}),
        "stage_overrides": {},
        "resource_overrides": {},
    }
