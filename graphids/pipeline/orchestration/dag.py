"""Pipeline DAG: topology + platform-agnostic execution via concurrent.futures.

build_dag_topology() defines the DAG (moved from dagster_defs.py).
run_dag() executes it through any concurrent.futures.Executor.
"""

from __future__ import annotations

import graphlib
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Callable

import structlog

from graphids.config import STAGE_DEPENDENCIES, STAGE_MODEL_MAP, resolve
from graphids.pipeline.executor import execute_stage

from .job import ResourceSpec
from .slurm_primitives import FAILURE_REACTIONS, get_resources, scale_resources, SlurmJobFailed

log = structlog.get_logger()


@dataclass(frozen=True)
class DagNode:
    stage: str
    cli_model: str
    resource_model: str
    scale: str
    auxiliaries: str
    deps: frozenset[str]


def _asset_name(resource_model: str, scale: str, stage: str, aux: str = "") -> str:
    name = f"{resource_model}_{scale}_{stage}"
    return f"{name}_{aux}" if aux else name


def build_dag_topology() -> dict[str, DagNode]:
    """Build pipeline DAG from PipelineConfig.variants + STAGE_DEPENDENCIES."""
    cfg = resolve("vgae", "large")
    nodes: dict[str, DagNode] = {
        "preprocess": DagNode("preprocess", "preprocess", "preprocess", "", "none", frozenset()),
    }

    for variant in cfg.variants:
        aux = variant.auxiliaries if variant.auxiliaries != "none" else ""
        for stage in variant.stages:
            resource_model = STAGE_MODEL_MAP[stage]
            cli_model = "vgae" if stage == "evaluation" else resource_model
            asset_nm = _asset_name(resource_model, variant.scale, stage, aux)

            dep_names: set[str] = set()
            if stage == "autoencoder":
                dep_names.add("preprocess")
            else:
                for dep_model_type, dep_stage in STAGE_DEPENDENCIES.get(stage, []):
                    dep_names.add(_asset_name(dep_model_type, variant.scale, dep_stage, aux))
                if stage == "evaluation" and not dep_names:
                    dep_names.add(_asset_name(STAGE_MODEL_MAP["fusion"], variant.scale, "fusion", aux))

            if variant.needs_teacher and stage in ("autoencoder", "curriculum"):
                teacher_nm = _asset_name(resource_model, "large", stage)
                if teacher_nm != asset_nm:
                    dep_names.add(teacher_nm)

            nodes[asset_nm] = DagNode(stage, cli_model, resource_model, variant.scale,
                                      variant.auxiliaries, frozenset(dep_names))
    return nodes


def run_dag(
    executor_factory: Callable[[ResourceSpec, list[Future]], object],
    dag: dict[str, DagNode],
    dataset: str,
    seeds: list[int],
    *,
    dry_run: bool = False,
) -> dict[str, Future]:
    """Execute pipeline DAG through any concurrent.futures.Executor.

    Parameters
    ----------
    executor_factory
        Callable(resources, dep_futures) -> Executor with .submit().
        Each backend decides how to handle dependencies:
        - SLURM: extract .job_id from futures, pass as afterok
        - Local: call .result() to block before submitting
    """
    topo_order = list(graphlib.TopologicalSorter(
        {name: set(node.deps) for name, node in dag.items()}
    ).static_order())

    all_futures: dict[str, Future] = {}
    for seed in seeds:
        futures: dict[str, Future] = {}
        for name in topo_order:
            node = dag[name]
            dep_futures = [futures[d] for d in node.deps if d in futures]
            resources = get_resources(node.resource_model, node.scale, node.stage)
            cfg = resolve(node.cli_model, node.scale, dataset=dataset, seed=seed)

            if dry_run:
                log.info("dry_run", asset=name, deps=[str(d) for d in node.deps])
                continue

            executor = executor_factory(resources, dep_futures)
            futures[name] = executor.submit(execute_stage, cfg, node.stage)
            all_futures[f"{name}__seed{seed}"] = futures[name]

    return all_futures
