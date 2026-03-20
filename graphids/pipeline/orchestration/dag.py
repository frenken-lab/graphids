"""Pipeline DAG: topology + platform-agnostic execution via concurrent.futures.

build_dag_topology() defines the DAG (moved from dagster_defs.py).
run_dag() executes it through any concurrent.futures.Executor.
"""

from __future__ import annotations

import graphlib
from concurrent.futures import Future
from dataclasses import dataclass
from datetime import timedelta
from typing import Callable

import structlog
import yaml

from graphids.config import CONFIG_DIR, STAGE_DEPENDENCIES, STAGE_MODEL_MAP, resolve
from graphids.pipeline.executor import execute_stage

from .job import ResourceSpec

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Resource profiles (loaded from resources.yaml)
# ---------------------------------------------------------------------------

_RESOURCES_YAML = CONFIG_DIR / "resources.yaml"


def _load_resources_yaml() -> dict:
    return yaml.safe_load(_RESOURCES_YAML.read_text())


def _parse_resource_profiles(raw: dict) -> dict[tuple[str, str, str], ResourceSpec]:
    profiles: dict[tuple[str, str, str], ResourceSpec] = {}
    for model, scales in raw.get("resource_profiles", {}).items():
        for scale, stages in scales.items():
            for stage, res in stages.items():
                profiles[(model, scale, stage)] = ResourceSpec.from_yaml(res)
    return profiles


_raw_resources = _load_resources_yaml()
RESOURCE_PROFILES = _parse_resource_profiles(_raw_resources)
FAILURE_REACTIONS: dict[str, dict] = _raw_resources.get("failure_reactions", {})
del _raw_resources


def get_resources(model: str, scale: str, stage: str) -> ResourceSpec:
    """Look up resource profile for a (model, scale, stage) tuple."""
    key = (model, scale, stage)
    if key not in RESOURCE_PROFILES:
        available = sorted(RESOURCE_PROFILES.keys())
        raise KeyError(
            f"No resource profile for {key}. "
            f"Add an entry to config/resources.yaml. Available: {available}"
        )
    return RESOURCE_PROFILES[key]


def scale_resources(resources: ResourceSpec, failure_reason: str) -> ResourceSpec:
    """Apply failure reaction scaling. OOM -> 2x mem, TIMEOUT -> 1.5x time."""
    reaction = FAILURE_REACTIONS.get(failure_reason, {})
    if not reaction:
        return resources
    updates: dict = {}
    if "scale_mem" in reaction:
        updates["memory_gb"] = int(resources.memory_gb * reaction["scale_mem"])
    if "scale_time" in reaction:
        total_secs = resources.walltime.total_seconds()
        updates["walltime"] = timedelta(seconds=int(total_secs * reaction["scale_time"]))
    return resources.model_copy(update=updates) if updates else resources


class SlurmJobFailed(Exception):
    """Raised when a SLURM job reaches a terminal failure state."""

    def __init__(self, reason: str, node: str | None = None,
                 ckpt_path: str | None = None, metadata: dict | None = None):
        self.reason = reason
        self.node = node
        self.ckpt_path = ckpt_path
        self.metadata = metadata or {}
        super().__init__(f"SLURM job failed: {reason} (node={node})")


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
