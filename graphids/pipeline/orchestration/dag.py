"""Pipeline DAG: topology, resource profiles, SLURM execution.

build_dag_topology() defines the DAG.
run_dag() topologically sorts and submits via submitit.
"""

from __future__ import annotations

import graphlib
from dataclasses import dataclass
from enum import Enum

import structlog
import submitit
import yaml

from graphids.config import CONFIG_DIR, SLURM_ACCOUNT, STAGE_DEPENDENCIES, STAGE_MODEL_MAP, resolve
from graphids.pipeline.executor import execute_stage

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Resource profiles (loaded from resources.yaml as plain dicts)
# ---------------------------------------------------------------------------

_RESOURCES_YAML = CONFIG_DIR / "resources.yaml"


def _load_resources() -> tuple[dict[tuple[str, str, str], dict], dict]:
    raw = yaml.safe_load(_RESOURCES_YAML.read_text())
    profiles: dict[tuple[str, str, str], dict] = {}
    for model, scales in raw.get("resource_profiles", {}).items():
        for scale, stages in scales.items():
            for stage, res in stages.items():
                profiles[(model, scale, stage)] = _normalize(res)
    return profiles, raw.get("failure_reactions", {})


def _normalize(res: dict) -> dict:
    """Normalize YAML resource entry to submitit-ready params."""
    mem_gb = res.get("memory_gb")
    if mem_gb is None and "mem" in res:
        mem_str = res["mem"]
        if mem_str.upper().endswith("G"):
            mem_gb = int(mem_str[:-1])
        elif mem_str.upper().endswith("M"):
            mem_gb = max(1, int(mem_str[:-1]) // 1024)
        else:
            mem_gb = int(mem_str)
    return {
        "gpus": res.get("gpus", 0),
        "cpus": res.get("cpus", 4),
        "memory_gb": mem_gb or 20,
        "walltime_min": _parse_walltime_min(res.get("walltime", "3:00:00")),
        "partition": res.get("partition", "cpu"),
        "exclude_nodes": res.get("exclude_nodes", ""),
    }


def _parse_walltime_min(wt) -> int:
    if isinstance(wt, (int, float)):
        return int(wt)
    parts = str(wt).split(":")
    if len(parts) == 3:
        return int(parts[0]) * 60 + int(parts[1])
    if len(parts) == 2:
        return int(parts[0])
    return 180


RESOURCE_PROFILES, FAILURE_REACTIONS = _load_resources()


def get_resources(model: str, scale: str, stage: str) -> dict:
    """Look up resource profile for a (model, scale, stage) tuple."""
    key = (model, scale, stage)
    if key not in RESOURCE_PROFILES:
        raise KeyError(
            f"No resource profile for {key}. "
            f"Add an entry to config/resources.yaml. Available: {sorted(RESOURCE_PROFILES)}"
        )
    return RESOURCE_PROFILES[key]


def scale_resources(resources: dict, failure_reason: str) -> dict:
    """Apply failure reaction scaling. OOM -> 2x mem, TIMEOUT -> 1.5x time."""
    reaction = FAILURE_REACTIONS.get(failure_reason, {})
    if not reaction:
        return resources
    scaled = dict(resources)
    if "scale_mem" in reaction:
        scaled["memory_gb"] = int(resources["memory_gb"] * reaction["scale_mem"])
    if "scale_time" in reaction:
        scaled["walltime_min"] = int(resources["walltime_min"] * reaction["scale_time"])
    return scaled


# ---------------------------------------------------------------------------
# SLURM executor
# ---------------------------------------------------------------------------


class FailureCategory(Enum):
    OOM = "oom"
    TIMEOUT = "timeout"
    INFRA = "infra"
    APPLICATION = "application"


_SLURM_FAILURE_MAP = {
    "OUT_OF_MEMORY": FailureCategory.OOM,
    "TIMEOUT": FailureCategory.TIMEOUT,
    "NODE_FAIL": FailureCategory.INFRA,
    "PREEMPTED": FailureCategory.INFRA,
}


def classify_failure(job: submitit.Job) -> FailureCategory:
    """Map SLURM job state to a failure category."""
    state = job.get_info().get("State", "FAILED").split()[0]
    return _SLURM_FAILURE_MAP.get(state, FailureCategory.APPLICATION)


def make_slurm_executor(
    resources: dict,
    dep_futures: list | None = None,
    *,
    log_folder: str = "slurm_logs/%j",
) -> submitit.SlurmExecutor:
    """Create a submitit SlurmExecutor from a resource dict."""
    executor = submitit.SlurmExecutor(folder=log_folder)

    dep_str = None
    if dep_futures:
        dep_ids = [str(f.job_id) for f in dep_futures]
        dep_str = f"afterok:{':'.join(dep_ids)}"

    executor.update_parameters(
        mem_gb=resources["memory_gb"],
        gpus_per_node=resources["gpus"],
        cpus_per_task=resources["cpus"],
        timeout_min=resources["walltime_min"],
        partition=resources["partition"],
        account=SLURM_ACCOUNT,
        setup=["source scripts/slurm/_preamble.sh"],
        dependency=dep_str,
        exclude=resources.get("exclude_nodes") or None,
        signal_delay_s=180,
    )
    return executor


# ---------------------------------------------------------------------------
# DAG topology + execution
# ---------------------------------------------------------------------------


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
    dag: dict[str, DagNode],
    dataset: str,
    seeds: list[int],
    *,
    dry_run: bool = False,
) -> dict[str, object]:
    """Execute pipeline DAG via submitit SLURM submission."""
    topo_order = list(graphlib.TopologicalSorter(
        {name: set(node.deps) for name, node in dag.items()}
    ).static_order())

    all_futures: dict[str, object] = {}
    for seed in seeds:
        futures: dict[str, object] = {}
        for name in topo_order:
            node = dag[name]
            dep_futs = [futures[d] for d in node.deps if d in futures]
            resources = get_resources(node.resource_model, node.scale, node.stage)
            cfg = resolve(node.cli_model, node.scale, dataset=dataset, seed=seed)

            if dry_run:
                log.info("dry_run", asset=name, deps=[str(d) for d in node.deps])
                continue

            executor = make_slurm_executor(resources, dep_futures=dep_futs)
            futures[name] = executor.submit(execute_stage, cfg, node.stage)
            all_futures[f"{name}__seed{seed}"] = futures[name]

    return all_futures
