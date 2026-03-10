"""Domain-aware pipeline planner for KD-GAT.

Builds a list of JobSpec objects from (datasets × seeds × variants).
Dependencies use opaque UUIDs — no string key encoding.
Uses graphlib.TopologicalSorter for cycle detection and validation.

This module is KD-GAT-specific. Other projects would write their own planner
that emits the same JobSpec objects.
"""

from __future__ import annotations

import logging
import sys
from datetime import timedelta
from graphlib import CycleError, TopologicalSorter
from typing import Any

from graphids.config.constants import STAGE_DEPENDENCIES, STAGE_MODEL_MAP

from .job import JobSpec, ResourceSpec

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Resource profiles: (model, scale, stage) → ResourceSpec
# Keyed with tuples, not fragile strings.
# ---------------------------------------------------------------------------

_RESOURCE_PROFILES: dict[tuple[str, str, str], ResourceSpec] = {
    # VGAE
    ("vgae", "large", "autoencoder"): ResourceSpec(
        gpus=1, cpus=4, memory_gb=20, walltime=timedelta(hours=3)
    ),
    ("vgae", "small", "autoencoder"): ResourceSpec(
        gpus=1, cpus=4, memory_gb=16, walltime=timedelta(hours=2)
    ),
    # GAT
    ("gat", "large", "curriculum"): ResourceSpec(
        gpus=1, cpus=4, memory_gb=16, walltime=timedelta(hours=3)
    ),
    ("gat", "large", "normal"): ResourceSpec(
        gpus=1, cpus=4, memory_gb=16, walltime=timedelta(hours=3)
    ),
    ("gat", "small", "curriculum"): ResourceSpec(
        gpus=1, cpus=4, memory_gb=12, walltime=timedelta(hours=1, minutes=30)
    ),
    ("gat", "small", "normal"): ResourceSpec(
        gpus=1, cpus=4, memory_gb=12, walltime=timedelta(hours=1, minutes=30)
    ),
    # DQN fusion (CPU-only)
    ("dqn", "large", "fusion"): ResourceSpec(
        gpus=0, cpus=4, memory_gb=16, walltime=timedelta(minutes=30)
    ),
    ("dqn", "small", "fusion"): ResourceSpec(
        gpus=0, cpus=4, memory_gb=16, walltime=timedelta(minutes=30)
    ),
    # Evaluation (CPU-only)
    ("eval", "large", "evaluation"): ResourceSpec(
        gpus=0, cpus=4, memory_gb=16, walltime=timedelta(minutes=30)
    ),
    ("eval", "small", "evaluation"): ResourceSpec(
        gpus=0, cpus=4, memory_gb=16, walltime=timedelta(minutes=30)
    ),
}

_DEFAULT_RESOURCE = ResourceSpec(gpus=1, cpus=4, memory_gb=20, walltime=timedelta(hours=3))

# Stage name → model type
_STAGE_MODEL: dict[str, str] = {
    "autoencoder": "vgae",
    "curriculum": "gat",
    "normal": "gat",
    "fusion": "dqn",
    "temporal": "gat",
}


def _get_resources(model: str, scale: str, stage: str) -> ResourceSpec:
    """Look up resource profile with fallback to default."""
    return _RESOURCE_PROFILES.get((model, scale, stage), _DEFAULT_RESOURCE)


# ---------------------------------------------------------------------------
# Plan builder
# ---------------------------------------------------------------------------


def build_plan(
    datasets: list[str],
    seeds: list[int],
    variants: list[dict[str, Any]],
) -> list[JobSpec]:
    """Build the complete job plan from variant definitions.

    Each job gets a UUID. Dependencies reference parent UUIDs.
    Parameters are stored as typed dict fields for queryability.

    Parameters
    ----------
    datasets : list[str]
        Dataset names (e.g. ["hcrl_sa", "hcrl_ch"]).
    seeds : list[int]
        Random seeds (e.g. [42, 123, 456]).
    variants : list[dict]
        Variant definitions with keys: name, scale, auxiliaries, needs_teacher, stages.

    Returns
    -------
    list[JobSpec]
        Topologically valid job list with UUID-based dependencies.
    """
    # Index for looking up jobs by (dataset, variant, model, stage, seed) → JobSpec
    job_index: dict[tuple[str, str, str, str, int], JobSpec] = {}
    jobs: list[JobSpec] = []

    # Identify teacher variant (first non-KD variant)
    teacher = next((v for v in variants if not v["needs_teacher"]), None)

    for dataset in datasets:
        for seed in seeds:
            for variant in variants:
                vname = variant["name"]
                vscale = variant["scale"]
                vaux = variant["auxiliaries"]

                # Build training stages
                training_stages: list[tuple[str, str]] = []
                for stage_name in variant["stages"]:
                    if stage_name == "evaluation":
                        continue
                    model = _STAGE_MODEL.get(stage_name)
                    if model is None:
                        log.warning(
                            "Unknown stage '%s' in variant '%s', skipping", stage_name, vname
                        )
                        continue
                    training_stages.append((model, stage_name))

                for model, stage_name in training_stages:
                    # Collect parent UUIDs
                    parent_ids = []

                    # Intra-variant dependencies (same variant, same seed)
                    for dep_model, dep_stage in STAGE_DEPENDENCIES.get(stage_name, []):
                        dep_key = (dataset, vname, dep_model, dep_stage, seed)
                        if dep_key in job_index:
                            parent_ids.append(job_index[dep_key].id)

                    # Cross-variant KD dependencies: teacher's prerequisite stages
                    if variant["needs_teacher"] and teacher and teacher["name"] != vname:
                        for dep_model, dep_stage in STAGE_DEPENDENCIES.get(stage_name, []):
                            teacher_key = (dataset, teacher["name"], dep_model, dep_stage, seed)
                            if (
                                teacher_key in job_index
                                and job_index[teacher_key].id not in parent_ids
                            ):
                                parent_ids.append(job_index[teacher_key].id)

                    job = JobSpec(
                        name=f"{dataset}/{vname}/{model}_{stage_name}/seed_{seed}",
                        executable=sys.executable,
                        arguments=[
                            "-m",
                            "graphids.pipeline.cli",
                            stage_name,
                            "--model",
                            model,
                            "--scale",
                            vscale,
                            "--dataset",
                            dataset,
                            "--seed",
                            str(seed),
                            *(["--auxiliaries", vaux] if vaux != "none" else []),
                        ],
                        parameters={
                            "dataset": dataset,
                            "seed": seed,
                            "variant": vname,
                            "model": model,
                            "stage": stage_name,
                            "scale": vscale,
                            "auxiliaries": vaux,
                        },
                        resources=_get_resources(model, vscale, stage_name),
                        parents=parent_ids,
                        tags={"variant": vname, "dataset": dataset},
                    )
                    key = (dataset, vname, model, stage_name, seed)
                    job_index[key] = job
                    jobs.append(job)

                # Evaluation stage (depends on all training stages for this variant)
                if "evaluation" in variant["stages"] and training_stages:
                    eval_parents = [
                        job_index[(dataset, vname, m, s, seed)].id
                        for m, s in training_stages
                        if (dataset, vname, m, s, seed) in job_index
                    ]
                    eval_job = JobSpec(
                        name=f"{dataset}/{vname}/eval_evaluation/seed_{seed}",
                        executable=sys.executable,
                        arguments=[
                            "-m",
                            "graphids.pipeline.cli",
                            "evaluation",
                            "--model",
                            "vgae",
                            "--scale",
                            vscale,
                            "--dataset",
                            dataset,
                            "--seed",
                            str(seed),
                            *(["--auxiliaries", vaux] if vaux != "none" else []),
                        ],
                        parameters={
                            "dataset": dataset,
                            "seed": seed,
                            "variant": vname,
                            "model": "eval",
                            "stage": "evaluation",
                            "scale": vscale,
                            "auxiliaries": vaux,
                        },
                        resources=_get_resources("eval", vscale, "evaluation"),
                        parents=eval_parents,
                        tags={"variant": vname, "dataset": dataset},
                    )
                    job_index[(dataset, vname, "eval", "evaluation", seed)] = eval_job
                    jobs.append(eval_job)

    # Validate: check for cycles
    _validate_dag(jobs)
    return jobs


def _validate_dag(jobs: list[JobSpec]) -> None:
    """Validate the job DAG is acyclic using graphlib."""
    id_set = {str(j.id) for j in jobs}
    graph: dict[str, set[str]] = {}
    for job in jobs:
        graph[str(job.id)] = {str(p) for p in job.parents if str(p) in id_set}

    try:
        ts = TopologicalSorter(graph)
        ts.prepare()
    except CycleError as e:
        raise ValueError(f"Pipeline DAG contains a cycle: {e}") from e

    log.info("DAG validated: %d jobs, no cycles", len(jobs))


def print_plan(jobs: list[JobSpec]) -> None:
    """Print a human-readable plan grouped by dataset → variant."""
    from collections import defaultdict

    by_dataset: dict[str, dict[str, list[JobSpec]]] = defaultdict(lambda: defaultdict(list))
    for job in jobs:
        ds = job.parameters.get("dataset", "unknown")
        var = job.parameters.get("variant", "unknown")
        by_dataset[ds][var].append(job)

    total = len(jobs)
    print(f"\n{'=' * 60}")
    print(f"Pipeline Plan: {total} jobs")
    print(f"{'=' * 60}")

    for ds in sorted(by_dataset):
        print(f"\n  Dataset: {ds}")
        for var in sorted(by_dataset[ds]):
            var_jobs = by_dataset[ds][var]
            gpu_jobs = sum(1 for j in var_jobs if j.resources.gpus > 0)
            cpu_jobs = len(var_jobs) - gpu_jobs
            print(f"    {var}: {len(var_jobs)} jobs ({gpu_jobs} GPU, {cpu_jobs} CPU)")
            for job in var_jobs:
                p = job.parameters
                deps = len(job.parents)
                res = job.resources
                print(
                    f"      {p.get('stage', '?'):15s} "
                    f"{'GPU' if res.gpus else 'CPU':3s} "
                    f"{res.memory_gb:2d}GB "
                    f"{res.walltime_str} "
                    f"deps={deps}"
                )

    print(f"\n{'=' * 60}\n")
