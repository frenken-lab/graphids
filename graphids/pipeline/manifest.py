"""YAML manifest → SLURM DAG submission with stage deduplication."""
from __future__ import annotations

import dataclasses
import graphlib
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import structlog
import yaml

log = structlog.get_logger()

# Reserved keys: popped from config before passing to resolve().
_RESERVED_KEYS = frozenset({"stages"})


@dataclass(frozen=True)
class StageJob:
    """One deduplicated SLURM job in the DAG."""

    node_id: str
    stage: str
    dataset: str
    seed: int
    overrides: tuple[str, ...]
    resources: dict[str, Any]
    dep_ids: tuple[str, ...]
    config_names: frozenset[str]


# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------

def load_manifest(path: Path) -> tuple[dict[str, list], dict[str, Any], dict[str, dict]]:
    """Parse manifest YAML → (sweep, defaults, configs)."""
    raw = yaml.safe_load(path.read_text())
    sweep = raw.get("sweep", {})
    defaults = raw.get("defaults", {})
    configs = raw.get("configs", {})
    return sweep, defaults, configs


# ---------------------------------------------------------------------------
# DAG builder
# ---------------------------------------------------------------------------

def _load_resources() -> dict[str, dict]:
    """Load flat resource profiles keyed by mode name (gpu_train, gpu_eval, cpu_preprocess)."""
    from graphids.config import CONFIG_DIR
    return yaml.safe_load((CONFIG_DIR / "resources.yaml").read_text()).get("resource_profiles", {})


def _resolve_stages(merged: dict[str, Any], default_stages: list[str]) -> list[str]:
    """Extract and return the stage list for a config, popping reserved keys."""
    return list(merged.pop("stages", default_stages))


def _identity_key(
    stage: str,
    dataset: str,
    seed: int,
    config: dict[str, Any],
    identity_keys: list[str],
) -> str:
    """Compute a dedup identity string for a stage run."""
    vals = "_".join(f"{k}={config.get(k, '_default_')}" for k in sorted(identity_keys))
    return f"{stage}|{dataset}|{seed}|{vals}" if vals else f"{stage}|{dataset}|{seed}"


def _config_to_overrides(config: dict[str, Any]) -> list[str]:
    """Convert flat config dict to resolve() dotlist strings."""
    return [f"{k}={v}" for k, v in config.items() if k not in _RESERVED_KEYS]


def build_dag(
    sweep: dict[str, list],
    defaults: dict[str, Any],
    configs: dict[str, dict],
) -> list[StageJob]:
    """Expand sweep x configs → dedup → topo-sort → ordered StageJob list."""
    from graphids.config import PIPELINE_YAML, STAGE_DEPENDENCIES, STAGE_MODEL_MAP

    pipeline = PIPELINE_YAML
    resources = _load_resources()
    stages_def = pipeline["stages"]
    default_stages = pipeline.get("default_stages", list(stages_def.keys()))

    # Expand sweep dimensions into Cartesian product
    sweep_keys = list(sweep.keys())
    sweep_vals = [sweep[k] for k in sweep_keys]
    sweep_points = [dict(zip(sweep_keys, combo)) for combo in product(*sweep_vals)]

    # Collect all stage jobs, deduplicating by identity
    jobs: dict[str, StageJob] = {}  # node_id -> StageJob
    # Track per-run stage->node_id mapping for dependency linking
    run_node_map: list[tuple[str, dict[str, str]]] = []  # (config_name, {stage: node_id})

    for config_name, overrides in configs.items():
        merged = {**defaults, **(overrides or {})}
        stages = _resolve_stages(merged, default_stages)

        for sweep_point in sweep_points:
            dataset = sweep_point.get("dataset", merged.get("dataset", "unknown"))
            seed = sweep_point.get("seed", merged.get("seed", 42))
            # Config for resolve = merged (without reserved) + sweep dims
            run_config = {**merged, **sweep_point}

            stage_node_ids: dict[str, str] = {}

            for stage in stages:
                if stage not in stages_def:
                    log.warning("unknown_stage", stage=stage, config=config_name)
                    continue

                sdef = stages_def[stage]
                identity_keys = sdef.get("identity_keys", [])
                node_id = _identity_key(stage, dataset, seed, run_config, identity_keys)

                if node_id in jobs:
                    existing = jobs[node_id]
                    jobs[node_id] = dataclasses.replace(
                        existing, config_names=existing.config_names | {config_name},
                    )
                else:
                    # New stage job
                    model = STAGE_MODEL_MAP[stage]
                    scale = run_config.get("scale", "small")
                    overrides_list = _config_to_overrides(run_config)
                    overrides_list = [
                        f"model_type={model}",
                        f"dataset={dataset}",
                        f"seed={seed}",
                        f"stage={stage}",
                    ] + overrides_list

                    mode = sdef.get("mode", "gpu_train")
                    res = resources.get(mode)
                    if not res:
                        raise ValueError(f"No resource profile '{mode}'. Check resources.yaml.")

                    # Dependency node_ids: look up upstream stage in this run's node map
                    dep_ids = []
                    for dep_model, dep_stage in STAGE_DEPENDENCIES.get(stage, []):
                        if dep_stage in stage_node_ids:
                            dep_ids.append(stage_node_ids[dep_stage])

                    jobs[node_id] = StageJob(
                        node_id=node_id,
                        stage=stage,
                        dataset=dataset,
                        seed=seed,
                        overrides=tuple(overrides_list),
                        resources=res,
                        dep_ids=tuple(dep_ids),
                        config_names=frozenset({config_name}),
                    )

                stage_node_ids[stage] = node_id

    # Topo-sort
    dep_graph = {j.node_id: set(j.dep_ids) for j in jobs.values()}
    order = list(graphlib.TopologicalSorter(dep_graph).static_order())
    return [jobs[nid] for nid in order if nid in jobs]


# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------

def submit_manifest(
    manifest_path: Path,
    *,
    dry_run: bool = False,
    filter_configs: list[str] | None = None,
) -> dict[str, object]:
    """Load manifest, build DAG, submit via submitit (or dry-run print)."""
    from graphids.config import SLURM_ACCOUNT, STAGE_MODEL_MAP, resolve
    from graphids.pipeline.stages import run_stage

    sweep, defaults, configs = load_manifest(manifest_path)
    if filter_configs:
        configs = {k: v for k, v in configs.items() if k in filter_configs}

    dag = build_dag(sweep, defaults, configs)
    plan_summary(dag, len(configs), sweep)

    if dry_run:
        for job in dag:
            log.info(
                "dry_run",
                node=job.node_id,
                stage=job.stage,
                deps=list(job.dep_ids),
                configs=sorted(job.config_names),
            )
        return {}

    import submitit

    futures: dict[str, object] = {}
    skipped = 0
    for job in dag:
        cfg = resolve(*job.overrides)

        # Skip-if-done: check if output checkpoint already exists
        ckpt_key = STAGE_MODEL_MAP.get(job.stage)
        if ckpt_key and job.stage != "evaluation":
            try:
                ckpt_path = cfg.checkpoints.get(ckpt_key)
                if ckpt_path and Path(ckpt_path).exists():
                    log.info("skip_completed", node=job.node_id, checkpoint=ckpt_path)
                    skipped += 1
                    # Register as done so downstream jobs have no dependency (run immediately)
                    futures[job.node_id] = None
                    continue
            except Exception:
                pass  # resolve failure = submit anyway

        # Filter out None sentinels (skipped stages) — those are already done
        dep_futs = [futures[d] for d in job.dep_ids if d in futures and futures[d] is not None]
        dep_str = (
            f"afterok:{':'.join(str(f.job_id) for f in dep_futs)}"
            if dep_futs
            else None
        )

        executor = submitit.SlurmExecutor(folder="slurm_logs/%j")
        mem_val = f"{job.resources['memory_gb']}G"
        partition = job.resources["partition"]

        # Dataset-scoped staging + skip TMPDIR for CPU inference jobs
        stage_args = f"--cache --dataset {job.dataset}"
        if partition == "cpu":
            stage_args += " --skip-tmpdir"
        setup_cmds = [
            "unset SLURM_TRES_PER_TASK",
            f'export STAGE_DATA_ARGS="{stage_args}"',
            "source scripts/slurm/_preamble.sh",
        ]

        executor.update_parameters(
            mem=mem_val,
            gpus_per_node=job.resources["gpus"],
            cpus_per_task=job.resources["cpus"],
            time=job.resources["walltime_min"],
            partition=partition,
            account=SLURM_ACCOUNT,
            setup=setup_cmds,
            additional_parameters={"dependency": dep_str} if dep_str else {},
        )
        futures[job.node_id] = executor.submit(run_stage, cfg, job.stage)
        log.info("submitted", node=job.node_id, job_id=futures[job.node_id].job_id)

    if skipped:
        log.info("dag_skipped_completed", count=skipped)
    return futures


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def plan_summary(
    jobs: list[StageJob],
    n_configs: int,
    sweep: dict[str, list],
) -> None:
    """Print dedup statistics."""
    from collections import Counter

    stage_counts = Counter(j.stage for j in jobs)
    n_sweep = 1
    for vals in sweep.values():
        n_sweep *= len(vals)
    naive = n_configs * n_sweep * len(stage_counts)

    log.info(
        "dag_plan",
        total_jobs=len(jobs),
        naive_jobs=naive,
        savings=f"{naive - len(jobs)} jobs saved by dedup",
        per_stage=dict(stage_counts),
    )

