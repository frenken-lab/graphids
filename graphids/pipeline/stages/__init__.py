"""Pipeline stages: dispatch, run, and DAG orchestration.

Public API:
    from graphids.pipeline.stages import run_stage, submit_dag
"""

from __future__ import annotations

import os
from pathlib import Path

import structlog

from .evaluation import evaluate
from .fusion import train_fusion
from .temporal import train_temporal
from .training import train_autoencoder, train_curriculum, train_normal

STAGE_FNS = {
    "autoencoder": train_autoencoder,
    "curriculum":  train_curriculum,
    "normal":      train_normal,
    "fusion":      train_fusion,
    "evaluation":  evaluate,
    "temporal":    train_temporal,
}


def run_stage(cfg, stage: str) -> dict:
    """Bind context, save config, run stage function."""
    from omegaconf import OmegaConf

    from graphids.config import STAGES

    if stage not in STAGES:
        raise ValueError(f"Unknown stage '{stage}'. Choose from: {list(STAGES.keys())}")

    structlog.contextvars.bind_contextvars(
        dataset=cfg.dataset, model=cfg.model_type, scale=cfg.scale,
        stage=stage, seed=cfg.seed,
        slurm_job_id=os.environ.get("SLURM_JOB_ID", ""),
    )
    OmegaConf.save(cfg, Path.cwd() / "config.yaml")
    return STAGE_FNS[stage](cfg)


def submit_dag(dataset: str, seeds: list[int], *, dry_run: bool = False) -> dict[str, object]:
    """Build DAG from pipeline.yaml, topo-sort, submit each node to SLURM."""
    import graphlib

    import submitit
    import yaml

    from graphids.config import CONFIG_DIR, SLURM_ACCOUNT, STAGE_DEPENDENCIES, STAGE_MODEL_MAP, resolve

    log = structlog.get_logger()

    # Resource profiles: (model, scale, stage) -> {memory_gb, walltime_min, ...}
    raw = yaml.safe_load((CONFIG_DIR / "resources.yaml").read_text())
    resources = {
        (model, scale, stage): res
        for model, scales in raw.get("resource_profiles", {}).items()
        for scale, stages in scales.items()
        for stage, res in stages.items()
    }

    pipeline = yaml.safe_load((CONFIG_DIR / "pipeline.yaml").read_text())

    def _name(model: str, scale: str, stage: str, aux: str = "") -> str:
        if model == "preprocess":
            return "preprocess"
        return f"{model}_{scale}_{stage}_{aux}" if aux else f"{model}_{scale}_{stage}"

    # Build nodes: name -> (stage, scale, cfg_model_type, deps, resources)
    nodes: dict[str, tuple[str, str, str, set[str], dict]] = {}

    for variant in pipeline["variants"].values():
        scale = variant["scale"]
        aux = variant["auxiliaries"] if variant["auxiliaries"] != "none" else ""

        for stage in variant["stages"]:
            model = STAGE_MODEL_MAP[stage]
            name = _name(model, scale, stage, aux)
            deps = {_name(dm, scale, ds, aux) for dm, ds in STAGE_DEPENDENCIES.get(stage, [])}

            if variant["needs_teacher"] and stage in ("autoencoder", "curriculum"):
                teacher = _name(model, "large", stage)
                if teacher != name:
                    deps.add(teacher)

            cfg_model = "vgae" if stage == "evaluation" else model
            nodes[name] = (stage, scale, cfg_model, deps, resources[(model, scale, stage)])

    nodes["preprocess"] = ("preprocess", "", "preprocess", set(), resources[("preprocess", "", "preprocess")])

    # Topo-sort and submit
    order = graphlib.TopologicalSorter({n: d for n, (_, _, _, d, _) in nodes.items()}).static_order()
    all_futures: dict[str, object] = {}

    for seed in seeds:
        futures: dict[str, object] = {}
        for name in order:
            stage, scale, cfg_model, deps, res = nodes[name]

            if dry_run:
                log.info("dry_run", asset=name, deps=sorted(deps))
                continue

            cfg = resolve(f"model_type={cfg_model}", f"scale={scale}", f"dataset={dataset}", f"seed={seed}")
            dep_futs = [futures[d] for d in deps if d in futures]
            dep_str = f"afterok:{':'.join(str(f.job_id) for f in dep_futs)}" if dep_futs else None

            executor = submitit.SlurmExecutor(folder="slurm_logs/%j")
            executor.update_parameters(
                mem_gb=res["memory_gb"], gpus_per_node=res.get("gpus", 0),
                cpus_per_task=res.get("cpus", 4), timeout_min=res["walltime_min"],
                partition=res.get("partition", "gpu"), account=SLURM_ACCOUNT,
                setup=["source scripts/slurm/_preamble.sh"], dependency=dep_str,
            )
            futures[name] = executor.submit(run_stage, cfg, stage)
            all_futures[f"{name}__seed{seed}"] = futures[name]

    return all_futures
