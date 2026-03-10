"""Ray-based pipeline orchestration for KD-GAT.

    preprocess ──┬──► large pipeline (vgae → gat → dqn → eval)
                 │         │ teacher checkpoints       ┌──► small_nokd pipeline
                 │         ▼                           │    (concurrent, no dependency
                 ├──► small_kd pipeline                │     on teacher checkpoints)
                 │    (vgae → gat → dqn → eval)       │
                 └────────────────────────────────────join

Each stage runs as a subprocess for clean CUDA context.
Ray handles DAG scheduling and per-dataset fan-out.

Pipeline variants are now config-driven via PipelineConfig.variants.
The default variant set (large, small_kd, small_nokd) preserves the
original hardcoded behavior.

Usage:
    python -m graphids.pipeline.cli flow --dataset hcrl_sa
    python -m graphids.pipeline.cli flow --dataset hcrl_sa --scale large
    python -m graphids.pipeline.cli flow --eval-only --dataset hcrl_sa
    python -m graphids.pipeline.cli flow --local  # Ray local mode
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time

import ray

from graphids.pipeline.subprocess_utils import build_cli_cmd

log = logging.getLogger(__name__)

_PY = sys.executable


def _init_ray(datasets: list[str] | None, local: bool) -> list[str]:
    """Resolve datasets and initialize Ray if needed."""
    from .ray_slurm import ray_init_kwargs

    if datasets is None:
        from graphids.config.paths import get_datasets

        datasets = get_datasets()
    if not ray.is_initialized():
        kwargs = ray_init_kwargs()
        if local:
            kwargs["num_gpus"] = 0
        ray.init(**kwargs)
    return datasets


# ---------------------------------------------------------------------------
# Subprocess dispatch
# ---------------------------------------------------------------------------


def _run_stage(
    stage: str,
    model: str,
    scale: str,
    dataset: str,
    auxiliaries: str = "none",
    seed: int | None = None,
) -> subprocess.CompletedProcess:
    """Run a pipeline stage as a subprocess via the CLI.

    Using subprocess ensures each stage gets a clean CUDA context
    (critical for spawn multiprocessing).

    KD teacher resolution is handled automatically by the training code
    via ``prepare_kd()`` — no need to pass teacher paths through the
    subprocess boundary.
    """
    cmd = build_cli_cmd(stage, model, scale, dataset, seed=seed, auxiliaries=auxiliaries)

    log.info("Running: %s", " ".join(cmd))

    t0 = time.monotonic()
    result = subprocess.run(cmd, check=True, capture_output=False)
    elapsed = time.monotonic() - t0
    log.info(
        "Stage %s/%s/%s completed in %.1fs (dataset=%s)",
        model,
        scale,
        stage,
        elapsed,
        dataset,
    )
    return result


# ---------------------------------------------------------------------------
# Ray remote tasks
# ---------------------------------------------------------------------------


@ray.remote(num_gpus=0)
def task_preprocess(dataset: str) -> None:
    """Ensure preprocessed graph cache exists for a dataset (CPU-only)."""
    from graphids.config import cache_dir, data_dir
    from graphids.config.resolver import resolve
    from graphids.core.training.datamodules import load_dataset

    cfg = resolve("vgae", "large", dataset=dataset)
    load_dataset(dataset, data_dir(cfg), cache_dir(cfg), seed=cfg.seed)
    log.info("Preprocessed cache ready for %s", dataset)


def _make_stage_task(stage: str, model: str):
    """Factory for Ray remote tasks that train a model and return its checkpoint path."""

    @ray.remote(num_gpus=1)
    def task(
        dataset: str,
        scale: str,
        auxiliaries: str = "none",
        seed: int | None = None,
    ) -> str:
        _run_stage(stage, model, scale, dataset, auxiliaries, seed=seed)
        from graphids.config import checkpoint_path
        from graphids.config.resolver import resolve

        cfg = resolve(model, scale, auxiliaries=auxiliaries, dataset=dataset)
        return str(checkpoint_path(cfg, stage))

    task.__name__ = f"task_{model}"
    return task


task_vgae = _make_stage_task("autoencoder", "vgae")
task_gat = _make_stage_task("curriculum", "gat")
task_dqn = _make_stage_task("fusion", "dqn")

_STAGE_TASKS = {
    "autoencoder": task_vgae,
    "curriculum": task_gat,
    "fusion": task_dqn,
}


@ray.remote(num_gpus=1)
def task_eval(dataset: str, scale: str, auxiliaries: str = "none", seed: int | None = None) -> None:
    """Run evaluation on all trained models for a variant."""
    _run_stage("evaluation", "vgae", scale, dataset, auxiliaries, seed=seed)


# ---------------------------------------------------------------------------
# Config-driven variant pipeline
# ---------------------------------------------------------------------------


def variant_pipeline(
    dataset: str,
    variant_name: str,
    scale: str,
    stages: list[str],
    auxiliaries: str = "none",
    seed: int | None = None,
) -> dict[str, str]:
    """Execute a config-driven stage chain for a single variant.

    Returns a dict of {stage_name: checkpoint_path} for stages that
    produce checkpoints (autoencoder, curriculum, fusion).

    KD teacher resolution is automatic — each stage's training code
    calls ``prepare_kd()`` which resolves the teacher from
    ``cfg.kd.teacher_scale`` via the artifact resolver.
    """
    ckpts: dict[str, str] = {}

    for stage_name in stages:
        if stage_name == "evaluation":
            ray.get(task_eval.remote(dataset, scale, auxiliaries=auxiliaries, seed=seed))
            continue

        task = _STAGE_TASKS.get(stage_name)
        if task is None:
            log.warning("Unknown stage '%s' in variant '%s', skipping", stage_name, variant_name)
            continue

        ckpt = ray.get(task.remote(dataset, scale, auxiliaries, seed=seed))
        ckpts[stage_name] = ckpt

    return ckpts


@ray.remote
def _variant_pipeline_remote(
    dataset: str,
    variant_name: str,
    scale: str,
    stages: list[str],
    auxiliaries: str = "none",
    seed: int | None = None,
) -> dict[str, str]:
    """Remote wrapper for variant_pipeline."""
    return variant_pipeline(dataset, variant_name, scale, stages, auxiliaries, seed)


# ---------------------------------------------------------------------------
# Per-dataset orchestration
# ---------------------------------------------------------------------------


@ray.remote
def dataset_pipeline(dataset: str, scale: str | None = None, seed: int | None = None) -> None:
    """All variants for a single dataset, driven by PipelineConfig.variants.

    When running all variants (scale=None), variants with needs_teacher=False
    launch concurrently with the teacher variant. Variants with
    needs_teacher=True wait for the teacher's checkpoint paths.
    """
    from graphids.config.resolver import resolve

    log.info("=== Pipeline for dataset: %s (seed=%s) ===", dataset, seed)

    # Load variant config from defaults
    cfg = resolve("vgae", "large", dataset=dataset)
    variants = cfg.variants

    # Preprocess (shared by all variants)
    ray.get(task_preprocess.remote(dataset))

    # Filter variants by scale if specified
    _scale_to_variant = {
        "large": "large",
        "small_kd": "small_kd",
        "small_nokd": "small_nokd",
    }
    if scale is not None:
        target_name = _scale_to_variant.get(scale, scale)
        variants = [v for v in variants if v.name == target_name]

    # Separate into teacher (first non-needs_teacher variant, usually "large")
    # and dependent/independent variants
    teacher_variant = None
    independent_variants = []
    dependent_variants = []

    for v in variants:
        if not v.needs_teacher and teacher_variant is None and v.scale == "large":
            teacher_variant = v
        elif v.needs_teacher:
            dependent_variants.append(v)
        else:
            independent_variants.append(v)

    # Launch independent variants concurrently (no teacher dependency)
    independent_refs = []
    for v in independent_variants:
        ref = _variant_pipeline_remote.remote(
            dataset,
            v.name,
            v.scale,
            v.stages,
            v.auxiliaries,
            seed=seed,
        )
        independent_refs.append(ref)
        log.info("Launched %s concurrently for %s", v.name, dataset)

    # Run teacher variant first (blocking — dependent variants need its checkpoints
    # to exist on disk so prepare_kd() can resolve them)
    if teacher_variant is not None:
        variant_pipeline(
            dataset,
            teacher_variant.name,
            teacher_variant.scale,
            teacher_variant.stages,
            teacher_variant.auxiliaries,
            seed=seed,
        )

    # Run dependent variants (teacher checkpoints auto-resolved by prepare_kd)
    for v in dependent_variants:
        variant_pipeline(
            dataset,
            v.name,
            v.scale,
            v.stages,
            v.auxiliaries,
            seed=seed,
        )

    # Join independent variants
    if independent_refs:
        ray.get(independent_refs)


# ---------------------------------------------------------------------------
# Top-level entry points
# ---------------------------------------------------------------------------


def train_pipeline(
    datasets: list[str] | None = None,
    scale: str | None = None,
    local: bool = False,
    seeds: list[int] | None = None,
) -> None:
    """Full KD-GAT training pipeline.

    Parameters
    ----------
    datasets : list[str] | None
        Datasets to train on.  None = all from catalog.
    scale : str | None
        If set, only run the specified scale variant
        ("large", "small_kd", "small_nokd").  None = all.
    local : bool
        If True, use Ray local mode (no cluster).
    seeds : list[int] | None
        Seeds to train with. None = single run with default seed.
    """
    datasets = _init_ray(datasets, local)

    if seeds:
        # Multi-seed: fan out all dataset×seed combos; Ray serializes on single-GPU
        refs = []
        for seed in seeds:
            log.info("=== Queuing seed %d ===", seed)
            refs.extend(dataset_pipeline.remote(ds, scale, seed=seed) for ds in datasets)
        ray.get(refs)
    else:
        # Single seed (default)
        refs = [dataset_pipeline.remote(ds, scale) for ds in datasets]
        ray.get(refs)

    log.info("=== Pipeline complete for %d dataset(s) ===", len(datasets))


def eval_pipeline(
    datasets: list[str] | None = None,
    scale: str | None = None,
    local: bool = False,
) -> None:
    """Re-run evaluation for existing trained models."""
    datasets = _init_ray(datasets, local)

    _EVAL_VARIANTS = [
        ("large", "large", "none"),
        ("small_kd", "small", "kd_standard"),
        ("small_nokd", "small", "none"),
    ]

    refs = []
    for ds in datasets:
        log.info("=== Evaluation for dataset: %s ===", ds)
        for name, sz, aux in _EVAL_VARIANTS:
            if scale is None or scale == name:
                kwargs = {"auxiliaries": aux} if aux != "none" else {}
                refs.append(task_eval.remote(ds, sz, **kwargs))

    ray.get(refs)
    log.info("=== Evaluation complete for %d dataset(s) ===", len(datasets))
