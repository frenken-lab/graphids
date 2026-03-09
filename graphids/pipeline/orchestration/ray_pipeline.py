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

import json
import logging
import os
import shutil
import subprocess
import sys
import time

import ray

log = logging.getLogger(__name__)

_PY = sys.executable

# Set KD_GAT_BENCHMARK=1 to enable detailed orchestration timing.
# Output written to KD_GAT_BENCHMARK_LOG (default: benchmark_timing.jsonl).
_BENCHMARK = os.environ.get("KD_GAT_BENCHMARK", "") == "1"
_BENCHMARK_LOG = os.environ.get("KD_GAT_BENCHMARK_LOG", "benchmark_timing.jsonl")

# Track when the last stage ended so we can measure inter-stage gaps.
_last_stage_end: float | None = None

from graphids.config.constants import STAGE_MODEL_MAP

# Stage name → (model_type, stage_cli_name). Derived from STAGE_MODEL_MAP + evaluation.
_STAGE_DISPATCH = {stage: (model, stage) for stage, model in STAGE_MODEL_MAP.items()}
_STAGE_DISPATCH["evaluation"] = ("vgae", "evaluation")


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


def _query_gpu_utilization() -> dict[str, float | None]:
    """Sample GPU utilization and memory via nvidia-smi. Returns {} on failure."""
    if not shutil.which("nvidia-smi"):
        return {}
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode != 0:
            return {}
        # Take the first GPU line (single-GPU jobs).
        parts = out.stdout.strip().split("\n")[0].split(",")
        return {
            "gpu_util_pct": float(parts[0].strip()),
            "gpu_mem_used_mib": float(parts[1].strip()),
            "gpu_mem_total_mib": float(parts[2].strip()),
        }
    except Exception:
        return {}


def _write_benchmark_record(record: dict) -> None:
    """Append a JSONL timing record to the benchmark log."""
    with open(_BENCHMARK_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Subprocess dispatch
# ---------------------------------------------------------------------------


def _run_stage(
    stage: str,
    model: str,
    scale: str,
    dataset: str,
    auxiliaries: str = "none",
    teacher_path: str | None = None,
    seed: int | None = None,
) -> subprocess.CompletedProcess:
    """Run a pipeline stage as a subprocess via the CLI.

    Using subprocess ensures each stage gets a clean CUDA context
    (critical for spawn multiprocessing). Logs wall-clock timing
    for benchmarking subprocess overhead vs training time.

    When KD_GAT_BENCHMARK=1, writes detailed timing to a JSONL log:
    - spawn_overhead_s: time for subprocess to start (Popen → first poll)
    - execution_s: wall-clock time of the subprocess itself
    - total_s: full wall time including spawn + teardown
    - inter_stage_gap_s: idle time since the previous stage ended
    - gpu_pre/gpu_post: nvidia-smi snapshots before/after
    """
    global _last_stage_end

    cmd = [
        _PY,
        "-m",
        "graphids.pipeline.cli",
        stage,
        "--model",
        model,
        "--scale",
        scale,
        "--dataset",
        dataset,
        "--auxiliaries",
        auxiliaries,
    ]
    if teacher_path:
        cmd.extend(["--teacher-path", teacher_path])
    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    log.info("Running: %s", " ".join(cmd))

    if not _BENCHMARK:
        # Fast path: original behavior, no extra overhead.
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

    # --- Benchmark path: detailed timing instrumentation ---
    gpu_pre = _query_gpu_utilization()
    inter_stage_gap = None
    if _last_stage_end is not None:
        inter_stage_gap = time.monotonic() - _last_stage_end

    t_call = time.monotonic()

    # Use Popen to measure spawn overhead separately from execution.
    proc = subprocess.Popen(cmd)
    t_spawned = time.monotonic()
    spawn_overhead = t_spawned - t_call

    proc.wait()
    t_done = time.monotonic()

    _last_stage_end = t_done

    execution_time = t_done - t_spawned
    total_time = t_done - t_call

    gpu_post = _query_gpu_utilization()

    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)

    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset": dataset,
        "model": model,
        "scale": scale,
        "stage": stage,
        "auxiliaries": auxiliaries,
        "spawn_overhead_s": round(spawn_overhead, 3),
        "execution_s": round(execution_time, 3),
        "total_s": round(total_time, 3),
        "inter_stage_gap_s": round(inter_stage_gap, 3) if inter_stage_gap is not None else None,
        "gpu_pre": gpu_pre,
        "gpu_post": gpu_post,
    }
    _write_benchmark_record(record)

    log.info(
        "Stage %s/%s/%s completed in %.1fs (spawn=%.3fs, exec=%.1fs, gap=%.3fs, dataset=%s)",
        model,
        scale,
        stage,
        total_time,
        spawn_overhead,
        execution_time,
        inter_stage_gap if inter_stage_gap is not None else 0.0,
        dataset,
    )
    return subprocess.CompletedProcess(cmd, proc.returncode)


# ---------------------------------------------------------------------------
# Ray remote tasks
# ---------------------------------------------------------------------------


@ray.remote(num_gpus=1)
def task_preprocess(dataset: str) -> None:
    """Ensure preprocessed graph cache exists for a dataset."""
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
        teacher_path: str | None = None,
        seed: int | None = None,
    ) -> str:
        _run_stage(stage, model, scale, dataset, auxiliaries, teacher_path, seed=seed)
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


def _get_teacher_ckpts(dataset: str) -> dict[str, str]:
    """Load teacher checkpoint paths from existing large variant runs."""
    from graphids.config import checkpoint_path
    from graphids.config.resolver import resolve

    teacher_paths = {}
    for model, stage in [("vgae", "autoencoder"), ("gat", "curriculum"), ("dqn", "fusion")]:
        cfg = resolve(model, "large", dataset=dataset)
        tp = checkpoint_path(cfg, stage)
        if not tp.exists():
            raise FileNotFoundError(
                f"Teacher checkpoint not found: {tp}. Run with --scale large first."
            )
        teacher_paths[model] = str(tp)
    return teacher_paths


def variant_pipeline(
    dataset: str,
    variant_name: str,
    scale: str,
    stages: list[str],
    auxiliaries: str = "none",
    teacher_ckpts: dict[str, str] | None = None,
    seed: int | None = None,
) -> dict[str, str]:
    """Execute a config-driven stage chain for a single variant.

    Returns a dict of {stage_name: checkpoint_path} for stages that
    produce checkpoints (autoencoder, curriculum, fusion).
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

        # Determine teacher path for this stage's model type
        teacher_path = None
        if teacher_ckpts and auxiliaries != "none":
            model_type, _ = _STAGE_DISPATCH[stage_name]
            teacher_path = teacher_ckpts.get(model_type)

        ckpt = ray.get(task.remote(dataset, scale, auxiliaries, teacher_path, seed=seed))
        ckpts[stage_name] = ckpt

    return ckpts


@ray.remote
def _variant_pipeline_remote(
    dataset: str,
    variant_name: str,
    scale: str,
    stages: list[str],
    auxiliaries: str = "none",
    teacher_ckpts: dict[str, str] | None = None,
    seed: int | None = None,
) -> dict[str, str]:
    """Remote wrapper for variant_pipeline."""
    return variant_pipeline(dataset, variant_name, scale, stages, auxiliaries, teacher_ckpts, seed)


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

    # Run teacher variant (blocking — dependent variants need its checkpoints)
    teacher_ckpts = None
    if teacher_variant is not None:
        teacher_ckpts = variant_pipeline(
            dataset,
            teacher_variant.name,
            teacher_variant.scale,
            teacher_variant.stages,
            teacher_variant.auxiliaries,
            seed=seed,
        )

    # Run dependent variants (they need teacher checkpoints)
    for v in dependent_variants:
        if teacher_ckpts is None:
            # Teacher must already exist on disk
            teacher_ckpts = _get_teacher_ckpts(dataset)
        variant_pipeline(
            dataset,
            v.name,
            v.scale,
            v.stages,
            v.auxiliaries,
            teacher_ckpts=teacher_ckpts,
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
        # Multi-seed: run full pipeline per seed sequentially
        for seed in seeds:
            log.info("=== Seed %d ===", seed)
            refs = [dataset_pipeline.remote(ds, scale, seed=seed) for ds in datasets]
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
