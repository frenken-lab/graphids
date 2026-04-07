"""Pipeline controller -- sequences stages via Monarch actor endpoints.

Runs on the login node or in a notebook. Creates a SlurmJob, spawns
PipelineActor, and sequences stage endpoint calls with retry.

Execution order: all trains first (critical path), then all evals
(test + analyze + finalize) as a lenient batch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from graphids.log import get_logger

log = get_logger(__name__)


@dataclass
class PipelineConfig:
    """What to run in a single Monarch allocation."""

    dataset: str = "hcrl_ch"
    seed: int = 42
    scale: str = "small"
    lake_root: str = ""
    fusion_method: str = "bandit"
    stages: list[str] = field(
        default_factory=lambda: ["autoencoder", "supervised", "fusion"],
    )
    conv_type: str = "gatv2"
    variational: bool = True
    tla_overrides: dict[str, Any] = field(default_factory=dict)
    max_retries: int = 2


def run_pipeline(config: PipelineConfig) -> dict[str, str]:
    """Run the training pipeline in a single SLURM allocation.

    Returns dict mapping stage name to checkpoint path.
    """
    from monarch.config import configure  # type: ignore[import-not-found]

    from graphids.config.constants import LAKE_ROOT
    from graphids.monarch.job import create_slurm_job, pipeline_job_spec

    lake_root = config.lake_root or LAKE_ROOT

    # Monarch runtime config — timeouts tuned for Lightning + NFS
    configure(
        enable_log_forwarding=True,
        process_exit_timeout="60s",
        cleanup_timeout="30s",
        mesh_terminate_timeout="30s",
        host_spawn_ready_timeout="120s",
    )

    # 1. Create SLURM allocation — job.state() handles caching/reconnect
    spec = pipeline_job_spec(config.scale, fusion_method=config.fusion_method)
    log.info(
        "monarch_pipeline_start",
        partition=spec.partition,
        time=spec.time,
        mem=spec.mem,
        stages=config.stages,
    )

    job = create_slurm_job(spec)
    state = job.state()  # blocks until allocation is ready; caches to .monarch/
    host_mesh = state.pipeline

    # 2. Spawn procs with data staging bootstrap, then spawn actor
    from graphids.monarch.actors import PipelineActor, bootstrap_staging

    ds = config.dataset
    proc_mesh = host_mesh.spawn_procs(
        per_host={"gpus": spec.gpus_per_node},
        bootstrap=lambda: bootstrap_staging(ds),
    )

    actor = proc_mesh.spawn(
        "pipeline",
        PipelineActor,
        dataset=config.dataset,
        seed=config.seed,
        scale=config.scale,
        lake_root=lake_root,
        conv_type=config.conv_type,
        variational=config.variational,
    )

    # 3. Critical path: all trains first (fail-fast, sequential deps)
    checkpoints: dict[str, str] = {}
    _common = dict(
        tla_overrides=config.tla_overrides,
        fusion_method=config.fusion_method,
    )

    for stage in config.stages:
        checkpoints[stage] = _run_with_retry(
            lambda s=stage: actor.train_stage.call_one(
                stage=s,
                vgae_ckpt_path=checkpoints.get("autoencoder"),
                gat_ckpt_path=checkpoints.get("supervised"),
                **_common,
            ),
            f"train:{stage}",
            config.max_retries,
        )

    log.info("all_trains_complete", checkpoints=checkpoints)

    # 4. Eval batch: test + analyze + finalize for each stage (lenient)
    for stage in config.stages:
        try:
            log.info("eval_start", stage=stage)
            actor.eval_stage.call_one(
                stage=stage,
                vgae_ckpt_path=checkpoints.get("autoencoder"),
                gat_ckpt_path=checkpoints.get("supervised"),
                **_common,
            ).get()
            log.info("eval_complete", stage=stage)
        except Exception as exc:
            log.warning("eval_failed", stage=stage, error=str(exc))

    log.info("monarch_pipeline_complete", checkpoints=checkpoints)

    try:
        job.kill()
    except Exception:
        log.warning("monarch_job_kill_failed")

    return checkpoints


def _run_with_retry(fn: Any, stage_name: str, max_retries: int) -> str:
    """Run a stage endpoint call with retry on failure.

    Monarch's supervision tree absorbs actor failures but does NOT
    auto-restart. Retry logic lives here in the controller.
    """
    for attempt in range(max_retries + 1):
        try:
            log.info("stage_start", stage=stage_name, attempt=attempt)
            future = fn()
            ckpt_path = future.get()
            log.info("stage_complete", stage=stage_name, ckpt=ckpt_path)
            return ckpt_path
        except Exception as exc:
            log.error(
                "stage_failed",
                stage=stage_name,
                attempt=attempt,
                error=str(exc),
            )
            if attempt >= max_retries:
                raise RuntimeError(
                    f"Stage {stage_name!r} failed after {max_retries + 1} attempts"
                ) from exc
    raise AssertionError("unreachable")  # pragma: no cover
