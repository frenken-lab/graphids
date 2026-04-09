"""Monarch pipeline orchestration — schemas, job specs, execution.

All public symbols are consumed exclusively by cli/_monarch.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal  # noqa: F401 (resolved by model_rebuild)

from pydantic import (  # noqa: F401 (AfterValidator resolved by model_rebuild)
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
)

from graphids.config.constants import (  # noqa: F401 (resolved by model_rebuild)
    PIPELINE_DEFAULTS,
    VALID_FUSION_METHODS,
    VALID_SCALES,
)
from graphids.config.topology import TOPOLOGY  # noqa: F401 (resolved by model_rebuild)
from graphids._otel import get_logger
from graphids.orchestrate.planning import StageConfig
from graphids.orchestrate.planning.recipes import (  # noqa: F401 (resolved by model_rebuild)
    TrainingRunConfig,
    check_in,
)

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def check_all_in(valid, label):  # noqa: F401 (resolved by model_rebuild)
    def _v(v):
        bad = [x for x in v if x not in valid]
        if bad:
            raise ValueError(f"Unknown {label}(s): {bad}. Valid: {sorted(valid)}")
        return v

    return _v


_D = PIPELINE_DEFAULTS

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class PipelineConfig(BaseModel):
    """What to run in a single Monarch allocation (monarch-run CLI)."""

    model_config = ConfigDict(frozen=True)

    dataset: str = _D.get("dataset", "hcrl_ch")
    seed: int = _D.get("seed", 42)
    scale: Annotated[str, AfterValidator(check_in(VALID_SCALES, "scale"))] = _D.get(
        "scale", "small"
    )
    lake_root: str = ""
    fusion_method: Annotated[
        str, AfterValidator(check_in(VALID_FUSION_METHODS, "fusion_method"))
    ] = _D.get("fusion_method", "bandit")
    stages: Annotated[list[str], AfterValidator(check_all_in(TOPOLOGY.stages, "stage"))] = Field(
        default_factory=lambda: list(_D.get("stages", ["autoencoder", "supervised", "fusion"])),
    )
    conv_type: Literal["gatv2", "gat", "gps"] = _D.get("conv_type", "gatv2")
    variational: bool = _D.get("variational", True)
    loss_fn: Literal["focal", "ce", "weighted_ce"] = _D.get("loss_fn", "focal")
    tla_overrides: dict[str, Any] = Field(default_factory=dict)
    max_retries: int = 2

    def to_training_run(self) -> TrainingRunConfig:
        """Convert CLI fields to a planner-ready TrainingRunConfig."""
        return TrainingRunConfig(
            stages=tuple(self.stages),
            scale=self.scale,
            conv_type=self.conv_type,
            variational=self.variational,
            loss_fn=self.loss_fn,
            fusion_method=self.fusion_method,
        )


PipelineConfig.model_rebuild()


# ---------------------------------------------------------------------------
# SLURM job spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JobSpec:
    """SLURM allocation spec for a multi-stage pipeline job."""

    partition: str
    time: str
    mem: str
    cpus: int
    gpus_per_node: int = 1
    account: str = ""
    job_name: str = "graphids-monarch"

    def __post_init__(self) -> None:
        if not self.account:
            from graphids._slurm import slurm_account

            object.__setattr__(self, "account", slurm_account())

    def create_job(self) -> Any:
        """Create a Monarch SlurmJob from this spec."""
        from monarch.job import SlurmJob  # type: ignore[import-not-found]

        _patch_clusterscope()

        from graphids.config.constants import PROJECT_ROOT
        from graphids._slurm import slurm_log_dir

        log_dir = Path(slurm_log_dir())
        log_dir.mkdir(parents=True, exist_ok=True)

        return SlurmJob(
            meshes={"pipeline": 1},
            job_name=self.job_name,
            partition=self.partition,
            time_limit=self.time,
            mem=self.mem,
            cpus_per_task=self.cpus,
            gpus_per_node=self.gpus_per_node,
            python_exe=str(PROJECT_ROOT / "scripts" / "slurm" / "monarch_python.sh"),
            log_dir=str(log_dir),
            slurm_args=(
                f"--account={self.account}",
                "--signal=B:USR1@300",
                "--export=ALL",
            ),
            exclusive=False,
        )


def _patch_clusterscope() -> None:
    """Fix clusterscope's sinfo parsers for OSC's multi-GRES output."""
    try:
        import clusterscope.cluster_info as _cci
        import clusterscope.slurm.partition as _csp
        from clusterscope.shell import run_cli
        from clusterscope.slurm.parser import parse_gres
    except ImportError:
        return

    def _fixed_partition_resources(partition: str) -> dict:
        result = run_cli(["sinfo", "-o", "%G,%c", f"--partition={partition}", "--noheader"])
        max_gpus = max_cpus = 0
        for line in result.strip().split("\n"):
            if not line:
                continue
            gres, _, cpus = line.rpartition(",")
            max_gpus = max(max_gpus, parse_gres(gres))
            max_cpus = max(max_cpus, int(cpus.rstrip("+")))
        return {"max_gpus": max_gpus, "max_cpus": max_cpus}

    _csp.get_partition_resources = _fixed_partition_resources

    def _fixed_get_gpu(self):
        cmd = ["sinfo", "-o", "%G,%P", "--noheader"]
        if self.partition:
            cmd.extend(["-p", self.partition])
        result = run_cli(cmd)
        results, seen = [], set()
        for line in result.strip().splitlines():
            gres, _, partition = line.rpartition(",")
            partition = partition.strip("* ")
            key = gres.split("(")[0] + partition
            if key in seen:
                continue
            seen.add(key)
            parts = gres.split(":")
            if len(parts) >= 3:
                results.append(
                    _cci.GPUInfo(
                        gpu_gen=parts[1],
                        gpu_count=int(parts[2].split("(")[0]),
                        vendor="nvidia",
                        partition=partition,
                    )
                )
        if not results:
            raise RuntimeError("No GPU information found")
        return results

    _cci.SlurmClusterInfo.get_gpu_generation_and_count = _fixed_get_gpu


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------


def build_pipeline_stages(config: PipelineConfig) -> list[StageConfig]:
    """PipelineConfig -> StageConfigs via the planner. Also used by CLI dry-run."""
    from graphids.orchestrate.planning import enumerate_assets

    recipe = {
        "defaults": config.to_training_run().model_dump(),
        "configs": {"default": {}},
        "sweep": {},
        "trainer_overrides": dict(config.tla_overrides),
        "stage_overrides": {},
        "resource_overrides": {},
    }
    configs = enumerate_assets(recipe)
    stage_order = {s: i for i, s in enumerate(config.stages)}
    configs.sort(key=lambda c: stage_order.get(c.stage, 99))
    return configs


def run_chain(
    stages: list[StageConfig],
    spec: JobSpec,
    *,
    dataset: str,
    seed: int,
    max_retries: int = 2,
    lake_root: str = "",
) -> dict[str, str]:
    """Run one chain in a single SLURM allocation. Returns {stage: ckpt_path}."""
    from monarch.config import configure  # type: ignore[import-not-found]

    from graphids.config.constants import LAKE_ROOT
    from graphids.orchestrate._setup import bootstrap_staging
    from graphids.orchestrate.actors import PipelineActor

    lake_root = lake_root or LAKE_ROOT
    configure(
        enable_log_forwarding=True,
        process_exit_timeout="60s",
        cleanup_timeout="30s",
        mesh_terminate_timeout="30s",
        host_spawn_ready_timeout="120s",
    )

    job = spec.create_job()
    try:
        proc_mesh = job.state().pipeline.spawn_procs(
            per_host={"gpus": spec.gpus_per_node},
            bootstrap=lambda: bootstrap_staging(dataset),
        )
        actor = proc_mesh.spawn("pipeline", PipelineActor, lake_root=lake_root)

        checkpoints: dict[str, str] = {}
        for cfg in stages:
            upstream = {n: checkpoints[n] for n in cfg.upstream_asset_names if n in checkpoints}
            call = lambda c=cfg, u=upstream: actor.train_stage.call_one(  # noqa: E731
                stage_config=c.model_dump(),
                dataset=dataset,
                seed=seed,
                upstream_ckpts=u,
            )
            for attempt in range(max_retries + 1):
                try:
                    checkpoints[cfg.asset_name] = call().get()
                    break
                except Exception as exc:
                    log.error("stage_failed", stage=cfg.stage, attempt=attempt, error=str(exc))
                    if attempt >= max_retries:
                        raise RuntimeError(
                            f"{cfg.stage} failed after {max_retries + 1} attempts"
                        ) from exc

        for cfg in stages:
            upstream = {n: checkpoints[n] for n in cfg.upstream_asset_names if n in checkpoints}
            try:
                actor.eval_stage.call_one(
                    stage_config=cfg.model_dump(),
                    dataset=dataset,
                    seed=seed,
                    upstream_ckpts=upstream,
                ).get()
            except Exception as exc:
                log.warning("eval_failed", stage=cfg.stage, error=str(exc))

        return {cfg.stage: checkpoints.get(cfg.asset_name, "") for cfg in stages}
    finally:
        try:
            job.kill()
        except Exception:
            pass
