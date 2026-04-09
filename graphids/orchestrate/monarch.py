"""Monarch pipeline orchestration — schemas, job specs, sweep planning, execution.

Consolidates the former schemas.py, job.py, sweep.py, and pipeline.py into
one module. All public symbols are consumed exclusively by cli/_monarch.py.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
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
from graphids.log import get_logger
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


class SweepConfig(BaseModel):
    """Full sweep run configuration (monarch-sweep CLI)."""

    model_config = ConfigDict(frozen=True)

    recipe_path: str
    datasets: list[str] = Field(default_factory=lambda: [_D.get("dataset", "hcrl_ch")])
    seeds: list[int] = Field(default_factory=lambda: [_D.get("seed", 42)])
    lake_root: str = ""
    max_retries: int = 2
    max_concurrent: int = 0  # 0 = all parallel


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
            from graphids.slurm.env import SLURM_ACCOUNT

            object.__setattr__(self, "account", SLURM_ACCOUNT)

    def create_job(self) -> Any:
        """Create a Monarch SlurmJob from this spec."""
        from monarch.job import SlurmJob  # type: ignore[import-not-found]

        _patch_clusterscope()

        from graphids.config.constants import PROJECT_ROOT
        from graphids.slurm.env import SLURM_LOG_DIR

        log_dir = Path(SLURM_LOG_DIR)
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


def chain_job_spec(
    stages: list[Any],
    *,
    job_name: str = "graphids-monarch",
    dataset: str | None = None,
) -> JobSpec:
    """Compute a combined allocation covering all stages in a chain."""
    from graphids.slurm.resources import get_resources

    resources = [
        get_resources(cfg.resource_model or cfg.model_type, cfg.scale, cfg.stage, dataset=dataset)
        for cfg in stages
    ]

    total_minutes = sum(r.time_minutes for r in resources) + 30
    h, m = divmod(total_minutes, 60)

    gpu_resources = [r for r in resources if r.gres]
    if gpu_resources:
        partition = gpu_resources[0].partition
        parts = gpu_resources[0].gres.split(":")
        gpus = int(parts[-1]) if parts[-1].isdigit() else 1
    else:
        partition = resources[0].partition
        gpus = 0

    return JobSpec(
        partition=partition,
        time=f"{h}:{m:02d}:00",
        mem=f"{max(r.mem_mb for r in resources) // 1024}G",
        cpus=max(r.cpus_per_task for r in resources),
        gpus_per_node=gpus,
        job_name=job_name,
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
# Sweep planning
# ---------------------------------------------------------------------------


@dataclass
class ChainSpec:
    """One maximal DAG path — becomes one Monarch allocation."""

    chain_id: str
    stages: list[StageConfig]
    dataset: str
    seed: int


def plan_chains(
    recipe_path: str | Path,
    datasets: list[str],
    seeds: list[int],
) -> list[ChainSpec]:
    """Expand a recipe into Monarch chain specs."""
    from graphids.config.jsonnet import render
    from graphids.orchestrate.planning import enumerate_assets, expand_recipe_configs

    raw = render(Path(recipe_path))
    expanded = expand_recipe_configs(raw)
    configs = enumerate_assets(expanded)

    log.info("sweep_plan", num_assets=len(configs), datasets=datasets, seeds=seeds)

    chains = _decompose_dag(configs)

    specs: list[ChainSpec] = []
    for idx, chain in enumerate(chains):
        leaf = chain[-1]
        label = f"{leaf.stage}_{leaf.identity or leaf.asset_name}"
        for ds in datasets:
            for seed in seeds:
                specs.append(
                    ChainSpec(
                        chain_id=f"chain_{idx}_{label}_{ds}_s{seed}",
                        stages=chain,
                        dataset=ds,
                        seed=seed,
                    )
                )

    log.info("sweep_chains", num_chains=len(specs), num_unique_dags=len(chains))
    return specs


def _decompose_dag(configs: list[StageConfig]) -> list[list[StageConfig]]:
    """Decompose a DAG of StageConfigs into maximal root-to-leaf chains."""
    by_name: dict[str, StageConfig] = {c.asset_name: c for c in configs}
    has_downstream: set[str] = set()
    for cfg in configs:
        for upstream in cfg.upstream_asset_names:
            has_downstream.add(upstream)

    leaves = [c for c in configs if c.asset_name not in has_downstream]
    if not leaves:
        return [[c] for c in configs]

    def _walk_to_root(leaf: StageConfig) -> list[StageConfig]:
        visited: set[str] = set()
        order: list[StageConfig] = []

        def _visit(cfg: StageConfig) -> None:
            if cfg.asset_name in visited:
                return
            visited.add(cfg.asset_name)
            for upstream_name in cfg.upstream_asset_names:
                upstream = by_name.get(upstream_name)
                if upstream is not None:
                    _visit(upstream)
            order.append(cfg)

        _visit(leaf)
        return order

    return [_walk_to_root(leaf) for leaf in leaves]


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
    chain: ChainSpec, max_retries: int = 2, lake_root: str = "", job_spec_override: JobSpec | None = None
) -> dict[str, str]:
    """Run one chain in a single SLURM allocation. Returns {stage: ckpt_path}."""
    from monarch.config import configure  # type: ignore[import-not-found]

    from graphids.config.constants import LAKE_ROOT
    from graphids.orchestrate._setup import bootstrap_staging
    from graphids.orchestrate.actors import PipelineActor

    lake_root = lake_root or LAKE_ROOT
    spec = job_spec_override or chain_job_spec(
        chain.stages, job_name=f"graphids-{chain.chain_id}", dataset=chain.dataset
    )
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
            bootstrap=lambda: bootstrap_staging(chain.dataset),
        )
        actor = proc_mesh.spawn("pipeline", PipelineActor, lake_root=lake_root)

        checkpoints: dict[str, str] = {}
        for cfg in chain.stages:
            upstream = {n: checkpoints[n] for n in cfg.upstream_asset_names if n in checkpoints}
            call = lambda c=cfg, u=upstream: actor.train_stage.call_one(  # noqa: E731
                stage_config=c.model_dump(),
                dataset=chain.dataset,
                seed=chain.seed,
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

        for cfg in chain.stages:
            upstream = {n: checkpoints[n] for n in cfg.upstream_asset_names if n in checkpoints}
            try:
                actor.eval_stage.call_one(
                    stage_config=cfg.model_dump(),
                    dataset=chain.dataset,
                    seed=chain.seed,
                    upstream_ckpts=upstream,
                ).get()
            except Exception as exc:
                log.warning("eval_failed", stage=cfg.stage, error=str(exc))

        return {cfg.stage: checkpoints.get(cfg.asset_name, "") for cfg in chain.stages}
    finally:
        try:
            job.kill()
        except Exception:
            pass


def run_sweep(config: SweepConfig) -> dict[str, dict[str, str] | str]:
    """Run all recipe chains in parallel. Returns {chain_id: checkpoints | error_str}."""
    chains = plan_chains(config.recipe_path, config.datasets, config.seeds)
    results: dict[str, dict[str, str] | str] = {}

    with ThreadPoolExecutor(max_workers=config.max_concurrent or len(chains) or 1) as pool:
        futures = {
            pool.submit(run_chain, c, max_retries=config.max_retries, lake_root=config.lake_root): c
            for c in chains
        }
        for future in as_completed(futures):
            chain = futures[future]
            try:
                results[chain.chain_id] = future.result()
            except Exception as exc:
                results[chain.chain_id] = str(exc)

    return results
