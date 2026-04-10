"""Top-level pipeline driver + ``PipelineConfig`` schema.

`run_pipeline` is the only module that sees every layer of the
orchestrate stack: plan → allocate → spawn → chain → analyze →
teardown. It composes only its Layer N+1 peers (``allocate``,
``chain``, ``analyze``) and owns the SlurmJob lifecycle's ``finally``
block.

The CLI-facing ``PipelineConfig`` schema and the planner bridge
(``build_pipeline_stages``) live here too so there's a single home
for "how do I run a pipeline" — the input spec, the planner glue,
and the driver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, Literal  # noqa: F401 (resolved by model_rebuild)

from pydantic import (  # noqa: F401 (AfterValidator resolved by model_rebuild)
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
)

from graphids._otel import get_logger
from graphids.config.constants import (  # noqa: F401 (resolved by model_rebuild)
    PIPELINE_DEFAULTS,
    VALID_FUSION_METHODS,
    VALID_SCALES,
)
from graphids.config.topology import TOPOLOGY  # noqa: F401 (resolved by model_rebuild)
from graphids.orchestrate.allocate import (
    JobSpec,
    build_slurm_job,
    configure_monarch,
    spawn_actor,
)
from graphids.orchestrate.analyze import analyze
from graphids.orchestrate.chain import ChainResult, run_chain
from graphids.orchestrate.planning import StageConfig
from graphids.orchestrate.planning.recipes import (  # noqa: F401 (resolved by model_rebuild)
    TrainingRunConfig,
    check_in,
)

log = get_logger(__name__)

_D = PIPELINE_DEFAULTS


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


# ---------------------------------------------------------------------------
# PipelineConfig — CLI-facing input schema
# ---------------------------------------------------------------------------


class PipelineConfig(BaseModel):
    """What to run in a single Monarch allocation (``monarch-run`` CLI)."""

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
        """Convert CLI fields to a planner-ready ``TrainingRunConfig``."""
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
# Planner bridge
# ---------------------------------------------------------------------------


def build_pipeline_stages(config: PipelineConfig) -> list[StageConfig]:
    """``PipelineConfig → list[StageConfig]`` via the planner.

    Also used by the CLI dry-run path to preview what would be run.
    """
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


# ---------------------------------------------------------------------------
# Pipeline composition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineResult:
    """Composite result of a full pipeline run."""

    chain: ChainResult
    analyzed_assets: list[str]

    def checkpoints_by_stage(self) -> dict[str, str]:
        return self.chain.ckpts_by_stage()


def run_pipeline(config: PipelineConfig, job_spec: JobSpec) -> PipelineResult:
    """Run a full pipeline in one SLURM allocation.

    Orchestrates: plan → allocate → spawn → chain → analyze → teardown.
    """
    from graphids.config.constants import LAKE_ROOT

    stages = build_pipeline_stages(config)
    lake_root = config.lake_root or LAKE_ROOT

    configure_monarch()
    job = build_slurm_job(job_spec)
    try:
        actor = spawn_actor(job, gpus_per_node=job_spec.gpus_per_node, lake_root=lake_root)
        chain = run_chain(
            actor, stages,
            dataset=config.dataset, seed=config.seed,
            max_retries=config.max_retries,
        )
        analyzed = analyze(actor, stages, chain, dataset=config.dataset, seed=config.seed)
        return PipelineResult(chain=chain, analyzed_assets=analyzed)
    finally:
        try:
            job.kill()
        except Exception as exc:
            log.warning("job_kill_failed", error=str(exc))
