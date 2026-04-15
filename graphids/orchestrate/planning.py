"""Single-config asset planning — Layer 1 of the orchestrate stack.

Pure planning: ``PipelineConfig`` → ``list[StageConfig]``. No torch
imports, no jsonnet subprocess at module scope, no side effects.

One entry point: ``build_pipeline_stages(cfg)`` for ``pipeline-run``.
Multi-run ablations are explicit jsonnet presets under
``configs/ablations/`` — one file per run, invoked via
``python -m graphids fit --config <file>``.
"""

from __future__ import annotations

from graphids.config.constants import PROJECT_ROOT
from graphids.config.topology import TOPOLOGY
from graphids.orchestrate.config import PipelineConfig, StageConfig

_STAGES_DIR = PROJECT_ROOT / "configs" / "stages"
_STAGE_JSONNET: dict[str, str] = {s: f"{s}.jsonnet" for s in TOPOLOGY.stages}


def resolve_jsonnet_path(stage: str) -> str:
    """Return the absolute path to the jsonnet file for a stage."""
    filename = _STAGE_JSONNET.get(stage)
    if filename is None:
        raise ValueError(
            f"No jsonnet stage file for stage={stage!r}. Known: {sorted(_STAGE_JSONNET)}"
        )
    return str(_STAGES_DIR / filename)


def build_pipeline_stages(config: PipelineConfig) -> list[StageConfig]:
    """``PipelineConfig → list[StageConfig]`` for ``pipeline-run``.

    Walks each stage's ``depends_on`` to wire upstream asset names and
    model families (one entry per distinct family — multiple deps on
    the same family collapse to the first).
    """
    training_run = config.to_training_run()
    trainer_overrides = dict(config.tla_overrides)
    stage_to_asset: dict[str, str] = {}
    stages: list[StageConfig] = []

    for stage in training_run.stages:
        if stage not in TOPOLOGY.stages:
            continue

        upstream_names: list[str] = []
        upstream_models: dict[str, str] = {}
        seen_families: set[str] = set()
        for dep in TOPOLOGY.stages[stage].depends_on:
            dep_asset = stage_to_asset.get(dep["stage"])
            if dep_asset and dep["family"] not in seen_families:
                seen_families.add(dep["family"])
                upstream_names.append(dep_asset)
                upstream_models[dep_asset] = dep["family"]

        cfg = StageConfig.for_stage(
            stage,
            training_run,
            upstream_names=upstream_names,
            upstream_models=upstream_models,
            trainer_overrides=trainer_overrides,
        )
        stage_to_asset[stage] = cfg.asset_name
        stages.append(cfg)

    stage_order = {s: i for i, s in enumerate(config.stages)}
    stages.sort(key=lambda c: stage_order.get(c.stage, 99))
    return stages
