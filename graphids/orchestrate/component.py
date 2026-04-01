"""Dagster Component: SLURM-based ML training pipeline.

Assets represent trained model checkpoints. AssetSpecs describe identity
(key, deps, tags, kinds). Multi-asset functions define materialization
behavior (submit to SLURM, return checkpoint path). IOManager handles
checkpoint path handoff between dependent stages.

NO torch/Lightning imports at definition time.
"""

from __future__ import annotations

import os
from pathlib import Path

import dagster as dg

from graphids.config import (
    CONFIG_DIR,
    DAGSTER_IO_DIR_TEMPLATE,
    LAKE_ROOT,
    PIPELINE_YAML,
    dataset_names,
    expand_recipe_configs,
)
from graphids.config.yaml_utils import read_yaml
from graphids.core.contracts import TrainingSpec
from graphids.orchestrate.analysis import supports_analysis
from graphids.orchestrate.assets import make_analysis_asset, make_training_asset
from graphids.orchestrate.checks import make_analysis_checks, make_checkpoint_checks
from graphids.orchestrate.planning import StageConfig, enumerate_assets
from graphids.orchestrate.resources import ResourceSpec
from graphids.orchestrate.slurm import SlurmJobClient, SubprocessSlurmJobClient

RECIPES_DIR = CONFIG_DIR / "recipes"
RECIPE_PATH = Path(os.environ.get("KD_GAT_RECIPE", RECIPES_DIR / "ablation.yaml"))


class SlurmTrainingResource(dg.ConfigurableResource):
    """Submits training jobs to SLURM and polls for completion."""

    dry_run: bool = False
    poll_interval: int = 60
    max_unknown: int = 5

    def _client(self) -> SlurmJobClient:
        return SubprocessSlurmJobClient(
            dry_run=self.dry_run,
            poll_interval=self.poll_interval,
            max_unknown=self.max_unknown,
        )

    def submit_and_wait(
        self,
        training_spec: TrainingSpec,
        resources: ResourceSpec,
        job_name: str,
        on_state=None,
    ) -> tuple[str, int]:
        """Submit SLURM job and poll. Returns (state, job_id)."""
        return self._client().run_training_job(
            training_spec=training_spec,
            resources=resources,
            job_name=job_name,
            on_state=on_state,
        )


# ---------------------------------------------------------------------------
# Component — assembles specs + behavior + resources into Definitions
# ---------------------------------------------------------------------------


class SlurmTrainingComponent(dg.Component, dg.Model, dg.Resolvable):
    """SLURM training pipeline.

    Reads compact config topology from graphids.config plus the selected recipe,
    then generates tagged assets with IOManager checkpoint handoff.
    """

    dry_run: bool = False
    poll_interval: int = 60
    max_concurrent: int = 0  # 0 = no limit (SLURM handles throttling)

    def build_defs(self, context: dg.ComponentLoadContext) -> dg.Definitions:
        recipe = expand_recipe_configs(read_yaml(RECIPE_PATH))

        # 1. Enumerate training configs (pure data)
        stage_configs = enumerate_assets(PIPELINE_YAML, recipe)

        # 2. Partitions
        datasets = dataset_names()
        seeds = [str(s) for s in recipe.get("sweep", {}).get("seeds", [42])]
        partitions = dg.MultiPartitionsDefinition({
            "dataset": dg.StaticPartitionsDefinition(datasets),
            "seed": dg.StaticPartitionsDefinition(seeds),
        })

        lake_root = os.environ.get("KD_GAT_LAKE_ROOT", LAKE_ROOT)
        user = os.environ.get("USER", "unknown")

        # 3. Build assets — one @asset per StageConfig, IOManager wires checkpoint paths
        assets = [make_training_asset(cfg, partitions, lake_root, user) for cfg in stage_configs]
        analysis_sources = [cfg for cfg in stage_configs if supports_analysis(cfg.model_type)]
        analysis_assets = [make_analysis_asset(cfg, partitions) for cfg in analysis_sources]
        assets.extend(analysis_assets)

        # 4. Build asset checks
        cfg_lookup = {cfg.asset_name: cfg for cfg in stage_configs}
        checks = make_checkpoint_checks(cfg_lookup, partitions, lake_root, user)
        analysis_lookup = {cfg.asset_name: cfg for cfg in analysis_sources}
        checks.extend(make_analysis_checks(analysis_lookup, partitions))

        # 6. Executor: multiprocess so independent assets run in parallel.
        # Each worker just does sbatch + poll (sleep loop), so concurrency is cheap.
        executor_cfg = {"max_concurrent": self.max_concurrent} if self.max_concurrent > 0 else {}
        executor = dg.multiprocess_executor.configured(executor_cfg)

        # 7. Resources
        return dg.Definitions(
            assets=assets,
            asset_checks=checks,
            resources={
                "slurm": SlurmTrainingResource(
                    dry_run=self.dry_run,
                    poll_interval=self.poll_interval,
                ),
                "io_manager": dg.fs_io_manager.configured(
                    {"base_dir": DAGSTER_IO_DIR_TEMPLATE.replace("{lake_root}", lake_root)}
                ),
            },
            executor=executor,
        )
