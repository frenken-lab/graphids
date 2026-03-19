"""Dagster asset definitions for KD-GAT pipeline orchestration.

Assets dynamically generated from STAGE_DEPENDENCIES + PipelineConfig.variants
+ resources.yaml. Adding a variant or stage = YAML edit, no Python changes.

Entry point: ``dagster dev -m graphids.pipeline.orchestration.dagster_defs``
"""

import logging
import subprocess
import sys
from dataclasses import dataclass
from functools import cached_property

import dagster as dg

from graphids.config import (
    DEFAULT_SEEDS,
    PROJECT_ROOT,
    STAGE_DEPENDENCIES,
    STAGE_MODEL_MAP,
    get_datasets,
)
from .slurm_client import (
    FAILURE_REACTIONS,
    PipesSlurmClient,
    SlurmJobFailed,
    clear_retry_state,
    get_resources,
    load_retry_state,
    save_retry_state,
    scale_resources,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Resource + partitions
# ---------------------------------------------------------------------------


class PipesSlurmResource(dg.ConfigurableResource):
    """Dagster resource wrapping PipesSlurmClient."""

    project_root: str = str(PROJECT_ROOT)
    poll_interval: int = 30
    dry_run: bool = False

    @cached_property
    def client(self) -> PipesSlurmClient:
        return PipesSlurmClient(
            project_root=self.project_root,
            poll_interval=self.poll_interval,
            dry_run=self.dry_run,
        )


pipeline_partitions = dg.MultiPartitionsDefinition({
    "dataset": dg.StaticPartitionsDefinition(get_datasets()),
    "seed": dg.StaticPartitionsDefinition([str(s) for s in DEFAULT_SEEDS]),
})


def _extract_partition(context: dg.AssetExecutionContext) -> tuple[str, int]:
    """Extract (dataset, seed) from multi-partition key."""
    if context.has_partition_key:
        mp_key = context.partition_key
        if isinstance(mp_key, dg.MultiPartitionKey):
            return mp_key.keys_by_dimension["dataset"], int(mp_key.keys_by_dimension["seed"])
        return str(mp_key), 42
    return "hcrl_sa", 42


def _asset_name(resource_model: str, scale: str, stage: str, aux: str = "") -> str:
    name = f"{resource_model}_{scale}_{stage}"
    return f"{name}_{aux}" if aux else name


# ---------------------------------------------------------------------------
# Asset factories
# ---------------------------------------------------------------------------

_MAX_RETRIES = max((r.get("max_retries", 0) for r in FAILURE_REACTIONS.values()), default=0)


def _make_stage_asset(
    name: str, stage: str, cli_model: str, resource_model: str,
    scale: str, dep_names: list[str], auxiliaries: str = "none",
):
    """Factory: create a @dg.asset that submits one SLURM job."""

    @dg.asset(
        name=name,
        deps=[dg.AssetKey(d) for d in dep_names],
        retry_policy=dg.RetryPolicy(max_retries=_MAX_RETRIES),
        partitions_def=pipeline_partitions,
        metadata={"cli_model": cli_model, "resource_model": resource_model,
                  "scale": scale, "stage": stage, "auxiliaries": auxiliaries},
    )
    def _asset(context: dg.AssetExecutionContext, slurm: PipesSlurmResource):
        dataset, seed = _extract_partition(context)
        asset_key_str = f"{name}__{dataset}__seed{seed}"

        retry_state = load_retry_state(asset_key_str)
        resources = get_resources(resource_model, scale, stage)

        ckpt_path = None
        if retry_state:
            resources = scale_resources(resources, retry_state["reason"])
            ckpt_path = retry_state.get("ckpt_path")
            if retry_state.get("node") and FAILURE_REACTIONS.get(
                retry_state["reason"], {}
            ).get("exclude_node"):
                resources = resources.model_copy(update={"exclude_nodes": retry_state["node"]})

        try:
            result = slurm.client.run(
                stage=stage, model=cli_model, scale=scale, dataset=dataset,
                resources=resources, seed=seed, auxiliaries=auxiliaries, ckpt_path=ckpt_path,
            )
            clear_retry_state(asset_key_str)
            return dg.MaterializeResult(metadata=result)
        except SlurmJobFailed as e:
            save_retry_state(asset_key_str, reason=e.reason, node=e.node, ckpt_path=e.ckpt_path)
            reaction = FAILURE_REACTIONS.get(e.reason, {})
            if reaction.get("max_retries", 0) > 0:
                raise dg.RetryRequested(max_retries=reaction["max_retries"], seconds_to_wait=10) from e
            raise dg.Failure(description=f"SLURM job failed: {e.reason}", metadata=e.metadata) from e

    _asset.__name__ = name
    _asset.__qualname__ = name
    return _asset


def _make_hf_push_asset(eval_dep_names: list[str]):
    @dg.asset(
        name="hf_push",
        deps=[dg.AssetKey(d) for d in eval_dep_names],
        partitions_def=pipeline_partitions,
    )
    def _asset(context: dg.AssetExecutionContext):
        dataset, _seed = _extract_partition(context)
        result = subprocess.run(
            [sys.executable, "scripts/data/push_experiments_to_hf.py"],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT),
        )
        if result.returncode != 0:
            context.log.warning("HF push failed (non-fatal): %s", result.stderr[:500])
        return dg.MaterializeResult(metadata={"returncode": result.returncode, "dataset": dataset})

    return _asset


def _make_rebuild_catalog_asset():
    @dg.asset(
        name="rebuild_catalog",
        deps=[dg.AssetKey("hf_push")],
        partitions_def=pipeline_partitions,
    )
    def _asset(context: dg.AssetExecutionContext):
        import os
        lake_root = os.environ.get("KD_GAT_LAKE_ROOT")
        if not lake_root:
            return dg.MaterializeResult(metadata={"skipped": True})
        from pathlib import Path
        from graphids.pipeline.catalog import rebuild_catalog
        catalog_path = rebuild_catalog(Path(lake_root))
        return dg.MaterializeResult(metadata={"catalog_path": str(catalog_path)})

    return _asset


# ---------------------------------------------------------------------------
# DAG topology (shared by build_dagster_assets + fire_and_forget)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DagNode:
    stage: str
    cli_model: str
    resource_model: str
    scale: str
    auxiliaries: str
    deps: frozenset[str]


def build_dag_topology() -> dict[str, DagNode]:
    """Build pipeline DAG from PipelineConfig.variants + STAGE_DEPENDENCIES."""
    from graphids.config import resolve

    cfg = resolve("vgae", "large")
    nodes: dict[str, DagNode] = {
        "preprocess": DagNode("preprocess", "preprocess", "preprocess", "", "none", frozenset()),
    }

    for variant in cfg.variants:
        aux = variant.auxiliaries if variant.auxiliaries != "none" else ""
        for stage in variant.stages:
            resource_model = STAGE_MODEL_MAP[stage]
            cli_model = "vgae" if stage == "evaluation" else resource_model
            asset_nm = _asset_name(resource_model, variant.scale, stage, aux)

            dep_names: set[str] = set()
            if stage == "autoencoder":
                dep_names.add("preprocess")
            else:
                for dep_model_type, dep_stage in STAGE_DEPENDENCIES.get(stage, []):
                    dep_names.add(_asset_name(dep_model_type, variant.scale, dep_stage, aux))
                if stage == "evaluation" and not dep_names:
                    dep_names.add(_asset_name(STAGE_MODEL_MAP["fusion"], variant.scale, "fusion", aux))

            if variant.needs_teacher and stage in ("autoencoder", "curriculum"):
                teacher_nm = _asset_name(resource_model, "large", stage)
                if teacher_nm != asset_nm:
                    dep_names.add(teacher_nm)

            nodes[asset_nm] = DagNode(stage, cli_model, resource_model, variant.scale,
                                      variant.auxiliaries, frozenset(dep_names))
    return nodes


def build_dagster_assets() -> list:
    """Build Dagster asset definitions from DAG topology."""
    dag = build_dag_topology()
    assets = []
    eval_names = []

    for name, node in dag.items():
        assets.append(_make_stage_asset(
            name=name, stage=node.stage, cli_model=node.cli_model,
            resource_model=node.resource_model, scale=node.scale,
            dep_names=list(node.deps), auxiliaries=node.auxiliaries,
        ))
        if node.stage == "evaluation":
            eval_names.append(name)

    assets.append(_make_hf_push_asset(eval_names))
    assets.append(_make_rebuild_catalog_asset())
    return assets


# ---------------------------------------------------------------------------
# Fire-and-forget (zero-daemon fallback)
# ---------------------------------------------------------------------------


def fire_and_forget(
    dataset: str, seeds: list[int] | None = None, dry_run: bool = False,
) -> dict[str, str]:
    """Submit all jobs with --dependency=afterok chains. No polling."""
    import graphlib
    from graphids.config import resolve

    client = PipesSlurmClient(dry_run=dry_run)
    seed_list = seeds or [resolve("vgae", "large").seed]
    dag = build_dag_topology()
    topo_order = list(graphlib.TopologicalSorter(
        {name: set(node.deps) for name, node in dag.items()}
    ).static_order())

    all_job_ids: dict[str, str] = {}
    for seed in seed_list:
        job_ids: dict[str, str] = {}
        for asset_nm in topo_order:
            node = dag[asset_nm]
            resources = get_resources(node.resource_model, node.scale, node.stage)
            parent_ids = [job_ids[dep] for dep in node.deps if dep in job_ids]
            dep_str = ",".join(parent_ids) if parent_ids else None

            job_ids[asset_nm] = client.submit_no_poll(
                stage=node.stage, model=node.cli_model, scale=node.scale,
                dataset=dataset, resources=resources, seed=seed,
                auxiliaries=node.auxiliaries, dependency_job_id=dep_str,
            )
        all_job_ids.update({f"{k}__seed{seed}": v for k, v in job_ids.items()})
    return all_job_ids


# ---------------------------------------------------------------------------
# Definitions entry point
# ---------------------------------------------------------------------------

defs = dg.Definitions(
    assets=build_dagster_assets(),
    resources={"slurm": PipesSlurmResource()},
)
