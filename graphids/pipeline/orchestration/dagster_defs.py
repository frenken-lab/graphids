"""Dagster asset definitions for KD-GAT pipeline orchestration.

Assets are dynamically generated from STAGE_DEPENDENCIES + PipelineConfig.variants
+ resources.yaml. Adding a variant or stage = YAML edit, no Python changes.

Entry point: ``dagster dev -m graphids.pipeline.orchestration.dagster_defs``

Asset dependency graph (3 default variants):

    preprocess ─────────────────────────────────────────────
      /              |                    \
  vgae_large_ae   vgae_small_ae_kd     vgae_small_ae
      |              |                    |
  gat_large_cur   gat_small_cur_kd     gat_small_cur
      |              |                    |
  dqn_large_fus   dqn_small_fus_kd     dqn_small_fus
      |              |                    |
  eval_large      eval_small_kd        eval_small
      \\             |                   /
       ─────────── hf_push ────────────
"""

import logging
import subprocess
import sys
from functools import cached_property

import dagster as dg

from graphids.config import (
    DEFAULT_SEEDS,
    PROJECT_ROOT,
    STAGE_DEPENDENCIES,
    STAGE_MODEL_MAP,
    get_datasets,
)

from .dagster_resources import (
    clear_retry_state,
    load_retry_state,
    save_retry_state,
)
from .pipes_slurm import (
    FAILURE_REACTIONS,
    PipesSlurmClient,
    SlurmJobFailed,
    get_resources,
    scale_resources,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dagster resource (wraps PipesSlurmClient)
# ---------------------------------------------------------------------------


class PipesSlurmResource(dg.ConfigurableResource):
    """Dagster resource providing SLURM job submission via PipesSlurmClient."""

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


# ---------------------------------------------------------------------------
# Partitions
# ---------------------------------------------------------------------------

pipeline_partitions = dg.MultiPartitionsDefinition(
    {
        "dataset": dg.StaticPartitionsDefinition(get_datasets()),
        "seed": dg.StaticPartitionsDefinition([str(s) for s in DEFAULT_SEEDS]),
    }
)


# ---------------------------------------------------------------------------
# Naming helpers
# ---------------------------------------------------------------------------


def _extract_partition(context: dg.AssetExecutionContext) -> tuple[str, int]:
    """Extract (dataset, seed) from multi-partition key, with defaults."""
    if context.has_partition_key:
        mp_key = context.partition_key
        if isinstance(mp_key, dg.MultiPartitionKey):
            return mp_key.keys_by_dimension["dataset"], int(mp_key.keys_by_dimension["seed"])
        # Fallback for string partition key (e.g. in tests)
        return str(mp_key), 42
    return "hcrl_sa", 42


def _asset_name(resource_model: str, scale: str, stage: str, aux: str = "") -> str:
    """Generate asset name: {resource_model}_{scale}_{stage}[_{aux}]."""
    name = f"{resource_model}_{scale}_{stage}"
    if aux:
        name += f"_{aux}"
    return name


# ---------------------------------------------------------------------------
# Asset factory
# ---------------------------------------------------------------------------


def _max_retries() -> int:
    """Max retries across all failure reactions."""
    return max(
        (r.get("max_retries", 0) for r in FAILURE_REACTIONS.values()),
        default=0,
    )


def _make_stage_asset(
    name: str,
    stage: str,
    cli_model: str,
    resource_model: str,
    scale: str,
    dep_names: list[str],
    auxiliaries: str = "none",
):
    """Factory: create a @dg.asset that submits one SLURM job.

    Parameters
    ----------
    cli_model : str
        Model name passed to ``--model`` in the CLI (e.g. "vgae", "gat", "dqn").
        For evaluation, this is "vgae" (evaluates all models).
    resource_model : str
        Model key for resource lookup in resources.yaml (e.g. "eval" for evaluation).
    """
    max_ret = _max_retries()

    @dg.asset(
        name=name,
        deps=[dg.AssetKey(d) for d in dep_names],
        retry_policy=dg.RetryPolicy(max_retries=max_ret),
        partitions_def=pipeline_partitions,
        metadata={
            "cli_model": cli_model,
            "resource_model": resource_model,
            "scale": scale,
            "stage": stage,
            "auxiliaries": auxiliaries,
        },
    )
    def _asset(context: dg.AssetExecutionContext, slurm: PipesSlurmResource):
        dataset, seed = _extract_partition(context)
        asset_key_str = f"{name}__{dataset}__seed{seed}"

        # Check for retry state from a previous failed attempt
        retry_state = load_retry_state(asset_key_str)
        resources = get_resources(resource_model, scale, stage)

        ckpt_path = None
        if retry_state:
            resources = scale_resources(resources, retry_state["reason"])
            ckpt_path = retry_state.get("ckpt_path")
            if retry_state.get("node") and FAILURE_REACTIONS.get(retry_state["reason"], {}).get(
                "exclude_node"
            ):
                resources = resources.model_copy(update={"exclude_nodes": retry_state["node"]})
            context.log.info(
                "Retrying with scaled resources: %s (reason: %s)",
                resources,
                retry_state["reason"],
            )

        try:
            result = slurm.client.run(
                stage=stage,
                model=cli_model,
                scale=scale,
                dataset=dataset,
                resources=resources,
                seed=seed,
                auxiliaries=auxiliaries,
                ckpt_path=ckpt_path,
            )
            clear_retry_state(asset_key_str)
            return dg.MaterializeResult(metadata=result)

        except SlurmJobFailed as e:
            save_retry_state(
                asset_key_str,
                reason=e.reason,
                node=e.node,
                ckpt_path=e.ckpt_path,
            )
            reaction = FAILURE_REACTIONS.get(e.reason, {})
            if reaction.get("max_retries", 0) > 0:
                raise dg.RetryRequested(
                    max_retries=reaction["max_retries"],
                    seconds_to_wait=10,
                ) from e
            raise dg.Failure(
                description=f"SLURM job failed: {e.reason}",
                metadata=e.metadata,
            ) from e

    _asset.__name__ = name
    _asset.__qualname__ = name
    return _asset


def _make_hf_push_asset(eval_dep_names: list[str]):
    """Create the hf_push asset that pushes experiment data to HF Dataset."""

    @dg.asset(
        name="hf_push",
        deps=[dg.AssetKey(d) for d in eval_dep_names],
        partitions_def=pipeline_partitions,
        metadata={"stage": "hf_push"},
    )
    def _asset(context: dg.AssetExecutionContext):
        dataset, _seed = _extract_partition(context)
        context.log.info("Pushing experiment data to HF Dataset for %s", dataset)
        result = subprocess.run(
            [sys.executable, "scripts/data/push_experiments_to_hf.py"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        if result.returncode != 0:
            context.log.warning("HF push failed (non-fatal): %s", result.stderr[:500])

        # Orphan cleanup (informational, from former ray_slurm.sbatch)
        subprocess.run(
            ["bash", "scripts/data/cleanup_orphans.sh", "--dry-run"],
            capture_output=True,
            cwd=str(PROJECT_ROOT),
        )

        return dg.MaterializeResult(
            metadata={
                "returncode": result.returncode,
                "dataset": dataset,
            }
        )

    return _asset


def _make_rebuild_catalog_asset(hf_push_dep: str = "hf_push"):
    """Create the rebuild_catalog asset that rebuilds the DuckDB catalog."""

    @dg.asset(
        name="rebuild_catalog",
        deps=[dg.AssetKey(hf_push_dep)],
        partitions_def=pipeline_partitions,
        metadata={"stage": "rebuild_catalog"},
    )
    def _asset(context: dg.AssetExecutionContext):
        import os

        lake_root = os.environ.get("KD_GAT_LAKE_ROOT")
        if not lake_root:
            context.log.info("KD_GAT_LAKE_ROOT not set — skipping catalog rebuild")
            return dg.MaterializeResult(metadata={"skipped": True})

        from pathlib import Path

        from graphids.lake.catalog import rebuild_catalog

        catalog_path = rebuild_catalog(Path(lake_root))
        context.log.info("Catalog rebuilt: %s", catalog_path)
        return dg.MaterializeResult(metadata={"catalog_path": str(catalog_path)})

    return _asset


# ---------------------------------------------------------------------------
# DAG builder (planner)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Shared DAG topology (single source of truth)
# ---------------------------------------------------------------------------


from dataclasses import dataclass


@dataclass(frozen=True)
class DagNode:
    """One node in the pipeline DAG."""

    stage: str
    cli_model: str
    resource_model: str
    scale: str
    auxiliaries: str
    deps: frozenset[str]


def build_dag_topology() -> dict[str, DagNode]:
    """Build the pipeline DAG from PipelineConfig.variants + STAGE_DEPENDENCIES.

    Returns {asset_name: DagNode} with preprocess + all variant stages.
    Both build_dagster_assets() and fire_and_forget() call this.
    """
    from graphids.config import resolve

    cfg = resolve("vgae", "large")
    nodes: dict[str, DagNode] = {}

    # Preprocess (CPU, shared, no deps)
    nodes["preprocess"] = DagNode(
        stage="preprocess",
        cli_model="preprocess",
        resource_model="preprocess",
        scale="",
        auxiliaries="none",
        deps=frozenset(),
    )

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
                    fusion_model = STAGE_MODEL_MAP["fusion"]
                    dep_names.add(_asset_name(fusion_model, variant.scale, "fusion", aux))

            if variant.needs_teacher and stage in ("autoencoder", "curriculum"):
                teacher_nm = _asset_name(resource_model, "large", stage)
                if teacher_nm != asset_nm:
                    dep_names.add(teacher_nm)

            nodes[asset_nm] = DagNode(
                stage=stage,
                cli_model=cli_model,
                resource_model=resource_model,
                scale=variant.scale,
                auxiliaries=variant.auxiliaries,
                deps=frozenset(dep_names),
            )

    return nodes


# ---------------------------------------------------------------------------
# Dagster asset builder
# ---------------------------------------------------------------------------


def build_dagster_assets(datasets: list[str] | None = None) -> list:
    """Build Dagster asset definitions from the shared DAG topology.

    Adding a new variant or stage = YAML edit, no Python changes here.
    """
    dag = build_dag_topology()
    assets = []
    variant_asset_names: dict[str, str] = {}  # tracks eval assets for hf_push

    for name, node in dag.items():
        assets.append(
            _make_stage_asset(
                name=name,
                stage=node.stage,
                cli_model=node.cli_model,
                resource_model=node.resource_model,
                scale=node.scale,
                dep_names=list(node.deps),
                auxiliaries=node.auxiliaries,
            )
        )
        if node.stage == "evaluation":
            variant_asset_names[name] = name

    # HF push asset (depends on all evaluation assets)
    assets.append(_make_hf_push_asset(list(variant_asset_names.keys())))

    # Catalog rebuild asset (depends on hf_push)
    assets.append(_make_rebuild_catalog_asset())

    return assets


# ---------------------------------------------------------------------------
# Fire-and-forget mode
# ---------------------------------------------------------------------------


def fire_and_forget(
    dataset: str,
    seeds: list[int] | None = None,
    dry_run: bool = False,
) -> dict[str, str]:
    """Submit all pipeline jobs with ``--dependency=afterok`` chains.

    No polling — SLURM handles execution ordering. Returns a dict of
    ``{asset_name: job_id}`` for all submitted jobs.

    Uses build_dag_topology() for identical topology and resource profiles.
    """
    import graphlib

    from graphids.config import resolve

    cfg = resolve("vgae", "large")
    client = PipesSlurmClient(dry_run=dry_run)
    seed_list = seeds or [cfg.seed]
    dag = build_dag_topology()

    all_job_ids: dict[str, str] = {}

    for seed in seed_list:
        # Build edges dict for topological sort
        edges = {name: set(node.deps) for name, node in dag.items()}

        sorter = graphlib.TopologicalSorter(edges)
        topo_order = list(sorter.static_order())

        job_ids: dict[str, str] = {}

        for asset_nm in topo_order:
            node = dag[asset_nm]
            resources = get_resources(node.resource_model, node.scale, node.stage)

            parent_ids = [job_ids[dep] for dep in node.deps if dep in job_ids]
            dep_str = ",".join(parent_ids) if parent_ids else None

            job_id = client.submit_no_poll(
                stage=node.stage,
                model=node.cli_model,
                scale=node.scale,
                dataset=dataset,
                resources=resources,
                seed=seed,
                auxiliaries=node.auxiliaries,
                dependency_job_id=dep_str,
            )
            job_ids[asset_nm] = job_id
            log.info("  %s → job %s (deps: %s)", asset_nm, job_id, dep_str or "none")

        all_job_ids.update({f"{k}__seed{seed}": v for k, v in job_ids.items()})
        log.info("Fire-and-forget: submitted %d jobs for %s (seed=%d)", len(job_ids), dataset, seed)

    return all_job_ids


# ---------------------------------------------------------------------------
# Definitions entry point
# ---------------------------------------------------------------------------

_assets = build_dagster_assets()

defs = dg.Definitions(
    assets=_assets,
    resources={"slurm": PipesSlurmResource()},
)
