"""Dagster asset check factory helpers for orchestrated assets."""

from __future__ import annotations

import os
from pathlib import Path

import dagster as dg

from graphids.config import LAKE_ROOT
from graphids.orchestrate.analysis import ANALYSIS_MANIFEST_NAME, build_analysis_spec, output_status
from graphids.orchestrate.execution import artifact_paths
from graphids.orchestrate.planning import StageConfig


def _checkpoint_check_result(
    *,
    context,
    cfg: StageConfig,
    lake_root: str,
    user: str,
) -> dg.AssetCheckResult:
    dataset = context.partition_key.keys_by_dimension["dataset"]
    seed = int(context.partition_key.keys_by_dimension["seed"])
    _, _, ckpt = artifact_paths(
        cfg,
        lake_root=lake_root,
        user=user,
        dataset=dataset,
        seed=seed,
    )
    return dg.AssetCheckResult(
        passed=ckpt.exists(),
        metadata={"path": dg.MetadataValue.path(str(ckpt))},
    )


def _analysis_check_result(*, context, cfg: StageConfig) -> dg.AssetCheckResult:
    dataset = context.partition_key.keys_by_dimension["dataset"]
    seed = int(context.partition_key.keys_by_dimension["seed"])
    _, _, ckpt = artifact_paths(
        cfg,
        lake_root=os.environ.get("KD_GAT_LAKE_ROOT", LAKE_ROOT),
        user=os.environ.get("USER", "unknown"),
        dataset=dataset,
        seed=seed,
    )
    spec = build_analysis_spec(
        cfg=cfg,
        dataset=dataset,
        seed=seed,
        ckpt_path=str(ckpt),
    )
    output_dir = Path(spec.output_dir)
    manifest_path = output_dir / ANALYSIS_MANIFEST_NAME
    expected, existing = output_status(spec)
    return dg.AssetCheckResult(
        passed=manifest_path.exists() and len(existing) == len(expected),
        metadata={
            "manifest": dg.MetadataValue.path(str(manifest_path)),
            "expected": len(expected),
            "existing": len(existing),
        },
    )


def make_checkpoint_checks(
    cfg_lookup: dict[str, StageConfig],
    partitions_def: dg.MultiPartitionsDefinition,
    lake_root: str,
    user: str,
) -> list[dg.AssetChecksDefinition]:
    """One checkpoint existence check per training asset."""
    checks = []
    for asset_name, cfg in cfg_lookup.items():

        def _make_check(name: str, c: StageConfig):
            @dg.asset_check(
                asset=dg.AssetKey(name),
                name=f"checkpoint_exists_{name}",
                blocking=True,
                description=f"Verify checkpoint for {name}",
                partitions_def=partitions_def,
            )
            def _check(context) -> dg.AssetCheckResult:
                return _checkpoint_check_result(
                    context=context,
                    cfg=c,
                    lake_root=lake_root,
                    user=user,
                )

            return _check

        checks.append(_make_check(asset_name, cfg))
    return checks


def make_analysis_checks(
    cfg_lookup: dict[str, StageConfig],
    partitions_def: dg.MultiPartitionsDefinition,
) -> list[dg.AssetChecksDefinition]:
    """Checks expected outputs exist for each analysis asset."""
    checks: list[dg.AssetChecksDefinition] = []
    for asset_name, cfg in cfg_lookup.items():
        analysis_asset_name = f"{asset_name}_analysis"

        def _make_check(name: str, c: StageConfig):
            @dg.asset_check(
                asset=dg.AssetKey(name),
                name=f"analysis_outputs_exist_{name}",
                blocking=True,
                description=f"Verify analysis outputs for {name}",
                partitions_def=partitions_def,
            )
            def _check(context) -> dg.AssetCheckResult:
                return _analysis_check_result(context=context, cfg=c)

            return _check

        checks.append(_make_check(analysis_asset_name, cfg))

    return checks
