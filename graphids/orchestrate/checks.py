"""Dagster asset check factory helpers for orchestrated assets."""

from __future__ import annotations

from pathlib import Path

import dagster as dg

from graphids.config import PathContext
from graphids.orchestrate.analysis import (
    ANALYSIS_MANIFEST_NAME,
    build_analysis_spec,
    output_status,
    supports_analysis,
)
from graphids.orchestrate.planning import StageConfig


def _paths_from_context(
    cfg: StageConfig, *, lake_root: str, user: str, dataset: str, seed: int,
) -> PathContext:
    return PathContext(
        lake_root=lake_root, user=user, dataset=dataset,
        model_type=cfg.model_type, scale=cfg.scale, stage=cfg.stage,
        identity=cfg.identity, kd_tag=cfg.kd_tag, seed=seed,
    )


def _check_result(
    *, context, cfg: StageConfig, lake_root: str, user: str,
) -> dg.AssetCheckResult:
    """Verify checkpoint + complete marker, and analysis outputs if supported."""
    dataset = context.partition_key.keys_by_dimension["dataset"]
    seed = int(context.partition_key.keys_by_dimension["seed"])
    paths = _paths_from_context(cfg, lake_root=lake_root, user=user, dataset=dataset, seed=seed)

    ckpt_ok = paths.ckpt_file.exists() and paths.complete_marker.exists()
    metadata: dict[str, dg.MetadataValue] = {
        "path": dg.MetadataValue.path(str(paths.ckpt_file)),
    }

    if supports_analysis(cfg.model_type):
        spec = build_analysis_spec(
            cfg=cfg, dataset=dataset, seed=seed, ckpt_path=str(paths.ckpt_file),
        )
        manifest_path = Path(spec.output_dir) / ANALYSIS_MANIFEST_NAME
        expected, existing = output_status(spec)
        analysis_ok = manifest_path.exists() and len(existing) == len(expected)
        metadata["manifest"] = dg.MetadataValue.path(str(manifest_path))
        metadata["expected"] = len(expected)
        metadata["existing"] = len(existing)
    else:
        analysis_ok = True  # no analysis expected

    return dg.AssetCheckResult(passed=ckpt_ok and analysis_ok, metadata=metadata)


def make_asset_checks(
    cfg_lookup: dict[str, StageConfig],
    partitions_def: dg.MultiPartitionsDefinition,
    lake_root: str,
    user: str,
) -> list[dg.AssetChecksDefinition]:
    """One combined check per training asset: checkpoint + analysis outputs."""
    checks = []
    for asset_name, cfg in cfg_lookup.items():

        def _make_check(name: str, c: StageConfig):
            @dg.asset_check(
                asset=dg.AssetKey(name),
                name=f"outputs_complete_{name}",
                blocking=True,
                description=f"Verify outputs for {name}",
                partitions_def=partitions_def,
            )
            def _check(context) -> dg.AssetCheckResult:
                return _check_result(
                    context=context, cfg=c, lake_root=lake_root, user=user,
                )

            return _check

        checks.append(_make_check(asset_name, cfg))
    return checks
