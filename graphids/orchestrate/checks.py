"""Dagster asset check factory helpers for orchestrated assets.

Two checks per training asset:
- checkpoint_complete — blocking, gates downstream assets
- analysis_complete  — non-blocking, informational only
"""

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


def _partition_keys(context) -> tuple[str, int]:
    """Extract (dataset, seed) from a multi-partition context."""
    dataset = context.partition_key.keys_by_dimension["dataset"]
    seed = int(context.partition_key.keys_by_dimension["seed"])
    return dataset, seed


def make_asset_checks(
    cfg_lookup: dict[str, StageConfig],
    partitions_def: dg.MultiPartitionsDefinition,
    lake_root: str,
    user: str,
) -> list[dg.AssetChecksDefinition]:
    """Two checks per training asset: checkpoint (blocking) + analysis (non-blocking)."""
    checks: list[dg.AssetChecksDefinition] = []
    for asset_name, cfg in cfg_lookup.items():

        # --- Checkpoint check: blocking ---
        def _make_ckpt_check(name: str, c: StageConfig):
            @dg.asset_check(
                asset=dg.AssetKey(name),
                name=f"checkpoint_complete_{name}",
                blocking=True,
                description=f"Verify checkpoint + complete marker for {name}",
                partitions_def=partitions_def,
            )
            def _check(context) -> dg.AssetCheckResult:
                dataset, seed = _partition_keys(context)
                paths = _paths_from_context(
                    c, lake_root=lake_root, user=user, dataset=dataset, seed=seed,
                )
                ckpt_ok = paths.ckpt_file.exists() and paths.complete_marker.exists()
                return dg.AssetCheckResult(
                    passed=ckpt_ok,
                    metadata={
                        "ckpt_path": dg.MetadataValue.path(str(paths.ckpt_file)),
                        "complete_marker": dg.MetadataValue.bool(
                            paths.complete_marker.exists()
                        ),
                    },
                )

            return _check

        checks.append(_make_ckpt_check(asset_name, cfg))

        # --- Analysis check: non-blocking, only for supported model types ---
        if supports_analysis(cfg.model_type):

            def _make_analysis_check(name: str, c: StageConfig):
                @dg.asset_check(
                    asset=dg.AssetKey(name),
                    name=f"analysis_complete_{name}",
                    blocking=False,
                    description=f"Verify analysis outputs for {name}",
                    partitions_def=partitions_def,
                )
                def _check(context) -> dg.AssetCheckResult:
                    dataset, seed = _partition_keys(context)
                    paths = _paths_from_context(
                        c, lake_root=lake_root, user=user, dataset=dataset, seed=seed,
                    )
                    spec = build_analysis_spec(
                        cfg=c, dataset=dataset, seed=seed,
                        ckpt_path=str(paths.ckpt_file),
                    )
                    manifest_path = Path(spec.output_dir) / ANALYSIS_MANIFEST_NAME
                    expected, existing = output_status(spec)
                    analysis_ok = manifest_path.exists() and len(existing) == len(expected)
                    return dg.AssetCheckResult(
                        passed=analysis_ok,
                        metadata={
                            "manifest": dg.MetadataValue.path(str(manifest_path)),
                            "expected": len(expected),
                            "existing": len(existing),
                        },
                    )

                return _check

            checks.append(_make_analysis_check(asset_name, cfg))

    return checks
