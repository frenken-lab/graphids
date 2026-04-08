"""Dagster asset check factory helpers for orchestrated assets.

Two checks per training asset, both emitted from a single ``@multi_asset_check``
op per asset (shared setup, can_subset=True):

- ``checkpoint_complete_*`` — blocking; gates downstream assets.
- ``analysis_complete_*``  — non-blocking; informational only (skipped for
  model families that don't produce analyzer artifacts).
"""

from __future__ import annotations

from pathlib import Path

import dagster as dg

from graphids.config.constants import PHASE_MARKERS
from graphids.config.topology import PathContext
from graphids.orchestrate.analysis import (
    ANALYSIS_MANIFEST_NAME,
    build_analysis_spec,
    output_status,
    supports_analysis,
)
from graphids.orchestrate.dagster.runtime import paths_for_context
from graphids.orchestrate.planning import StageConfig


def _ckpt_result(
    check_name: str, asset_key: dg.AssetKey, paths: PathContext
) -> dg.AssetCheckResult:
    """Evaluate the checkpoint-complete check for one materialization."""
    ckpt = paths.resolved_ckpt
    ckpt_ok = ckpt.exists() and paths.complete_marker.exists()
    phase_status = {
        phase: (paths.run_dir / marker).exists() for phase, marker in PHASE_MARKERS.items()
    }
    return dg.AssetCheckResult(
        check_name=check_name,
        asset_key=asset_key,
        passed=ckpt_ok,
        metadata={
            "ckpt_path": dg.MetadataValue.path(str(ckpt)),
            "complete_marker": dg.MetadataValue.bool(paths.complete_marker.exists()),
            **{f"phase_{phase}": dg.MetadataValue.bool(ok) for phase, ok in phase_status.items()},
        },
    )


def _analysis_result(
    check_name: str,
    asset_key: dg.AssetKey,
    paths: PathContext,
    cfg: StageConfig,
) -> dg.AssetCheckResult:
    """Evaluate the analysis-complete check for one materialization."""
    spec = build_analysis_spec(
        cfg=cfg,
        dataset=paths.dataset,
        seed=paths.seed,
        ckpt_path=str(paths.resolved_ckpt),
    )
    manifest_path = Path(spec.output_dir) / ANALYSIS_MANIFEST_NAME
    expected, existing = output_status(spec)
    return dg.AssetCheckResult(
        check_name=check_name,
        asset_key=asset_key,
        passed=manifest_path.exists() and len(existing) == len(expected),
        metadata={
            "manifest": dg.MetadataValue.path(str(manifest_path)),
            "expected": len(expected),
            "existing": len(existing),
        },
    )


def _make_asset_checks(asset_name: str, cfg: StageConfig) -> dg.AssetChecksDefinition:
    """Build one ``@multi_asset_check`` op emitting all checks for a single asset.

    Both checks share ``paths_for_context(context, cfg)``, so the op runs that
    once per materialization instead of once per check. ``can_subset=True``
    lets dagster target individual checks (e.g. re-run analysis without
    re-running the blocking checkpoint check).
    """
    asset_key = dg.AssetKey(asset_name)
    ckpt_name = f"checkpoint_complete_{asset_name}"
    analysis_name = f"analysis_complete_{asset_name}"

    specs = [
        dg.AssetCheckSpec(
            name=ckpt_name,
            asset=asset_key,
            blocking=True,
            description=f"Verify checkpoint + complete marker for {asset_name}",
        ),
    ]
    if supports_analysis(cfg.model_type):
        specs.append(
            dg.AssetCheckSpec(
                name=analysis_name,
                asset=asset_key,
                blocking=False,
                description=f"Verify analysis outputs for {asset_name}",
            )
        )

    @dg.multi_asset_check(
        name=f"checks_{asset_name}",
        specs=specs,
        can_subset=True,
    )
    def _checks(context: dg.AssetCheckExecutionContext):
        paths = paths_for_context(context, cfg)
        selected = {key.name for key in context.selected_asset_check_keys}

        if ckpt_name in selected:
            yield _ckpt_result(ckpt_name, asset_key, paths)
        if analysis_name in selected:
            yield _analysis_result(analysis_name, asset_key, paths, cfg)

    return _checks


def make_asset_checks(
    cfg_lookup: dict[str, StageConfig],
) -> list[dg.AssetChecksDefinition]:
    """One ``AssetChecksDefinition`` per training asset, each emitting 1-2 checks."""
    return [_make_asset_checks(name, cfg) for name, cfg in cfg_lookup.items()]
