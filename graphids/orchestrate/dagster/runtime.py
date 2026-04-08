"""Runtime helpers for Dagster asset execution and checks."""

from __future__ import annotations

import os
from pathlib import Path

import dagster as dg

from graphids.config.constants import COMPLETE_MARKER
from graphids.config.topology import PathContext
from graphids.config.settings import get_settings
from graphids.orchestrate.planning import StageConfig


def _runtime_lake_root() -> str:
    return get_settings().lake_root


def _runtime_user() -> str:
    return os.environ.get("USER", "unknown")


def _touch_complete(run_dir: Path) -> None:
    """Write the ``.complete`` marker after a successful run.

    Uses ``fsync`` on both the file and its parent directory so the marker
    is durable on NFS before this function returns — otherwise dagster and
    downstream resume-from-checkpoint checks can race the kernel cache.
    """
    marker = run_dir / COMPLETE_MARKER
    marker.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(marker), os.O_CREAT | os.O_WRONLY, 0o664)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    dir_fd = os.open(str(marker.parent), os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def partition_keys(
    context: dg.AssetExecutionContext | dg.AssetCheckExecutionContext,
) -> tuple[str, int]:
    """Extract (dataset, seed) from a dagster multi-partition context."""
    return (
        context.partition_key.keys_by_dimension["dataset"],
        int(context.partition_key.keys_by_dimension["seed"]),
    )


def paths_for_context(
    context: dg.AssetExecutionContext | dg.AssetCheckExecutionContext, cfg: StageConfig
) -> PathContext:
    """Build a PathContext for one materialization from dagster context + StageConfig.

    Reads lake_root/user from env at call time so the same helper serves both
    asset materialization and asset-check bodies.
    """
    dataset, seed = partition_keys(context)
    return PathContext(
        lake_root=_runtime_lake_root(),
        user=_runtime_user(),
        dataset=dataset,
        seed=seed,
        model_type=cfg.model_type,
        scale=cfg.scale,
        stage=cfg.stage,
        identity=cfg.identity,
        kd_tag=cfg.kd_tag,
    )
