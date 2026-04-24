"""Centralized environment settings via pydantic-settings.

Single source of truth for all ``GRAPHIDS_*`` environment variables.
Typed, validated, read once at first access via ``get_settings()``.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class GraphIDSSettings(BaseSettings):
    """All ``GRAPHIDS_*`` env vars — typed, with defaults."""

    model_config = SettingsConfigDict(env_prefix="GRAPHIDS_", frozen=True)

    # --- Paths ---
    lake_root: str = "experimentruns"
    scratch: Path | None = None
    data_root: Path | None = None

    # --- Feature flags ---
    lake_write: bool = False
    dry_run: bool = False

    # --- Budget tuning ---
    # The two-point probe isolates the batch-scaling slope (bpn_node) and
    # leaves fixed overhead in an implicit intercept — resident / optimizer
    # state / cuDNN workspaces are already accounted for. That lets us push
    # the pack ceiling close to full VRAM without the old 0.85 cushion that
    # was covering up measurement bias in the single-probe estimator.
    budget_safety_margin: float = 0.95
    budget_fallback_bpn: int = 32768
    # Fraction of baseline scalable VRAM that may be lost before the drift
    # callback warns. 0.20 avoids allocator-fragmentation false positives;
    # genuine leaks / co-resident processes cross it quickly.
    vram_drift_threshold: float = 0.20

    # --- Budget fallback constants ---
    # Used by _gps_budget and node_budget when the probe can't run or degenerates.
    # Single source so callers don't inline magic numbers.
    gps_fallback_attention_divisor: int = 6  # budget = sqrt(free / (heads * divisor))
    fallback_edge_node_ratio: float = (
        10.0  # edge_budget = budget * this when no per-batch epn measured
    )
    empirical_epn_headroom: float = 1.1  # multiplier on measured edges-per-node when probe succeeds

    # --- SLURM ---
    slurm_account: str = "PAS1266"
    slurm_log_dir: str = ""
    cluster: str = ""

    @model_validator(mode="after")
    def _derive_slurm_log_dir(self) -> GraphIDSSettings:
        if not self.slurm_log_dir:
            object.__setattr__(self, "slurm_log_dir", f"{self.lake_root}/slurm")
        return self

    @model_validator(mode="after")
    def _derive_cluster(self) -> GraphIDSSettings:
        # Fall back to SLURM_CLUSTER_NAME when GRAPHIDS_CLUSTER isn't
        # exported by the submitter shell. SLURM sets SLURM_CLUSTER_NAME in
        # every job environment, so this works from any submission path
        # without per-script coordination (see gh#40).
        if not self.cluster:
            object.__setattr__(self, "cluster", os.environ.get("SLURM_CLUSTER_NAME", ""))
        return self


@lru_cache(maxsize=1)
def get_settings() -> GraphIDSSettings:
    """Lazy singleton — reads environment on first call, caches thereafter."""
    return GraphIDSSettings()


def _reset_settings_cache() -> None:
    """Clear cached settings (test helper)."""
    get_settings.cache_clear()


class LakeWriteError(PermissionError):
    """Raised when a lake-writing operation runs without
    ``GRAPHIDS_LAKE_WRITE=1``. Gate against accidental writes from
    login-node smoke runs or from interactive sessions.
    """


def require_lake_write() -> None:
    """Guard called by lake-writing code paths. Raises
    :class:`LakeWriteError` unless ``GRAPHIDS_LAKE_WRITE=1`` is set —
    SLURM jobs get this from ``.env`` via ``scripts/slurm/_preamble.sh``.
    """
    if not get_settings().lake_write:
        raise LakeWriteError(
            "Lake write blocked: set GRAPHIDS_LAKE_WRITE=1 "
            "(SLURM jobs get this from .env via _preamble.sh)"
        )
