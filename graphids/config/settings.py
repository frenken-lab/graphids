"""Centralized environment settings via pydantic-settings.

Single source of truth for all ``GRAPHIDS_*`` environment variables.
Typed, validated, read once at first access via ``get_settings()``.
"""

from __future__ import annotations

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
    budget_safety_margin: float = 0.85
    budget_fallback_bpn: int = 32768
    # Fraction of baseline scalable VRAM that may be lost before the drift
    # callback warns. 0.20 avoids allocator-fragmentation false positives;
    # genuine leaks / co-resident processes cross it quickly.
    vram_drift_threshold: float = 0.20

    # --- SLURM ---
    slurm_account: str = "PAS1266"
    slurm_log_dir: str = ""
    cluster: str = ""

    @model_validator(mode="after")
    def _derive_slurm_log_dir(self) -> GraphIDSSettings:
        if not self.slurm_log_dir:
            object.__setattr__(self, "slurm_log_dir", f"{self.lake_root}/slurm")
        return self


@lru_cache(maxsize=1)
def get_settings() -> GraphIDSSettings:
    """Lazy singleton — reads environment on first call, caches thereafter."""
    return GraphIDSSettings()


def _reset_settings_cache() -> None:
    """Clear cached settings (test helper)."""
    get_settings.cache_clear()


class LakeWriteError(PermissionError):
    pass


def require_lake_write() -> None:
    if not get_settings().lake_write:
        raise LakeWriteError(
            "Lake write blocked: set GRAPHIDS_LAKE_WRITE=1 "
            "(SLURM jobs get this from .env via _preamble.sh)"
        )
