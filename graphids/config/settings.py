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

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class GraphIDSSettings(BaseSettings):
    """All ``GRAPHIDS_*`` env vars — typed, with defaults."""

    model_config = SettingsConfigDict(
        env_prefix="GRAPHIDS_",
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",  # .env is multi-consumer (shell scripts read vars not declared here)
        frozen=True,
    )

    # --- Paths ---
    # Required. No default — a relative fallback ("experimentruns") used to
    # silently create a stub mlflow.db / state tree under CWD when
    # GRAPHIDS_LAKE_ROOT was unexported. Auto-loaded from the project's .env
    # so login-node invocations don't need `set -a; source ./.env`.
    #
    # ``lake_root`` is the SHARED data root: mlflow.db (cross-user metadata),
    # cache/, mlartifacts/, slurm_logs/. ``run_root`` is the PER-USER root
    # where this user's run_dirs / checkpoints / traces / predictions land.
    # On OSC the convention is ``run_root = lake_root / dev / $USER``, but
    # we keep them as independent env vars so each can move independently
    # without conflating shared and per-user state — that conflation is
    # what produced the 2026-04-24 lake_root drift between Python and
    # the jsonnet preset defaults.
    lake_root: str
    run_root: str
    scratch: Path | None = None

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
    cluster: str = ""

    # --- HuggingFace export ---
    # Target dataset repo for `python -m graphids push-hf`. Override via
    # GRAPHIDS_HF_REPO_ID for forks / lab-org migrations without touching code.
    hf_repo_id: str = "buckeyeguy/graphids-kd-gat"

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
