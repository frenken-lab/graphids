"""Canonical path scheme for ablation run_dirs and shared upstream artifacts.

Single source of truth used by both Python (``slurm/dag.py``) and jsonnet
presets (via ``native_callbacks`` registered in
:func:`graphids.config.jsonnet.render`). Keeps the layout consistent across
the two languages — historically ``configs/ablations/_paths.libsonnet``
and ``slurm/dag.py:_run_dir`` were two implementations of the same scheme,
which drifted.

Layout under ``RUN_ROOT`` (= per-user, e.g.
``/fs/ess/PAS1266/graphids/dev/$USER``):

    {RUN_ROOT}/{dataset}/ablations/{group}/{variant}/seed_{N}/
    {RUN_ROOT}/{dataset}/ablations/fusion_states/seed_{N}/

Returns plain strings so jsonnet ``std.native(...)`` calls deserialize
cleanly. Python callers wanting ``Path`` should wrap with ``Path(...)``.
"""

from __future__ import annotations

from graphids.config.settings import get_settings


def run_dir(dataset: str, group: str, variant: str, seed: int) -> str:
    """Per-(variant, seed) run_dir path."""
    return f"{get_settings().run_root}/{dataset}/ablations/{group}/{variant}/seed_{int(seed)}"


def best_ckpt(dataset: str, group: str, variant: str, seed: int) -> str:
    """Best-model checkpoint path for an ablation run.

    Equivalent to ``f"{run_dir(...)}/checkpoints/best_model.ckpt"`` —
    callers should prefer this over jsonnet string-concat so the suffix
    lives in one place.
    """
    return f"{run_dir(dataset, group, variant, seed)}/checkpoints/best_model.ckpt"


def vgae_ckpt(dataset: str, seed: int) -> str:
    """Best-model checkpoint for the unsupervised vgae upstream (shorthand)."""
    return best_ckpt(dataset, "unsupervised", "vgae", seed)


def states_dir(dataset: str, seed: int) -> str:
    """Fusion-states directory shared across the 4 fusion methods for a seed."""
    return f"{get_settings().run_root}/{dataset}/ablations/fusion_states/seed_{int(seed)}"
