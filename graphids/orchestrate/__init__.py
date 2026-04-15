"""Orchestration — render/validate a rendered config, instantiate, train.

- ``config.py``        — ResolvedConfig, InstantiatedRun.
- ``instantiate.py``   — build_run (+ build_model / datamodule / trainer / ...).
- ``stage.py``         — build, train, evaluate primitives.

No planner, no cross-stage driver: multi-stage runs are a bash loop over
``scripts/run <preset.jsonnet>`` with ``SBATCH_DEP=afterok:<jid>`` deps.
"""

from __future__ import annotations

from graphids.orchestrate.config import InstantiatedRun, ResolvedConfig
from graphids.orchestrate.instantiate import build_run
from graphids.orchestrate.stage import build, evaluate, train

__all__ = [
    "InstantiatedRun",
    "ResolvedConfig",
    "build",
    "build_run",
    "evaluate",
    "train",
]
