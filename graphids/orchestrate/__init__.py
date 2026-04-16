"""Orchestration — render/validate a rendered config, instantiate, train.

- ``config.py``        — ResolvedConfig, InstantiatedRun.
- ``instantiate.py``   — build_run (class_path resolver).
- ``stage.py``         — build, train, evaluate primitives.

Import from the submodules directly. No re-export shim — call sites use
``from graphids.orchestrate.stage import build`` etc.

No planner, no cross-stage driver: multi-stage runs are a bash loop over
``scripts/run <preset.jsonnet>`` with ``SBATCH_DEP=afterok:<jid>`` deps.
"""
