"""Pipeline orchestration — planning, resolution, execution, and ops.

Module layout:
- planning/    : recipe expansion + StageConfig + enumerate_assets
- resolve.py   : ResolvedConfig + cross-field validation
- schemas.py   : PipelineConfig, SweepConfig (CLI input models)
- actors.py    : PipelineActor (Monarch)
- pipeline.py  : run_chain, run_sweep, build_pipeline_stages
- sweep.py     : ChainSpec, plan_chains, decompose_dag
- job.py       : JobSpec
- _setup.py    : ensure_spawn, touch_marker, bootstrap_staging
- analysis.py  : shared analysis runner
- ops/         : finalize, catalog, status (CLI entry points)
"""

from __future__ import annotations


def available() -> bool:
    """Return True if the monarch package is importable."""
    try:
        import monarch  # noqa: F401

        return True
    except ImportError:
        return False
