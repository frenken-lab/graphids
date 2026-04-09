"""Pipeline orchestration — planning, resolution, execution, and ops.

Module layout:
- planning/    : recipe expansion + StageConfig + enumerate_assets
- resolve.py   : ResolvedConfig
- monarch.py   : PipelineConfig, JobSpec, build_pipeline_stages, run_chain
- actors.py    : PipelineActor (Monarch)
- _setup.py    : ensure_spawn, touch_marker
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
