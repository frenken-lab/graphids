"""Pipeline orchestration.

Public API:
    from graphids.pipeline.orchestration import fire_and_forget, build_dagster_assets
    from graphids.pipeline.orchestration import PipesSlurmClient, SlurmJobFailed
    from graphids.pipeline.orchestration import ResourceSpec
"""

from graphids.pipeline.orchestration.job import ResourceSpec
from graphids.pipeline.orchestration.slurm_primitives import SlurmJobFailed


# Lazy imports for Dagster (heavy dependency)
def __getattr__(name):
    if name in ("fire_and_forget", "build_dagster_assets"):
        from graphids.pipeline.orchestration import dagster_defs

        return getattr(dagster_defs, name)
    if name == "PipesSlurmClient":
        from graphids.pipeline.orchestration.pipes_slurm import PipesSlurmClient

        return PipesSlurmClient
    raise AttributeError(f"module 'graphids.pipeline.orchestration' has no attribute {name!r}")
