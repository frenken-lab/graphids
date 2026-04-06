"""SLURM infrastructure: resource profiles, job submission, accounting."""

from graphids.slurm.core.accounting import job_accounting, parse_elapsed, sacct_by_user, sacct_query
from graphids.slurm.core.submit import submit
from graphids.slurm.pipeline import SlurmJobClient, SubprocessSlurmJobClient
from graphids.slurm.resources import (
    ResourceSpec,
    apply_resource_overrides,
    get_resources,
    scale_resources,
)

__all__ = [
    "ResourceSpec",
    "apply_resource_overrides",
    "get_resources",
    "scale_resources",
    "SlurmJobClient",
    "SubprocessSlurmJobClient",
    "job_accounting",
    "parse_elapsed",
    "sacct_by_user",
    "sacct_query",
    "submit",
]
