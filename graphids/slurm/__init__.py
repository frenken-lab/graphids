"""SLURM infrastructure: resource profiles, job submission, accounting."""

from graphids.slurm.resources import (
    ResourceSpec,
    apply_resource_overrides,
    get_resources,
    scale_resources,
)
from graphids.slurm.slurm import (
    SlurmJobClient,
    SubprocessSlurmJobClient,
    job_accounting,
    sacct_by_user,
    sacct_query,
    submit,
)

__all__ = [
    "ResourceSpec",
    "apply_resource_overrides",
    "get_resources",
    "scale_resources",
    "SlurmJobClient",
    "SubprocessSlurmJobClient",
    "job_accounting",
    "sacct_by_user",
    "sacct_query",
    "submit",
]
