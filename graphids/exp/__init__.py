"""New experiment seam.

This package is the replacement boundary for the old row/orchestrate chassis.
It intentionally stays small:

- ``config``: typed experiment/run config objects
- ``journal``: manifest + event log utilities
- ``runtime``: Lightning execution with MLflow/journal tracking
"""

from graphids.exp.config import (
    ExperimentConfig,
    FitRunPayload,
    OutputConfig,
    ResourceConfig,
    RunConfig,
)
from graphids.exp.journal import (
    EventRecord,
    RunManifest,
    append_event,
    load_events,
    load_manifest,
    write_manifest,
)

__all__ = [
    "ExperimentConfig",
    "FitRunPayload",
    "OutputConfig",
    "ResourceConfig",
    "RunConfig",
    "EventRecord",
    "RunManifest",
    "append_event",
    "load_events",
    "load_manifest",
    "write_manifest",
]
