"""New experiment seam.

This package is the replacement boundary for the old row/orchestrate chassis.
It intentionally stays small:

- ``config``: typed experiment/run config objects
- ``journal``: manifest + event log utilities
- ``runtime``: execution helpers that can later switch to Ray/Hydra
"""

from graphids.exp.config import (
    AnalyzeRunPayload,
    ExperimentConfig,
    ExtractRunPayload,
    FitRunPayload,
    OutputConfig,
    ResourceConfig,
    RunConfig,
)
from graphids.exp.journal import EventRecord, RunManifest, append_event, load_events, load_manifest, write_manifest

__all__ = [
    "ExperimentConfig",
    "FitRunPayload",
    "ExtractRunPayload",
    "AnalyzeRunPayload",
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
