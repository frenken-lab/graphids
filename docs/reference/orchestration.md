# Orchestration

> Status: **historical archive**

The old `graphids/orchestrate.py` row dispatcher has been retired. The
live launch surface is Ray-backed `graphids.exp.ray_backend`, which works
with typed `RunConfig` objects instead of rendered rows.

This page stays around so the older docs have a home, but it should be
treated as reference history rather than the current architecture map.

## Current entrypoints

- `graphids.exp.ray_backend.launch_run(run)`
- `graphids.exp.config.ExperimentConfig`
- `graphids.exp.config.RunConfig`
