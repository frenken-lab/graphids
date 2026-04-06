"""Dagster-based pipeline orchestrator.

Module layout:
- dagster/   : Dagster-facing component, assets, checks, resources
- planning/  : pure planning + recipe expansion + StageConfig
- resolve/   : ConfigResolver + cross-field validation
- ops/       : CLI entry points (from-spec, pipeline-status, catalog)

Entry points:
  dg list defs                              — list all assets
  dg launch --assets autoencoder_*          — materialize assets
"""
