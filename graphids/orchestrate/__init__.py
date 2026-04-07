"""Pipeline orchestration — shared planning, resolution, and ops.

Module layout:
- planning/  : recipe expansion + StageConfig + enumerate_assets
- resolve/   : ConfigResolver + cross-field validation
- contracts/ : TrainingSpec + TLA dict construction
- ops/       : shared ops (from-spec, finalize, catalog, status)
- analysis.py: shared analysis runner (Monarch + dagster)
- dagster/   : dagster-specific component, assets, checks, resources, definitions
"""
