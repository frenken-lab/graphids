"""Pipeline orchestration — planning, resolution, and ops.

Module layout:
- contracts.py : TrainingSpec + TLA dict construction
- planning/    : recipe expansion + StageConfig + enumerate_assets
- resolve.py   : ConfigResolver + cross-field validation
- ops/         : finalize, catalog, status (CLI entry points)
- analysis.py  : shared analysis runner (Monarch)
"""
