"""Pipeline orchestration: scheduler-agnostic job management.

Components:
- job.py: JobSpec, ResourceSpec (Pydantic v2 frozen models)
- planner.py: Domain-aware DAG builder (KD-GAT specific)
- store.py: SQLite state store (job/attempt/transition tables)
- executor.py: SLURM/Flux/DryRun backends
- driver.py: Submit-and-poll loop + fire-and-forget mode
- ray_pipeline.py: Ray-based local/interactive orchestration
"""
