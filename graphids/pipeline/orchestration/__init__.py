"""Pipeline orchestration.

Components:
- job.py: JobSpec, ResourceSpec (Pydantic v2 frozen models)
- store.py: SQLite state store (job/attempt/transition tables)
- ray_pipeline.py: Ray-based local/interactive orchestration
- sweep_pipeline.py: HPO sweep DAG
- tune_config.py: Ray Tune search space + ASHA
"""
