"""Dagster-based pipeline orchestrator.

Assets submit LightningCLI stages to SLURM. Dagster handles DAG ordering,
partitions, retry, and concurrency.

Entry point: dagster asset materialize -m graphids.orchestrate.dagster_defs
"""
