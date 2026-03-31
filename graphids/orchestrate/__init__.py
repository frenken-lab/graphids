"""Dagster-based pipeline orchestrator.

Assets represent trained model checkpoints. SlurmTrainingComponent reads
pipeline.yaml + ablation.yaml, generates tagged assets with IOManager
checkpoint handoff and SLURM submission via SlurmTrainingResource.

Entry points:
  dg list defs                              — list all assets
  dg launch --assets autoencoder_*          — materialize assets
  python -m graphids.orchestrate validate   — validate config chains
"""
