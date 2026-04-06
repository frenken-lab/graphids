"""Dagster-based pipeline orchestrator.

Assets represent trained model checkpoints. SlurmTrainingComponent reads
the pipeline topology + a Jsonnet recipe, generates tagged assets with
IOManager checkpoint handoff and SLURM submission via SlurmTrainingResource.

Entry points:
  dg list defs                              — list all assets
  dg launch --assets autoencoder_*          — materialize assets
"""
