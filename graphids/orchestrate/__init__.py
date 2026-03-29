"""Dagster-based pipeline orchestrator.

SlurmTrainingComponent reads pipeline.yaml topology and ablation.yaml recipe,
generates one dagster asset per unique (stage, identity_hash) pair.
Assets submit LightningCLI stages to SLURM via sbatch.

Entry points:
  dg list defs                              — list all assets
  dg launch --assets autoencoder_*          — materialize assets
  python -m graphids.orchestrate validate   — validate config chains
  python -m graphids.orchestrate smoke      — smoke test on gpudebug
"""
