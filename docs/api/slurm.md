# SLURM

The live SLURM surface is `gx exp submit <experiment.yml>`.

Implementation lives in `graphids.exp.slurm` and is exposed through
`graphids.cli.exp.submit`. It validates an `ExperimentConfig`, renders an
sbatch script under the configured SLURM log directory, and submits it with
`sbatch`.

Useful commands:

```bash
gx exp submit configs/experiments/gat_snapshot_sequence_real.yml -C pitzer
gx exp submit configs/experiments/gat_snapshot_sequence_real.yml -C pitzer --dry-run
```

## `graphids.exp.slurm`

::: graphids.exp.slurm
