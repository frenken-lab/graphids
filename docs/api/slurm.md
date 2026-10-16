# SLURM

The live SLURM surface is `gx exp submit <experiment.yml>`.

Implementation lives in `graphids.exp.slurm` and is exposed through
`graphids.cli.exp.submit`. It validates an `ExperimentConfig`, renders a Ray
allocation sbatch script under the configured SLURM log directory, and submits
it with `sbatch`.

Useful commands:

```bash
gx exp submit configs/experiments/graph_mamba_events_source.yml -C ascend
gx exp submit configs/experiments/graph_mamba_events_source.yml -C ascend --dry-run
```

## `graphids.exp.slurm`

::: graphids.exp.slurm
