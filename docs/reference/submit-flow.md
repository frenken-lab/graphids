# Submit Flow

> Status: **current**

The live submit path is experiment-YAML based.

## Ray Launch

```bash
gx exp launch configs/experiments/gat_snapshot_sequence_real.yml
```

Flow:

1. Load and validate `ExperimentConfig`.
2. Build a `RunConfig`.
3. Call `graphids.exp.ray_backend.launch_run`.
4. Ray Train schedules the worker.
5. The worker writes run journal and MLflow state, then dispatches by `stage`.

## SLURM Submit

```bash
gx exp submit configs/experiments/gat_snapshot_sequence_real.yml -C pitzer
```

Flow:

1. Load and validate `ExperimentConfig`.
2. Build a `RunConfig` as a submit-time validation check.
3. Render a Ray allocation sbatch script with resource directives from `resources`.
4. Write the script to `{slurm_log_dir}/scripts/ray-{experiment_name}.sbatch`.
5. Submit it with `sbatch`.

The compute node then runs:

```bash
python -m graphids exp launch /abs/path/to/experiment.yml --address "${RAY_ADDRESS}"
```

## Dry Run

```bash
gx exp submit configs/experiments/gat_snapshot_sequence_real.yml -C pitzer --dry-run
```

This prints the sbatch script and does not submit.

## Files Of Interest

- `graphids/cli/exp.py`
- `graphids/exp/config.py`
- `graphids/exp/ray_backend.py`
- `graphids/exp/slurm.py`
- `scripts/slurm/_preamble.sh`
- `scripts/slurm/_epilog.sh`

The old `gx run`, `gx exec --row`, `gx submit --row`, and
`gx plans submit` path is historical and should not be used for new runs.
