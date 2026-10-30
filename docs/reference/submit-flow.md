# Submit Flow

> Status: **current**

The live submit path is experiment-YAML based.

## Ray Launch

```bash
gx exp launch configs/experiments/graph_mamba_events_source.yml
```

Flow:

1. Expand any relative base YAML and validate `ExperimentConfig`.
2. Build a `RunConfig`.
3. Call `graphids.exp.run.launch_run`.
4. Ray Train schedules the worker.
5. The worker saves configuration and records startup, executes `fit` or `test`,
   then publishes final metrics or failure in one manifest.

## SLURM Submit

```bash
gx exp submit configs/experiments/graph_mamba_events_source.yml -C pitzer
```

Flow:

1. Expand any relative base YAML and validate `ExperimentConfig`.
2. Build a `RunConfig` as a submit-time validation check.
3. Render a Ray allocation sbatch script with resource directives from `resources`.
4. Write the script to `{slurm_log_dir}/scripts/ray-{experiment_name}.sbatch`
   and an expanded config snapshot named by experiment and content hash.
5. Submit it with `sbatch`.

The compute node then runs:

```bash
python -m graphids exp launch /abs/path/to/expanded-snapshot.yml --address "${RAY_ADDRESS}"
```

## Dry Run

```bash
gx exp submit configs/experiments/graph_mamba_events_source.yml -C pitzer --dry-run
```

This prints the sbatch script without writing config snapshots, scripts, or
submitting jobs. The printed snapshot path is created by actual submission.

## Files Of Interest

- `graphids/cli/exp.py`
- `graphids/exp/config.py`
- `graphids/exp/run.py`
- `graphids/exp/slurm.py`
- `scripts/slurm/_preamble.sh`
- `scripts/slurm/_epilog.sh`

The old `gx run`, `gx exec --row`, `gx submit --row`, and
`gx plans submit` path is historical and should not be used for new runs.

Submission supports one node/device. `--source-root` selects a frozen checkout,
and `--dependency` supplies a Slurm dependency expression. Python callers use
`build_experiment_job`, `write_job` and `submit_script`; recipes do not modify
rendered shell text.
