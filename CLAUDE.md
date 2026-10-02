# GraphIDS: CAN Bus Intrusion Detection

GraphIDS trains and evaluates CAN bus intrusion-detection models over
materialized temporal event caches. The current launch surface is typed experiment YAML,
not the retired `graphids/plan` row chassis.

## Key Commands

The CLI is `graphids` or its short alias `gx`.

```bash
# Validate one experiment YAML.
gx exp config configs/experiments/gat_temporal_smoke.yml

# Launch through Ray Train.
gx exp launch configs/experiments/gat_temporal_smoke.yml

# Submit to SLURM.
gx exp submit configs/experiments/gat_temporal_smoke.yml -C pitzer
gx exp submit configs/experiments/gat_temporal_smoke.yml -C pitzer --dry-run

# Inspect run state.
gx exp status "$GRAPHIDS_RUN_ROOT/set_01/gat_temporal_smoke/gat_temporal_smoke"
gx exp manifest "$GRAPHIDS_RUN_ROOT/set_01/gat_temporal_smoke/gat_temporal_smoke"

# Query configured result views.
gx exp results
```

## Current Architecture

```text
configs/experiments/<run>.yml
    -> graphids.exp.config.ExperimentConfig
    -> graphids.exp.config.RunConfig
    -> graphids.exp.ray_backend.launch_run
    -> run_stage: fit | test
    -> .graphids/manifest.json + .graphids/events.jsonl + .graphids/mlflow_ingest.json
    -> gx exp ingest <run_dir> writes MLflow serially after training
```

`gx exp submit` uses `graphids.exp.slurm` to render an sbatch script that
runs:

```bash
python -m graphids exp launch /abs/path/to/experiment.yml
```

No Parsl row submission is used in the live path.

## Important Files

| Area | Files |
|---|---|
| Experiment schema | `graphids/exp/config.py` |
| Shared experiment construction | `graphids/exp/runtime.py` |
| Shared benchmark trials | `graphids/exp/trials.py` |
| Ray launch | `graphids/exp/ray_backend.py` |
| SLURM submit | `graphids/exp/slurm.py`, `graphids/cli/exp.py` |
| Primitives | `graphids/primitives.py`, `graphids/primitives_data.py`, `graphids/primitives_models.py`, `graphids/primitives_losses.py` |
| Data/cache | `graphids/core/data/`, `graphids/core/data/preprocessing/` |
| Models | `graphids/core/models/` |
| Callbacks | `graphids/core/callbacks.py` |
| MLflow ingest | `graphids/exp/ingest.py`, `graphids/_mlflow.py` |

## Config Contract

Experiment YAML may `extends` one relative base file under
`configs/experiments/_base/`. Mappings merge, scalars/lists/null replace, and
changed component types replace the whole component. Load through
`ExperimentConfig.from_yaml`; benchmark variants and submitted snapshots are
expanded standalone YAML. Shared validation lives in `graphids/config_validation.py`.

Experiment YAML contains:

- `experiment_name`, `dataset`, `seed`, `plan_id`, `stage`
- top-level `representation_cfg`
- `resources`
- stage payload under `config`

For `fit` and `test`, the config layer verifies that
`config.data.source.representation_cfg` matches the top-level
`representation_cfg`.

Supported Ray stages:

- `fit`: run Lightning training.
- `test`: run Lightning test.

## Data Layout

`$GRAPHIDS_LAKE_ROOT` is the shared persistent lake, normally
`/fs/ess/PAS1266/graphids`.

It holds:

- `raw/{dataset}/` source CSVs
- `cache/v{PREPROCESSING_VERSION}/{dataset}/...` materialized temporal event caches
- `mlflow.db` MLflow backend populated by serialized `gx exp ingest`
- `slurm_logs/` sbatch scripts and stdout/stderr

Run directories default to:

```text
$GRAPHIDS_RUN_ROOT/{dataset}/{experiment_name}/{run_name}/
```

Run roots fall back to `{lake_root}/runs`, then `{repo}/runs`. Diagnostic
artifacts belong under `$GRAPHIDS_RUN_ROOT/experimentruns/`, outside the repo.

Each run directory contains `resolved-run.yml`, `.graphids/manifest.json`,
`.graphids/events.jsonl`, `.graphids/mlflow_ingest.json`, and, for training
runs, `checkpoints/` when checkpointing is enabled.

## Temporal Training

The live representation is `kind: temporal`. Models include the event
classifier, RNN, GAT, VGAE, and hybrid. Lane training uses continuous recording
state by default; segmented execution and CUDA graphs remain optional.
Preserve coverage, successor metadata, warmup masks, checkpoint keys, and
capture storage ownership when refactoring. Run pytest and GPU workloads in
SLURM allocations; static lint and config inspection are login-node safe.
