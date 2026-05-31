# GraphIDS: CAN Bus Intrusion Detection

GraphIDS trains and evaluates CAN bus intrusion-detection models over
materialized graph caches. The current launch surface is typed experiment YAML,
not the retired `graphids/plan` row chassis.

## Key Commands

The CLI is `graphids` or its short alias `gx`.

```bash
# Validate one experiment YAML.
gx exp config configs/experiments/gat_snapshot_sequence_real.yml

# Run locally/in-process.
gx exp launch configs/experiments/gat_snapshot_sequence_real.yml

# Submit to SLURM.
gx exp submit configs/experiments/gat_snapshot_sequence_real.yml -C pitzer
gx exp submit configs/experiments/gat_snapshot_sequence_real.yml -C pitzer --dry-run

# Inspect run state.
gx exp status ~/ray_results/set_01/gat_snapshot_sequence_real/gat_snapshot_sequence_real
gx exp manifest ~/ray_results/set_01/gat_snapshot_sequence_real/gat_snapshot_sequence_real

# Query configured result views.
gx exp results --view fusion

# Cluster helpers.
gx q
gx qpend
gx qhist
gx nodes
gx disk
```

## Current Architecture

```text
configs/experiments/<run>.yml
    -> graphids.exp.config.ExperimentConfig
    -> graphids.exp.config.RunConfig
    -> graphids.exp.runtime.launch_run
    -> run_stage: cache | fit | test | extract | analyze
    -> MLflow + .graphids/manifest.json + .graphids/events.jsonl
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
| Runtime dispatch | `graphids/exp/runtime.py` |
| SLURM submit | `graphids/exp/slurm.py`, `graphids/cli/exp.py` |
| Primitives | `graphids/primitives.py`, `graphids/primitives_data.py`, `graphids/primitives_models.py`, `graphids/primitives_losses.py` |
| Data/cache | `graphids/core/data/`, `graphids/core/data/preprocessing/` |
| Models | `graphids/core/models/` |
| Callbacks | `graphids/core/callbacks.py` |
| MLflow | `graphids/_mlflow.py` |

## Config Contract

Experiment YAML contains:

- `experiment_name`, `dataset`, `seed`, `plan_id`, `stage`
- top-level `representation_cfg`
- `resources`
- stage payload under `config`

For `fit`, `test`, and `cache`, the config layer verifies that
`config.data.source.representation_cfg` matches the top-level
`representation_cfg`.

Supported runtime stages:

- `cache`: build graph cache for the configured data source.
- `fit`: run Lightning training.
- `test`: run Lightning test.
- `extract`: precompute fusion states.
- `analyze`: generate checkpoint artifacts.

`hf_push` exists in the schema but is not wired in runtime.

## Data Layout

`$GRAPHIDS_LAKE_ROOT` is the shared persistent lake, normally
`/fs/ess/PAS1266/graphids`.

It holds:

- `raw/{dataset}/` source CSVs
- `cache/v{PREPROCESSING_VERSION}/{dataset}/...` materialized graph caches
- `mlflow.db` MLflow backend
- `slurm_logs/` sbatch scripts and stdout/stderr

Run directories default to:

```text
~/ray_results/{dataset}/{experiment_name}/{run_name}/
```

Each run directory contains `.graphids/manifest.json`,
`.graphids/events.jsonl`, and, for training runs, `checkpoints/` when
checkpointing is enabled.

## Snapshot-Sequence Path

Current sequence experiments use:

```yaml
representation_cfg:
  kind: snapshot_sequence
  window_size: 100
  stride: 100
  sequence_length: 3
  sequence_stride: 1
```

Cache jobs materialize ordered sequences of snapshot graphs. GAT training can
consume that metadata with `sequence_pool: gru`, `mean`, `attention`, `flat`,
or `auto`.

Training configs that rely on prebuilt caches should set:

```yaml
require_cache: true
```

The datamodule then fails fast if the cache is missing or incomplete.
