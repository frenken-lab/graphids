# Config Architecture

> Status: **current**

GraphIDS now launches experiments from typed YAML files. The old
`graphids/plan` row renderer and `graphids/orchestrate.py` dispatcher are
retired.

## 1. User Surface

Experiment configs live under `configs/experiments/`.

```bash
gx exp config configs/experiments/gat_snapshot_sequence_real.yml
gx exp launch configs/experiments/gat_snapshot_sequence_real.yml
gx exp submit configs/experiments/gat_snapshot_sequence_real.yml -C pitzer
gx exp status ~/ray_results/set_01/gat_snapshot_sequence_real/gat_snapshot_sequence_real
```

The YAML contains:

- `experiment_name`, `dataset`, `seed`, `plan_id`, `stage`
- top-level `representation_cfg`
- `resources` for local/SLURM execution
- `config`, whose shape depends on `stage`

Supported stages are `fit`, `test`, `cache`, `extract`, `analyze`, and
`hf_push` in the schema. Runtime support currently covers `fit`, `test`,
`cache`, `extract`, and `analyze`; `hf_push` is still a placeholder.

## 2. Typed Boundary

`graphids.exp.config.ExperimentConfig.from_yaml(path)` loads YAML through
OmegaConf and validates it with Pydantic.

Important classes:

- `ExperimentConfig`: top-level YAML contract.
- `RunConfig`: fully resolved single launch.
- `FitRunPayload`: `data`, `model`, `loss_fn`, `trainer`, `callbacks`.
- `CacheRunPayload`: `data` plus seed.
- `ExtractRunPayload`: checkpoint extraction options.
- `AnalyzeRunPayload`: per-checkpoint artifact options.
- `ResourceConfig`: cluster/resource metadata.
- `OutputConfig`: run directory, journal, artifact, and checkpoint paths.

For `fit`, `test`, and `cache`, the config layer checks that
`config.data.source.representation_cfg` matches the top-level
`representation_cfg`. That prevents the run metadata and materialized cache
from silently drifting.

## 3. Primitives

`graphids.primitives` is the public primitive module. It re-exports data,
model, loss, scaler, representation, ID-encoding, and discovery primitives.

Primitive config objects are Pydantic models with `extra="forbid"`:

- data: `graphids.primitives_data`
- models: `graphids.primitives_models`
- losses: `graphids.primitives_losses`
- representations: `graphids.core.data.preprocessing.representations`

Runtime instantiation accepts either:

- dictionaries with `type`, resolved through `graphids.primitives`, or
- dictionaries with `class_path` and optional `init_args`, resolved by
  importlib.

This lets YAML use compact primitive specs for common objects and explicit
class paths for callbacks.

## 4. Runtime Dispatch

`graphids.exp.runtime.launch_run(run)` owns the execution lifecycle:

1. Create the MLflow logger.
2. Log hyperparameters and tags from `RunConfig`.
3. Write `.graphids/manifest.json`.
4. Append `.graphids/events.jsonl`.
5. Dispatch to `run_stage(run)`.
6. Mark the MLflow run and manifest `finished` or `failed`.

Stage dispatch:

| Stage | Runtime action |
|---|---|
| `fit` | Build datamodule/model/loss/callbacks, run `trainer.fit`. |
| `test` | Build datamodule/model/loss/callbacks, run `trainer.test`. |
| `cache` | Build the datamodule/source and call setup/build to materialize cache. |
| `extract` | Run `graphids.core.data.extract.extract_states`. |
| `analyze` | Run `graphids.core.artifacts.analyzer.Analyzer`. |

After `fit` and `test`, runtime also copies final
`trainer.callback_metrics` into MLflow and into the returned run event
payload. This makes short runs and smoke runs leave explicit final metrics.

## 5. SLURM Submit

`gx exp submit <yaml> -C pitzer` uses `graphids.exp.slurm`:

1. Validate the YAML by building a `RunConfig`.
2. Render an sbatch script under `{slurm_log_dir}/scripts/`.
3. Submit that script with `sbatch`.

The sbatch body is intentionally simple:

```bash
cd /users/PAS2022/rf15/graphids
source scripts/slurm/_preamble.sh
python -m graphids exp launch /abs/path/to/config.yml
source scripts/slurm/_epilog.sh
```

`--dry-run` prints the script without submitting. The submit command has
resource overrides for cluster, partition, walltime, and gres.

## 6. Current Snapshot-Sequence Path

The current cache/training line uses:

```yaml
representation_cfg:
  kind: snapshot_sequence
  window_size: 100
  stride: 100
  sequence_length: 3
  sequence_stride: 1
```

The preprocessing layer builds ordered sequences of snapshot graphs and
attaches sequence metadata to graph, node, and edge tensors. The GAT can
consume this through `sequence_pool`, currently including `auto`, `flat`,
`mean`, `attention`, and `gru`.

Training configs should set `require_cache: true` when they depend on a
prebuilt cache. The datamodule then fails before building anything if the
cache is incomplete.

## 7. Adding A Run

1. Create `configs/experiments/<name>.yml`.
2. Validate with `gx exp config <file>`.
3. For cache builds, set `stage: cache`.
4. For training, set `stage: fit`, `require_cache: true`, callbacks, and
   trainer options.
5. Submit with `gx exp submit <file> -C pitzer`.

Add tests when the YAML exercises new runtime behavior, primitive fields, or
representation semantics.
