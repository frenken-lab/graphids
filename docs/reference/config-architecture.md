# Config Architecture

> Status: **current**

GraphIDS now launches experiments from typed YAML files. The old
`graphids/plan` row renderer and `graphids/orchestrate.py` dispatcher are
retired.

## 1. User Surface

Experiment configs live under `configs/experiments/`.

```bash
gx exp config configs/experiments/graph_mamba_events_source.yml
gx exp launch configs/experiments/graph_mamba_events_source.yml
gx exp submit configs/experiments/graph_mamba_events_source.yml -C ascend
gx exp status "$GRAPHIDS_RUN_ROOT/pooled_can/graph_mamba_events_source_seed42/graph_mamba_events_source_seed42"
```

The YAML contains:

- `experiment_name`, `dataset`, `seed`, `plan_id`, `stage`
- top-level `representation_cfg`
- `resources` for Ray/SLURM execution
- `config`, whose shape depends on `stage`

Supported stages are currently `fit` and `test`.

## 2. Typed Boundary

`graphids.exp.config.ExperimentConfig.from_yaml(path)` loads YAML and validates
it with Pydantic. Optional `extends` selects one base YAML relative to the
experiment file. Shared partial configs live in `configs/experiments/_base/`;
launch a leaf experiment rather than a base.

```yaml
experiment_name: graph_mamba_events_source_seed42
dataset: pooled_can
seed: 42
stage: fit
representation_cfg:
  kind: can_event
  schema_version: 1
config:
  model:
    type: event_graph_mamba
  data:
    type: can_event_dm
    manifest_path: /path/to/reviewed-manifest.json
    datasets: [set_01, set_03]
    cache_root: /path/to/event-cache
    chunk_events: 2048
    stream_batch_size: 4
    effective_batch_events: 8192
  trainer:
    devices: 1
    precision: 32-true
```

Mappings merge recursively. Child scalars, lists, and explicit nulls replace
inherited values. Changing a component's `type` or `class_path` replaces that
whole component so obsolete arguments cannot leak into it. Missing bases,
cycles, and non-mapping YAML fail with the source paths. Bases may themselves
extend a base; there are no templates, list merges, or deletion directives.
Existing standalone experiment YAML remains supported.

Use `load_experiment_yaml(path)` when a runner needs an expanded dictionary;
use `ExperimentConfig.from_yaml(path)` for validation. Benchmark variants are
written as expanded standalone YAML. Every executed run saves
`resolved-run.yml`, loadable as `RunConfig`, with concrete output paths.

Important classes:

- `ExperimentConfig`: top-level YAML contract.
- `RunConfig`: fully resolved single launch.
- `FitRunPayload`: `data`, `model`, `loss_fn`, `trainer`, `callbacks`.
- `ResourceConfig`: cluster/resource metadata.
- `OutputConfig`: run directory, journal, artifact, and checkpoint paths.

For `fit` and `test`, the config layer checks that
`config.data.representation_cfg` matches the top-level
`representation_cfg`. That prevents the run metadata and materialized cache
from silently drifting.

## 3. Primitives

`graphids.primitives` is the public primitive module. It re-exports data,
model and message-preserving event representation primitives.

Primitive config objects are Pydantic models with `extra="forbid"`:

- data: `graphids.primitives_data`
- models: `graphids.primitives_models`
- representations: `graphids.core.data.preprocessing.representations`

Ray launch instantiation accepts either:

- dictionaries with `type`, resolved through `graphids.primitives`, or
- dictionaries with `class_path` and optional `init_args`, resolved by
  importlib.

`graphids.exp.runtime` owns resolution, seeding, model preparation, callbacks,
and trainer kwargs for Ray launches and bounded event benchmarks.
Existing imports from `graphids.exp.ray_backend` remain available. This lets
YAML use compact primitive specs for common objects and explicit class paths for
callbacks without putting import logic in the YAML loader or CLI.

## 4. Ray Dispatch

`graphids.exp.ray_backend.launch_run(run)` owns driver-side execution and
uses Ray Train. Each Ray worker runs the backend worker loop:

1. Translate GraphIDS resources/output settings to Ray Train configs.
2. Start or connect to Ray.
3. Run a `TorchTrainer`.
4. In the worker, choose tracking mode from `GRAPHIDS_MLFLOW_MODE`.
5. Write `.graphids/manifest.json`.
6. Append `.graphids/events.jsonl`.
7. In offline mode, write `.graphids/mlflow_ingest.json`.
8. Build configured data/model/trainer objects and run `fit` or `test`.
9. Mark the manifest and ingest payload `finished` or `failed`.

Stage dispatch:

| Stage | Ray action |
|---|---|
| `fit` | Build datamodule/model/loss/callbacks, run `trainer.fit`. |
| `test` | Build datamodule/model/loss/callbacks, run `trainer.test`. |

After `fit` and `test`, the Ray backend captures final `trainer.callback_metrics` into
the returned run event payload and offline MLflow ingest payload. A separate
`gx exp ingest <run_dir>` or `gx exp ingest-root <root>` command serializes
completed runs into MLflow.

## 5. SLURM Submit

`gx exp submit <yaml> -C ascend` uses `graphids.exp.slurm`:

1. Validate the YAML by building a `RunConfig`.
2. Expand and save a standalone config under `{slurm_log_dir}/scripts/`, named
   by experiment and configuration content hash.
3. Render an sbatch script pointing to that snapshot and submit it with `sbatch`.

Later base or leaf edits do not change a queued job's configuration. Model
source still comes from the selected code checkout.

The sbatch body starts a Ray head and workers inside the allocation, then runs
the Ray driver:

```bash
cd /users/PAS2022/rf15/projects/lab/graphids
source scripts/slurm/_preamble.sh
python -m graphids exp launch /abs/path/to/config.yml --address "${RAY_ADDRESS}"
source scripts/slurm/_epilog.sh
```

`--dry-run` prints the script without submitting. The submit command has
resource overrides for cluster, partition, walltime, gres, and node count.

## 6. Continuous event training

The sole active contract is `can_event` / `can_event_dm` / `event_graph_mamba`.
Use one GPU and FP32. Training schedules independent chronological recording
streams, carries history, detaches learned state between chunks, and scores each
physical row once. `effective_batch_events` controls accumulation with separate
eligible-target denominators for classification and normal-only payload loss.
Use `accumulate_grad_batches=1`; configure checkpoint selection with
`val/classification_frame_ap`. See [the event contract](graph-mamba-events.md).

Only epoch-boundary resume is supported through `config.ckpt_path`. The model
rejects source/config mismatches and checkpoints without complete epoch identity.
Retired representation, model and loss factories point to the verified snapshot
archive. Historical datasets, results, clock repairs and reviewed manifests remain
in place.

## 7. Adding A Run

1. Create `configs/experiments/<name>.yml`.
2. Validate with `gx exp config <file>`.
3. For training, set `stage: fit`, callbacks, and trainer options.
4. For evaluation, set `stage: test` and provide `ckpt_path`.
5. Submit with `gx exp submit <file> -C ascend`.
6. After completion, run `gx exp ingest <run_dir>` or `gx exp ingest-root <root>`.

Add tests when the YAML exercises new launch behavior, primitive fields, or
representation semantics.
