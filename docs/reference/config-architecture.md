# Config Architecture

> Status: **current**

GraphIDS now launches experiments from typed YAML files. The old
`graphids/plan` row renderer and `graphids/orchestrate.py` dispatcher are
retired.

## 1. User Surface

Experiment configs live under `configs/experiments/`.

```bash
gx exp config configs/experiments/temporal_event_classifier_smoke.yml
gx exp launch configs/experiments/temporal_event_classifier_smoke.yml
gx exp submit configs/experiments/temporal_event_classifier_smoke.yml -C pitzer
gx exp status "$GRAPHIDS_LAKE_ROOT/runs/set_01/temporal_event_classifier_smoke/temporal_event_classifier_smoke"
```

The YAML contains:

- `experiment_name`, `dataset`, `seed`, `plan_id`, `stage`
- top-level `representation_cfg`
- `resources` for Ray/SLURM execution
- `config`, whose shape depends on `stage`

Supported stages are currently `fit` and `test`.

## 2. Typed Boundary

`graphids.exp.config.ExperimentConfig.from_yaml(path)` loads YAML and validates
it with Pydantic.

Important classes:

- `ExperimentConfig`: top-level YAML contract.
- `RunConfig`: fully resolved single launch.
- `FitRunPayload`: `data`, `model`, `loss_fn`, `trainer`, `callbacks`.
- `ResourceConfig`: cluster/resource metadata.
- `OutputConfig`: run directory, journal, artifact, and checkpoint paths.

For `fit` and `test`, the config layer checks that
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

Ray launch instantiation accepts either:

- dictionaries with `type`, resolved through `graphids.primitives`, or
- dictionaries with `class_path` and optional `init_args`, resolved by
  importlib.

`graphids.exp.ray_backend` owns that resolution as part of launching. This lets
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

`gx exp submit <yaml> -C pitzer` uses `graphids.exp.slurm`:

1. Validate the YAML by building a `RunConfig`.
2. Render an sbatch script under `{slurm_log_dir}/scripts/`.
3. Submit that script with `sbatch`.

The sbatch body starts a Ray head and workers inside the allocation, then runs
the Ray driver:

```bash
cd /users/PAS2022/rf15/graphids
source scripts/slurm/_preamble.sh
python -m graphids exp launch /abs/path/to/config.yml --address "${RAY_ADDRESS}"
source scripts/slurm/_epilog.sh
```

`--dry-run` prints the script without submitting. The submit command has
resource overrides for cluster, partition, walltime, gres, and node count.

## 6. Current Temporal Path

The current cache/training line uses:

```yaml
representation_cfg:
  kind: temporal
config:
  data:
    type: temporal_dm
    source:
      type: can_bus
      dataset: set_01
      seed: 42
      representation_cfg:
        kind: temporal
    batch_size: 512
    val_warmup_events: 64
    test_warmup_events: 64
  model:
    type: temporal_event_classifier
    scale: small
```

The preprocessing layer builds temporal event tables and packs them as PyG
`TemporalData`. The primary batch unit is an event count, not a graph node or
edge budget. Validation and test splits may include warmup events, which are
fed through the model but excluded from metrics through `is_scored`.

Available temporal model types are `temporal_event_classifier`,
`temporal_rnn_classifier`, `temporal_gat`, and `temporal_vgae`.

Training configs select a dataset and temporal representation; the data source
maps that to the versioned cache path under `$GRAPHIDS_LAKE_ROOT/cache`.

## 7. Adding A Run

1. Create `configs/experiments/<name>.yml`.
2. Validate with `gx exp config <file>`.
3. For training, set `stage: fit`, callbacks, and trainer options.
4. For evaluation, set `stage: test` and provide `ckpt_path`.
5. Submit with `gx exp submit <file> -C pitzer`.
6. After completion, run `gx exp ingest <run_dir>` or `gx exp ingest-root <root>`.

Add tests when the YAML exercises new launch behavior, primitive fields, or
representation semantics.
