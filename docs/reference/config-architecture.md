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
experiment_name: graph_mamba_events_id_gap_seed42
dataset: pooled_can
seed: 42
stage: fit
representation_cfg:
  kind: can_event
  schema_version: 2
config:
  model:
    type: event_graph_mamba
    schema_version: 2
    use_id_gap: true
    max_graph_tokens: 8388608
  data:
    type: can_event_dm
    manifest_path: /path/to/reviewed-manifest.json
    datasets: [set_01, set_03]
    cache_root: /path/to/event-cache
    chunk_events: 16384
    stream_batch_size: 1
    effective_batch_events: 65536
    max_microbatch_events: 65536
    num_workers: 2
    prefetch_factor: 2
    persistent_workers: true
    cuda_prefetch: false
    pin_memory: true
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
- `RunPayload`: `data`, `model`, `trainer`, `callbacks`, optional `compile` and `ckpt_path`.
- `ResourceConfig`: cluster/resource metadata.
- `OutputConfig`: run directory, journal, artifact, and checkpoint paths.

For `fit` and `test`, the config layer checks that
`config.data.representation_cfg` matches the top-level
`representation_cfg`. That prevents the run metadata and materialized cache
from silently drifting.

## 3. Component configuration

Data/model config objects are Pydantic models with `extra="forbid"`:

- data: `graphids.data.config.CANEventDataConfig`
- models: `graphids.models.config.EventGraphMambaConfig`

`graphids.data.representation.CANEventRepresentationConfig` is the frozen
dataclass describing the message-preserving event representation.

Ray launch instantiation accepts either:

- dictionaries with `type: can_event_dm` or `type: event_graph_mamba`, resolved
  directly to the corresponding config class, or
- dictionaries with `class_path` and optional `init_args`, resolved by
  importlib.

`graphids.exp.run` owns component construction, seeding, model preparation,
callbacks, Lightning execution and Ray launch. It consumes a resolved `RunConfig`;
only the Ray process boundary serializes and revalidates that object. Bounded
event benchmarks reuse the component builders.
Direct data/model constructors reuse component validation, including checkpoint
reconstruction. Builders propagate errors without argument filtering or retries.
The event model owns its objectives: a non-null `loss_fn` setting is rejected
when loading an experiment or resolved payload. Old payloads with `loss_fn: null`
remain readable. A zero learning rate is supported for frozen-weight diagnostics.
Import construction helpers from `graphids.exp.run`. This lets YAML use
compact component specs for common objects and explicit class paths for
callbacks without putting import logic in the YAML loader or CLI.

## 4. Run execution and records

The experiment package has six implementation modules:

| Module | Responsibility |
| --- | --- |
| `config.py` | YAML composition, validation, resolved configuration |
| `run.py` | Component construction, Lightning execution, Ray launch |
| `records.py` | Resolved input file, manifest, events and status summaries |
| `slurm.py` | Single-node scripts, source snapshots, dependencies and submission |
| `tracking.py` | MLflow tags, export and ingestion, including historical readers |
| `results.py` | MLflow result queries and comparison views |

`RunConfig` is configuration data. File writing and manifest construction belong
in `records`; MLflow formatting belongs in `tracking`. The former `runtime`,
`ray_backend`, `journal` and `ingest` modules have been replaced by these owners;
Python callers should use the current module paths.

`run.launch_run(run)` passes a resolved config to one Ray `TorchTrainer` worker.
`run.execute_run(run, ray=True)` records startup, builds and executes Lightning,
and saves success or failure. `run_lightning` supports the same `fit`/`test`
execution locally for diagnostics. Training dependencies remain lazy imports,
so configuration inspection does not load Torch or Lightning.

Each run saves `resolved-run.yml` and `.graphids/manifest.json`. Schema-2 manifests
hold authoritative status, metrics, result metadata and failure details. Events
record lifecycle history without another copy of the final metrics. Logger and
backend startup failures are recorded as failures as well as training failures.
Ray reports use the result directly; execution does not read MLflow exports back.

Offline runs write no new `mlflow_ingest.json`. `gx exp ingest` derives the export
from the saved config and schema-2 manifest. Historical schema-1 manifests and
separate MLflow ingest files remain readable. Run paths, input YAML fields,
model checkpoints and frame-score artifacts retain their existing contracts.

## 5. Slurm submission

`gx exp submit <yaml>` renders one node and one device; unsupported device counts
fail before files are written or jobs submitted. Multi-node bootstrap branches
have been removed. Cluster, partition, walltime and GRES overrides remain available.

The public Python API separates preparation from side effects:

```python
from graphids.exp.slurm import build_experiment_job, write_job, submit_script

job = build_experiment_job(
    cfg,
    source_root=frozen_source,
    log_dir=logs,
    modules=["cuda/12.8.1"],
    preflight=[["python", "verify.py", "--suite", suite_path]],
)
write_job(job)
receipt = submit_script(job.script_path, dependency="afterok:12345")
```

`build_experiment_job` validates once at preparation and returns the script plus
its expanded, content-named config. Preflight argv is shell-quoted and runs before
Ray starts. `write_job` writes both artifacts. `submit_script` handles parsable
scheduler receipts and cancellation on invalid dependencies for either a training
job or a recipe-specific gate/report script. `submit_experiment` combines these
steps for the CLI. `--dry-run` prints without writing or submitting.

`snapshot_source` freezes selected runtime source, including dirty and untracked
code, plus a Git provenance receipt. It refuses to overwrite an existing snapshot;
callers supply their runtime environment settings separately. `dependency_versions`
records the Python environment. The quality-suite recipe uses these APIs and
contains only its experiment definitions, provenance checks and scheduling policy.
The CLI also accepts `--source-root` and `--dependency`.

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
Constructor and typed-config defaults share `graphids/event_defaults.py` and
match the production preset. Historical factories and migration registries have
been removed; unknown component types fail during construction, and missing YAML
reports its path. Historical source remains in the verified snapshot archive.
Datasets, results, clock repairs and reviewed manifests remain in place.

## 7. Adding A Run

1. Create `configs/experiments/<name>.yml`.
2. Validate with `gx exp config <file>`.
3. For training, set `stage: fit`, callbacks, and trainer options.
4. For evaluation, set `stage: test` and provide `ckpt_path`.
5. Submit with `gx exp submit <file> -C ascend`.
6. After completion, run `gx exp ingest <run_dir>` or `gx exp ingest-root <root>`.

Add tests when the YAML exercises new launch behavior, component fields, or
representation semantics.
