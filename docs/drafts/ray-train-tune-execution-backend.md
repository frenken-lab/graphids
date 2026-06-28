# Ray Train/Tune Backend Research And Migration Plan

Status: superseded by Ray-default experiment backend
Date: 2026-06-27

## Question

GraphIDS now has a small experiment layer:

- YAML experiment configs.
- domain primitives for data/model/loss specs.
- `graphids.exp.config` for typed run metadata.
- `graphids.exp.ray_backend` for Ray Train + Lightning `fit`/`test`.
- per-run `.graphids/` journals and offline MLflow ingest.
- Slurm script rendering in `graphids.exp.slurm`.

The question is whether Ray Train and Ray Tune can reduce this code burden,
especially because Ray already has training configs, resource configs,
callbacks, checkpointing, storage, and trial management.

## Research Summary

Ray Tune is broader than hyperparameter optimization. It is Ray's experiment
execution and trial-management layer. A `Tuner` combines:

- a trainable,
- `param_space`,
- `TuneConfig`,
- `RunConfig`,
- storage/checkpoint settings,
- search/scheduler choices.

This maps well to GraphIDS variant execution. `param_space` can express a grid,
random/search distributions, or fixed experiment variants.

Ray Train is closer to GraphIDS' current config-driven launch code. It provides:

- `ScalingConfig` for worker/GPU/CPU allocation,
- `RunConfig` for storage path, run name, callbacks, failure config, and
  checkpoint config,
- `CheckpointConfig` for checkpoint retention/scoring,
- `train.report(metrics, checkpoint=...)` for metrics/checkpoint reporting,
- `Result` objects for final metrics and checkpoints,
- PyTorch and Lightning helpers such as `TorchTrainer`,
  `ray.train.lightning.prepare_trainer`, `RayDDPStrategy`,
  `RayLightningEnvironment`, and `RayTrainReportCallback`.

Ray on Slurm is feasible, but it changes the execution model. Instead of one
Slurm job per experiment YAML, a Ray-backed path normally starts a Ray cluster
inside one Slurm allocation and lets Ray schedule workers/trials within that
allocation.

## Mapping To GraphIDS

| GraphIDS concept | Ray equivalent | Replace? |
|---|---|---:|
| `ResourceConfig` | `ray.train.ScalingConfig` | likely |
| `OutputConfig` run root/name | `ray.train.RunConfig(storage_path, name)` | likely |
| checkpoint retention/scoring | `ray.train.CheckpointConfig` | likely |
| final metric capture | `ray.train.report` / `Result.metrics` | likely |
| run result summary | `ray.train.Result` | likely |
| many variants/sweeps | `ray.tune.Tuner(param_space=...)` | likely |
| Slurm one-job-per-YAML variants | Ray cluster + Tune trial scheduler | maybe |
| `graphids.primitives_*` | no direct equivalent | keep |
| representation/cache validation | no direct equivalent | keep |
| `.graphids/manifest.json` | Ray result metadata plus custom files | partially keep |
| offline MLflow ingest | no direct equivalent | keep |
| model registry/promotion | no direct equivalent | keep or DVC/MLflow |

## What Is Actually Code Debt

Likely reducible:

- custom resource dataclass fields that duplicate `ScalingConfig`,
- parts of output/run location handling that duplicate `RunConfig`,
- checkpoint scoring/retention if moved to `CheckpointConfig`,
- result aggregation around final metrics/checkpoints,
- manual multi-variant Slurm submission if Ray Tune is reliable on OSC,
- old launch ceremony now that Ray Train owns the training controller.

Not code debt:

- primitives: these are GraphIDS' domain vocabulary,
- dataset catalog and representation/cache rules,
- model/loss/datamodule construction,
- offline MLflow ingest, unless replaced by a real registry/reporting backend,
- lightweight translation tests for smoke coverage without starting Ray.

## Recommended Direction

Ray is the current `exp` execution backend behind the existing domain config.

Target shape:

```text
GraphIDS YAML / primitives
  -> ExperimentConfig / RunConfig
  -> build data/model/loss
  -> backend:
       ray_train       (single train/test run)
       ray_tune        (many ray_train trials)
  -> per-run artifacts/results
  -> optional serialized MLflow ingest
```

Ray is now the stable default behind `gx exp launch` and `gx exp submit`:

```bash
gx exp launch configs/experiments/gat_temporal_smoke.yml
gx exp tune configs/sweeps/gat_grid.yml
```

The direct legacy launch/submit path has been removed from the public CLI.

## Proposed Migration Plan

### Phase 0: Pin And Probe Ray

Ray is now a required experiment dependency:

```toml
[project]
dependencies = ["ray[train,tune]>=2.55", ...]
```

Create a small local smoke script that imports:

- `ray.train.RunConfig`,
- `ray.train.ScalingConfig`,
- `ray.train.CheckpointConfig`,
- `ray.train.torch.TorchTrainer`,
- `ray.train.lightning.prepare_trainer`,
- `ray.train.lightning.RayTrainReportCallback`.

This avoids committing to APIs that are absent or deprecated in the installed
Ray version.

### Phase 1: Build Ray Objects Directly

Keep `graphids/exp/ray_backend.py` small and construct Ray objects directly:

```text
RunConfig.resources -> ScalingConfig
RunConfig.outputs   -> RunConfig(storage_path, name)
checkpoint settings -> CheckpointConfig
RunConfig.payload   -> train_loop_per_worker config
```

### Phase 2: Single-Run Ray Train Backend

Add:

```bash
gx exp launch <experiment.yml>
```

Initial behavior:

- build the same `RunConfig`,
- resolve data/model/loss specs inside `graphids.exp.ray_backend`,
- run a Ray `TorchTrainer` or Lightning-compatible train function,
- write `.graphids/manifest.json`,
- write `.graphids/events.jsonl`,
- write `.graphids/mlflow_ingest.json`,
- report final metrics/checkpoints through Ray.

For the first implementation, preserve existing checkpoint files if possible.
Ray checkpoint reporting can wrap those files later.

Acceptance criteria:

- local CPU smoke run works,
- one OSC Slurm single-node GPU smoke run works,
- output directory is deterministic and recoverable,
- offline MLflow ingest still works,
- no live shared SQLite writes from workers.

### Phase 3: Slurm Ray Allocation Script

Use the Ray Slurm script builder:

```bash
gx exp submit <experiment.yml>
```

This script should:

- request one or more nodes,
- start a Ray head and workers with `srun`,
- run the driver command once,
- write Ray logs under the Slurm log directory,
- keep `.graphids/` artifacts under the GraphIDS run directory.

This is operationally more complex than the removed one-task sbatch path, so
OSC smoke testing should happen before broad queue use.

### Phase 4: Ray Tune Sweeps

Add sweep configs only after Ray-backed `exp launch` is stable.

Proposed command:

```bash
gx exp tune configs/sweeps/gat_grid.yml
```

Sweep config should reference a base experiment YAML plus overrides:

```yaml
base: configs/experiments/gat_temporal_smoke.yml
name: gat_grid
metric: val_loss
mode: min
grid:
  config.model.scale: [small, large]
  config.loss_fn.type: [ce, focal]
  seed: [1, 2, 3]
```

The adapter should convert this to `ray.tune.Tuner(..., param_space=...)`.

Acceptance criteria:

- multiple variants run in one Ray allocation,
- each trial has an isolated run directory,
- results can be loaded from Ray's result grid,
- selected/all trial results can be mirrored through `gx exp ingest`.

### Phase 5: Delete Or Collapse Duplicated Code

Only after Phases 2-4 are stable:

- replace `ResourceConfig` fields with a thinner GraphIDS wrapper or direct
  Ray config translation,
- move checkpoint retention/scoring to `CheckpointConfig`,
- keep `graphids.exp.ray_backend` as the Ray launcher and worker loop,
- consider replacing parts of `graphids.exp.results` with Ray result loading
  for Ray-backed sweeps,
- keep offline MLflow ingest for searchable historical results.

## Risks And Open Questions

- OSC Slurm Ray startup may be the largest operational risk.
- Ray's Lightning APIs have moved over time; verify exact installed API before
  coding against examples.
- Ray result/checkpoint paths must coexist with GraphIDS run directories and
  existing checkpoint sidecars.
- Ray Tune trial scheduling may not match the current preference for fully
  independent Slurm jobs.
- Ray does not solve model registry or promotion. Keep MLflow/DVC/GraphIDS
  registry logic separate.
- Avoid reintroducing shared SQLite writes from Ray callbacks or trial loggers.

## Recommendation

Adopt Ray as the default execution backend.

The likely best long-term split is:

```text
GraphIDS owns domain config and primitives.
Ray Train owns training execution/resource/checkpoint machinery.
Ray Tune owns variant scheduling.
GraphIDS ingest owns MLflow/reporting interoperability.
```

This can reduce code debt, but only if Ray-on-Slurm is reliable in the target
OSC environment. The current `graphids.exp` backend is Ray Train; keep domain
config, primitives, and offline ingest separate from Ray-specific scheduling.
