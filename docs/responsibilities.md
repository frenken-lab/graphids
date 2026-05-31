# Module Responsibilities

**Experiment configs** (`configs/experiments/*.yml`) define one launchable
unit: dataset, stage, representation, resources, and the stage-specific
payload under `config`.

**Typed config** (`graphids/exp/config.py`) validates YAML and builds a
`RunConfig`. It owns the contract for `ExperimentConfig`, `ResourceConfig`,
stage payloads, output paths, MLflow tags/params, and the representation drift
check between top-level metadata and `data.source`.

**Primitives** (`graphids/primitives*.py`) are the public object factory
surface for YAML specs. Data, model, loss, scaler, representation, ID
encoding, and discovery primitives live here or are re-exported here.

**Experiment runtime** (`graphids/exp/runtime.py`) owns launch lifecycle and
stage dispatch. It writes the manifest/events journal, creates the MLflow
logger, builds config-driven objects, runs Lightning for `fit`/`test`,
materializes caches for `cache`, and calls extraction/analyzer code for
`extract`/`analyze`.

**SLURM submit** (`graphids/exp/slurm.py`, `graphids/cli/exp.py`) validates an
experiment YAML, renders an sbatch script, and submits it. The compute node
runs `python -m graphids exp launch <yaml>` after sourcing
`scripts/slurm/_preamble.sh`.

**Data sources and datamodules** (`graphids/core/data/`) own raw CAN loading,
representation selection, cache paths, materialization, metadata, and
Lightning dataloaders. `GraphDataModule(require_cache=True)` fails fast when a
training config expects a cache that is missing or incomplete.

**Preprocessing** (`graphids/core/data/preprocessing/`) turns raw rows into
materialized graph views. Snapshot, snapshot-sequence, multi-scale, temporal,
and entity representations are explicit. Snapshot-sequence materialization
stores sequence metadata on graph/node/edge tensors.

**Models** (`graphids/core/models/`) own Lightning modules and metrics. The
GAT now supports sequence-aware graph pooling through `sequence_pool`.

**Callbacks** (`graphids/core/callbacks.py`) hold graphids-specific Lightning
policy such as `Sha256ModelCheckpoint`, tau-norm, and VRAM drift warnings.

**MLflow** (`graphids/_mlflow.py`) resolves the shared tracking URI, builds the
Lightning MLflow logger, and starts/stops MLflow system-metrics monitoring for
Lightning-created runs.

The live flow is:

```text
configs/experiments/<run>.yml
    -> ExperimentConfig.from_yaml
    -> ExperimentConfig.build_run
    -> gx exp launch OR gx exp submit
    -> graphids.exp.runtime.launch_run
    -> run_stage: cache | fit | test | extract | analyze
    -> MLflow + .graphids/manifest.json + .graphids/events.jsonl
```

The old `graphids/plan` row renderer, `gx run`, `gx plans submit`, and
`graphids/orchestrate.py` row dispatcher are historical and should not be used
for new work.
