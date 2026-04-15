# Module Responsibilities

**Jsonnet** (`configs/`) — structure and composition only. Every preset
under `configs/ablations/*.jsonnet` produces a raw merged dict and
computes its own `run_dir`. No validation, no types.

**`render`** (`graphids/config/jsonnet.py`) — `_jsonnet` C-binding call with typed
`tla_codes` args (JSON-encoded so ints/bools/null round-trip correctly).
Returns the rendered dict.

**Pydantic / `validate_config`** (`graphids/config/schemas.py`) — validation gate immediately
after render. Catches null list fields, monitor/mode mismatches, un-namespaced class_paths,
and LearningRateMonitor without a logger. Fails fast before any torch import.

**`build_run`** (`graphids/orchestrate/instantiate.py`) — imports class_paths via
importlib, applies `filter_kwargs` against each target's `__init__`
signature, builds forced callbacks (`ModelCheckpoint`, `EarlyStopping`,
`OTelTrainingCallback`, `VRAMDriftCallback` when CUDA is available), wires
`OTelTrainingLogger`, and returns an `InstantiatedRun(trainer, model,
datamodule)`.

**Stage primitives** (`graphids/orchestrate/stage.py`) — `build`, `train`,
`evaluate`. `fit` / `test` call these directly. No pipeline driver. Multi-stage
chains are bash loops submitting each preset with `SBATCH_DEP=afterok:<jid>`.

**SLURM** (`graphids/slurm/`, `scripts/run`, `scripts/slurm/submit.sh`) —
resource allocation and job submission. `scripts/run` launches training
presets; `submit.sh` covers non-training jobs (tests, rebuild-caches,
analyze, etc.). Both read `configs/resources/submit_profiles.json`.

The pipeline is strictly one-directional:

```
jsonnet renders (render)
    ↓
Pydantic validates (validate_config → ValidatedConfig)
    ↓
ResolvedConfig.from_rendered → build_run → (trainer, model, datamodule)
    ↓
trainer.fit / trainer.test
```

> Authoritative detail: `.claude/rules/config-system.md`
