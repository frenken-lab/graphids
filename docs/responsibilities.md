# Module Responsibilities

**Jsonnet** (`configs/`) — structure and composition only. Stage functions produce a raw
merged dict. No validation, no types.

**`render_config`** (`graphids/config/jsonnet.py`) — subprocess shim that calls the `jsonnet`
binary with typed `--tla-code` args (JSON-encoded so ints/bools/null round-trip correctly).
Returns the rendered dict.

**Pydantic / `validate_config`** (`graphids/config/schemas.py`) — validation gate immediately
after render. Catches null list fields, monitor/mode mismatches, un-namespaced class_paths,
and LearningRateMonitor without a logger. Fails fast before any torch import.

**`instantiate`** (`graphids/instantiate.py`) — imports class_paths via importlib,
applies signature-filtered link_arguments, builds forced callbacks (ModelCheckpoint,
EarlyStopping, OTelTrainingCallback), wires OTelTrainingLogger, and returns
a wired `(trainer, model, datamodule)` triple.

**Monarch** (`graphids/orchestrate/monarch.py`, `graphids/cli/_monarch.py`) — custom
SLURM-backed pipeline orchestrator. Enumerates `StageConfig`s from recipes, calls
`ResolvedConfig.resolve` per stage, submits SLURM jobs via torchmonarch actors.

**SLURM** (`graphids/slurm/`, `scripts/slurm/submit.sh`) — resource allocation and job
submission. CPUs, GPUs, memory, wall time. All jobs submitted via `submit.sh <profile>`.

The pipeline is strictly one-directional:

```
jsonnet renders (render_config)
    ↓
Pydantic validates (validate_config → ValidatedConfig)
    ↓
instantiate → (trainer, model, datamodule)
    ↓
trainer.fit / trainer.test
```

> Authoritative detail: `.claude/rules/config-system.md`
