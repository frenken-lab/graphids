# Module Responsibilities

**Jsonnet** (`configs/`) — structure and composition only. Stage functions produce a raw
merged dict. No validation, no types.

**`render`** (`graphids/config/jsonnet.py`) — `_jsonnet` C-binding call with typed
`tla_codes` args (JSON-encoded so ints/bools/null round-trip correctly).
Returns the rendered dict.

**Pydantic / `validate_config`** (`graphids/config/schemas.py`) — validation gate immediately
after render. Catches null list fields, monitor/mode mismatches, un-namespaced class_paths,
and LearningRateMonitor without a logger. Fails fast before any torch import.

**`instantiate`** (`graphids/orchestrate/instantiate.py`) — imports class_paths via
importlib, applies signature-filtered link_arguments, builds forced callbacks
(`ModelCheckpoint`, `EarlyStopping`, `OTelTrainingCallback`, `VRAMDriftCallback`
when CUDA is available), wires `OTelTrainingLogger`, and returns a wired
`(trainer, model, datamodule)` triple.

**Pipeline driver** (`graphids/orchestrate/run.py`, `graphids/cli/pipeline.py`) —
`run_pipeline(config)` composes `build_pipeline_stages` (planner) → for each
stage: `ResolvedConfig.resolve` → `stage.build` → `stage.train` → `stage.evaluate`.
Runs in-process inside whatever SLURM allocation `submit.sh pipeline-run` hands
it. Skips a stage when `checkpoints/best_model.ckpt` is already on disk.
Analysis is intentionally not part of the driver — run `python -m graphids
analyze --ckpt-path <p>` once training is done.

**SLURM** (`graphids/slurm/`, `scripts/slurm/submit.sh`) — resource allocation and job
submission. CPUs, GPUs, memory, wall time. All jobs submitted via `submit.sh <profile>`.

The pipeline is strictly one-directional:

```
jsonnet renders (render)
    ↓
Pydantic validates (validate_config → ValidatedConfig)
    ↓
instantiate → (trainer, model, datamodule)
    ↓
trainer.fit / trainer.test
```

> Authoritative detail: `.claude/rules/config-system.md`
