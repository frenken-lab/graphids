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
`MLflowTrainingCallback`, `VRAMDriftCallback` when CUDA is available),
and returns an `InstantiatedRun(trainer, model, datamodule)`.

**Stage primitives** (`graphids/orchestrate/stage.py`) — `build`, `train`,
`evaluate`. `fit` / `test` call these directly. No pipeline driver. Multi-stage
chains are declared in a *plan jsonnet* (`configs/plans/*.jsonnet`); `python
-m graphids run <plan>` parses it via `graphids.slurm.dag` (Pydantic),
toposorts, and calls `graphids.slurm.submit.submit()` per node with `dep_jids`
afterok chaining held in memory.

**SLURM** (`graphids/slurm/`, `graphids/cli/submit.py`) — resource allocation
and job submission. One Typer command: `python -m graphids submit --row <json>`
submits a single blueprint row. Library entrypoint is
`graphids.slurm.submit.submit_row()`. Reads
`configs/resources/submit_profiles.json` keyed `[mode][cluster][length]`
where each leaf is a `parsl.providers.SlurmProvider` kwargs dict.

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
