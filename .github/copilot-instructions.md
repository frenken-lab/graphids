# Copilot instructions for GraphIDS

## Build, test, lint
- **Lint:** `ruff check graphids/ tests/`
- **Tests (SLURM required):** author a one-row plan under `configs/plans/ops/`
  invoking `python -m pytest`, then `graphids run | submit`. Never `pytest` on
  login nodes (`.claude/rules/slurm-hpc.md`).

## High-level architecture
- **Four-step chassis (`.claude/rules/single-submission-primitive.md`):**
  `render → blueprint → exec → submit`. `graphids run <plan.jsonnet> -o
  plan.json` renders the plan, validates as `BlueprintArray`, writes a JSON
  array. `graphids exec --row <json>` runs one row in-process via
  `graphids.orchestrate.run_row`. `graphids submit --row <json> --cluster
  <c>` submits one row to SLURM via Parsl `SlurmProvider`. The sbatch script
  carries the literal `python -m graphids exec --row '<json>'` command — no
  pickle.
- **Config route (single):** `graphids.config.jsonnet.render(path, tla=...)`
  returns a dict; `graphids.blueprint.{TrainRow,BlueprintArray}` validates
  it. `graphids.orchestrate.run_row` walks `class_path` blocks and
  instantiates via `importlib` with signature-filtered kwargs. `run_dir` is
  computed inside the plan jsonnets themselves from `run_root` + `dataset` +
  `group` + `variant` + `seed` TLAs (no Python `paths.py`).
- **Training loop:** `lightning.pytorch.Trainer`. `_ModelBase` inherits from
  `pl.LightningModule`; callbacks inherit from `pl.Callback`. The custom
  `core/trainer.py`, `core/_metric_acc.py`, and `core/_ckpt.py` modules were
  removed in the 2026-05-02 Lightning migration (commit `c974185`).
  graphids-specific callbacks live in `graphids.core.callbacks`
  (`Sha256ModelCheckpoint`, `TauNormCallback`, `VRAMDriftCallback`);
  `MLflowTrainingCallback` lives in `graphids._mlflow`.
- **CLI:** Typer-based in `graphids/cli/` — `app.py` owns the root app;
  submodules (`run.py`, `exec.py`, `submit.py`, `analysis.py`, `data.py`,
  `export.py`) register commands via `@app.command()`. `graphids/__main__.py`
  imports the submodules to register all commands.

## Key conventions
- **Logging:** `structlog` configured by `graphids/runtime.py:_configure_logging`
  (auto-injects SLURM context). Use
  `from structlog import get_logger; log = get_logger(__name__)` and
  `log.info("event_name", key=value)`.
- **Jsonnet merge semantics:** nested merges must use `+:` (a bare `+` on
  nested dicts replaces the subtree). Lists replace on conflict. Use
  `~/.local/bin/jsonnet <path>` to verify after edits.
- **MLflow:** the only telemetry sink for cross-run analysis. Hand-composed
  `filter_string=` is banned — go through
  `graphids._mlflow.build_search_filter(...)`. See
  `.claude/rules/data-layout.md` for the store-ownership table.
- **PyTorch/PyG safety (`.claude/rules/critical-constraints.md`):** `Data.to()`
  is in-place — use `data.clone().to(device)`; always `spawn` multiprocessing
  for DataLoaders with CUDA; save/restore `model.training` around
  `model.eval()`; clamp skewness/kurtosis features to ±10 and VGAE `logvar`
  to ±10 to avoid fp16 overflow.
