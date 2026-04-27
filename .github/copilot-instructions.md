# Copilot instructions for GraphIDS

## Build, test, lint
- **Lint:** `ruff check graphids/ tests/`
- **Tests (SLURM required):** `python -m graphids submit --mode cpu --length short --command "python -m pytest"`
- **Single test:** `python -m graphids submit --mode cpu --length short --command "python -m pytest -k <pattern>"`

## High-level architecture
- **Config route (single):** every preset under `configs/ablations/*.jsonnet` is a top-level function that computes its own `run_dir` via `std.native('paths.run_dir')(...)` (Python source: `graphids/config/paths.py`, registered as native callbacks in `graphids/config/jsonnet.py:render`). `python -m graphids fit --config <preset>` renders via `render`, validates via `graphids.config.schemas.validate_config`, wraps in `ResolvedConfig.from_rendered`, and instantiates via `graphids.orchestrate.instantiate.build_run`. PyTorch Lightning was dropped — the project uses a custom `graphids.core.trainer.Trainer`.
- **CLI:** Typer-based in `graphids/cli/` — `app.py` owns the root app; submodules (`training.py`, `analysis.py`, `data.py`, `submit.py`, `run.py`, `compare.py`) register commands via `@app.command()`. `graphids/__main__.py` imports the submodules to register all commands.
- **Multi-stage chains:** Declared in a *plan jsonnet* (`configs/plans/*.jsonnet`); `python -m graphids run <plan>` parses via `graphids.slurm.dag.parse_plan` (Pydantic), toposorts, and calls `graphids.slurm.submit.submit()` per node — `dep_jids` held in memory, no subprocess. `python -m graphids status <plan>` queries MLflow per node; FINISHED nodes are skipped on re-run (idempotent).
- **SLURM submission:** one Typer command, `python -m graphids submit` (library: `graphids.slurm.submit.submit()`). Training: `python -m graphids submit <preset.jsonnet>`. Ops: `python -m graphids submit --mode {gpu|cpu} --command "..." [--mem-gb M --timeout-min T]`. Profile JSON (`configs/resources/submit_profiles.json`) stores raw `submitit.AutoExecutor` kwargs keyed `[mode][cluster][length]` — per-job overrides are flags, never JSON.

## Key conventions
- **Logging:** use `from graphids._otel import get_logger` and `log.info("event_name", key=value)` (structured kwargs, no format strings). Handlers/level are installed by the Typer root callback (`cli/app.py`) via `init_providers()`; add `--verbose/-v` to the CLI to bump the graphids logger to DEBUG.
- **Jsonnet presets:** every `configs/ablations/**/*.jsonnet` is a top-level function with defaults. Adding a new launcher-level TLA requires updating the jsonnet signature + the matching flat flag in `graphids/cli/submit.py` (which builds the `flag_tlas` list inline — there is no separate helper).
- **Jsonnet merge semantics:** nested merges must use `+:` (a bare `+` on nested dicts replaces the subtree). Lists replace on conflict.
- **Tests on HPC:** never run `pytest` directly on login nodes; submit via `python -m graphids submit --mode cpu --length short --command "python -m pytest"` and use `@pytest.mark.slurm` only for tests that actually train or require CUDA.
- **PyTorch/PyG safety:** `Data.to()` is in-place — use `data.clone().to(device)`; always use spawn multiprocessing for DataLoaders; save/restore `model.training` around `model.eval()`; clamp skewness/kurtosis features to ±10 to avoid fp16 overflow.
