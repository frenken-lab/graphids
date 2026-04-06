# Copilot instructions for KD-GAT (GraphIDS)

## Build, test, lint
- **Lint:** `ruff check graphids/ tests/`
- **Tests (SLURM required):** `scripts/slurm/submit.sh tests`
- **Single test:** `scripts/slurm/submit.sh tests -k <pattern>`

## High-level architecture
- **Config pipeline:** Jsonnet stage configs (`configs/stages/*.jsonnet`) are the single source of composition. Every path renders configs via `graphids.config.jsonnet.render_config(...)`, validates with `graphids.config.schemas.validate_config(...)`, then instantiates the Lightning stack through `graphids.instantiate.instantiate(...)`.
- **CLI routes:** Typer-based CLI in `graphids/cli/` — `app.py` defines the root app, submodules (`_training.py`, `_analysis.py`, `_data.py`, `_orchestrate.py`, `_slurm.py`) register commands via `@app.command()`. `graphids/__main__.py` imports the submodules to register all commands.
- **Pipeline path:** Dagster materializes assets (`graphids.orchestrate.dagster`), `ConfigResolver` (`orchestrate/resolve/resolver.py`) builds TLAs, then SLURM jobs run `python -m graphids from-spec` with the rendered spec.
- **SLURM submission:** All jobs (tests, validation, profiling, cache rebuilds, etc.) are launched through `scripts/slurm/submit.sh`, which reads resource profiles from `configs/resources/*.json`.

## Key conventions
- **Logging:** use `from graphids.log import get_logger` and `log.info("event_name", key=value)` (structured kwargs, no format strings).
- **Jsonnet stages:** every `configs/stages/*.jsonnet` is a top-level function with defaults. Adding a new TLA requires updating both the jsonnet signature and `build_tla_dict` in `graphids.orchestrate.contracts`.
- **Jsonnet merge semantics:** nested merges must use `+:` (a bare `+` on nested dicts replaces the subtree). Lists replace on conflict.
- **Tests on HPC:** never run `pytest` directly on login nodes; submit via `scripts/slurm/submit.sh tests` and use `@pytest.mark.slurm` only for tests that actually train or require CUDA.
- **PyTorch/PyG safety:** `Data.to()` is in-place—use `data.clone().to(device)`; always use spawn multiprocessing for DataLoaders; save/restore `model.training` around `model.eval()`; clamp skewness/kurtosis features to ±10 to avoid fp16 overflow.
- **Run records:** training runs write `{run_dir}/run_record.json`; run directories are content-addressed via `compute_identity_hash()` and missing identity keys should raise.
