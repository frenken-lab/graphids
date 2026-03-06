# KD-GAT Architecture Decisions

> Import hierarchy: See code-style.md (enforced by tests/test_layer_boundaries.py).

## Config Architecture

- Pydantic v2 frozen BaseModels + YAML composition + JSON serialization.
- Sub-configs: `cfg.vgae`, `cfg.gat`, `cfg.dqn`, `cfg.training`, `cfg.fusion`, `cfg.temporal` — nested Pydantic models. Always use nested access, never flat.
- Auxiliaries: `cfg.auxiliaries` is a list of `AuxiliaryConfig`. KD is a composable loss modifier, not a model identity. Use `cfg.has_kd` / `cfg.kd` properties.
- Constants: domain/infrastructure constants live in `config/constants.py` (not in PipelineConfig). Hyperparameters live in PipelineConfig.

> Experiment tracking (W&B, datalake, DuckDB): See experiment-tracking.md.

## Orchestration

- Ray (`graphids/pipeline/orchestration/`) with `@ray.remote` tasks. `train_pipeline()` fans out per-dataset work concurrently.
- `--local` flag uses Ray local mode. HPO via Ray Tune with OptunaSearch + ASHAScheduler.
- **Subprocess dispatch**: Each stage runs as `subprocess.run()` for clean CUDA context. Overhead (~3-5s) is negligible vs training time (minutes-hours). CUDA contexts (~300-500 MB) are not reclaimable without process restart.
- **Per-stage granularity**: Finer (per-epoch) has massive scheduling overhead; coarser (per-variant) loses ability to re-run single stages.
- **Checkpoint passing**: Filesystem paths, not Ray object store (subprocesses can't access object store; checkpoints are small; path-based is debuggable).
- **Concurrent variants**: `small_nokd_pipeline` launches concurrently with `large_pipeline` (no teacher checkpoint dependency). On single-GPU, Ray serializes GPU tasks automatically; on multi-GPU, enables true parallelism.
- **No Ray Data**: Datasets fit in memory and PyG's heterogeneous graph `Data` objects are incompatible with Ray Data's Arrow-based tabular format.
- Archive restore: `graphids/pipeline/cli.py` archives previous runs before re-running, restores on failure.
- **Benchmark mode**: Set `KD_GAT_BENCHMARK=1` to log per-stage spawn overhead, execution time, inter-stage gaps, and GPU utilization to JSONL. See `scripts/profiling/benchmark_orchestration.sh`.

### Orchestration Design Rationale
Subprocess-per-stage kept for CUDA context isolation (~300-500 MB per model), fault isolation, and stage-level restartability. Overhead (~3-5s) is <0.1% of pipeline wall time.
Full analysis: `~/plans/orchestration-redesign-decision.md`

## Inference Serving

`graphids/pipeline/serve.py` — FastAPI endpoints (`/predict`, `/health`) loading VGAE+GAT+DQN from `experimentruns/`.

## Dashboard & Reports

**Dashboard: Quarto** (`reports/dashboard.qmd`, ~570 lines). Spec-driven multi-page dashboard. ~25 charts rendered declaratively from YAML specs in `reports/figures/` via `renderSpec()`. Remaining charts use `renderTable()` (SQL → table) or thin OJS stubs (reactive logic). Data loaded from `reports/data/` (Parquet + JSON) into DuckDB-WASM. Pages: Overview, Performance, Training, GAT & DQN, Knowledge Distillation, Loss Landscape, Graph Structure, Datasets, Staging. See `reports/FIGURES.md` for the full chart registry.

**YAML specs:** Declarative Mosaic vgplot specs in `reports/figures/*.yaml`, rendered by `renderSpec()` from `_ojs/mosaic-renderer.js`. Supports YAML (via js-yaml CDN) and JSON. Specs define `data` (file or SQL query), `params` (selections), `vconcat`/`plot` arrays. The `skipDataLoad: true` option skips file-based data loading when tables are already in DuckDB (dashboard init block). Specs with SQL-derived tables (`data.*.query`) use default mode (no skipDataLoad).

**Paper:** `reports/paper/` contains the full research paper (10 chapters). Chapters use `{{< include _setup.qmd >}}` (Quarto shortcode) to initialize Mosaic/vgplot. Paper figures now reference shared YAML specs in `reports/figures/` (via `../figures/fig-*.yaml`). Paper data lives in `reports/paper/data/` (CSVs) and `reports/data/` (Parquet + JSON).

Dashboard data: `graphids/pipeline/export.py` exports leaderboard, runs, metrics, training curves, datasets, KD transfer, model sizes (JSON + Parquet), graph samples, graph statistics, and pre-computed graph layouts (`graph_nodes.parquet`, `graph_edges.parquet` via `spring_layout`) (~2s, login node safe) directly to `reports/data/`. `export_data_for_reports()` copies datalake Parquet to `reports/data/`. Heavy analysis (UMAP, attention, CKA, etc.) lives in `notebooks/analysis/`.

**Playground:** Two-tier prototyping — `notebooks/playground.ipynb` (pyobsplot + DuckDB Python, rapid inline iteration) → `reports/playground.qmd` (Mosaic/vgplot + DuckDB-WASM, production preview). See `~/plans/playground-conventions.md` for shared patterns.

**Deployment:** GitHub Actions renders Quarto on push to main and deploys via `actions/deploy-pages` (not gh-pages branch). CI: lint → test → quarto-build → deploy. Mosaic/vgplot loaded from jsdelivr CDN (`@uwdata/vgplot@0.21.1`).

**Verification caveat:** `quarto render` only proves `.qmd` → HTML compilation — it does NOT execute OJS/JS. Mosaic/vgplot bugs (DuckDB-WASM init, CDN failures, API misuse) are runtime-only. **Use `/verify-site` skill** for comprehensive Playwright-based verification. Key gotchas: (1) dashboard tabs lazy-load — must click all 9 tabs; (2) restart HTTP server on a new port after each re-render; (3) paper needs 15s wait for async OJS. See `~/.claude/rules/tooling.md` → Playwright Capabilities.

**Mosaic mark constraint:** vgplot@0.21.1 does NOT have `boxX`/`boxY`. Use `dotX`/`dotY` for distribution plots. `tickX`/`tickY` parse OK but crash during render. Always test new mark types in browser before committing.

## General Principles

- Delete unused code completely. No compatibility shims or `# removed` comments.
- Dataset catalog: `graphids/config/datasets.yaml` — single place to register new datasets.
