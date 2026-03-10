# KD-GAT Session Plan

> Last updated: 2026-03-10

## Priority: Scheduler-Agnostic Orchestration Validation

Replaced monolithic `coordinator.py` + JSON `state.py` with a 5-component scheduler-agnostic orchestration system (job.py, planner.py, store.py, executor.py, driver.py). CLI: `python -m graphids.pipeline.cli orchestrate`.

### Next Steps

1. ~~**Orchestration architecture + implementation (Phases 1-4)**~~ — done (2026-03-10)
2. ~~**Migrate sweep_pipeline.py off state.py**~~ — done (2026-03-10)
3. ~~**Delete stale coordinator.py + state.py**~~ — done (2026-03-10)
4. **Write orchestration tests** — unit tests for store.py, planner.py, driver.py (DryRun backend)
5. **Dry-run validation** — `python -m graphids.pipeline.cli orchestrate --dataset hcrl_sa --seeds 42 --dry-run`
6. **Live single-dataset validation** — single dataset/seed via `orchestrate` on SLURM
7. **Full sweep** — all datasets × 3 seeds × 3 variants via orchestrate
8. **Flux backend** — implement at LLNL internship (summer 2026)

## In Progress

- Ops dashboard (`buckeyeguy/kd-gat-dashboard`) — running on HF Spaces (Streamlit SDK). Shows 181 experiment runs + 37 sweep trials.

## Blocked

(none)

## Open Questions

### Is RL justified for fusion?

See `~/plans/fusion-redesign.md` for full analysis.

## Next Up

- Run full sweep via coordinator (all datasets × 3 seeds)
- Fusion method comparison experiment (see Open Questions above)
- Evaluate research questions R1–R3
- **Research visualization Space** — plan and build a second HF Space (`buckeyeguy/kd-gat-research`) for paper-ready figures. Replaces the deleted Quarto `reports/` dashboard with a Streamlit + Plotly stack. Needs a dedicated planning session covering:
  - Which old Quarto pages to rebuild (Performance, Training, GAT & DQN, KD Transfer, Loss Landscape, Graph Structure — skip Staging/Datasets)
  - What additional artifacts `push_experiments_to_hf.py` must export (attention weights, embeddings, CKA similarity, DQN policy, loss landscape, graph samples — currently only flat metrics are pushed)
  - Data schema: single expanded Parquet vs multiple files on HF Dataset
  - Plotly equivalents for each Mosaic/vgplot spec (most are straightforward; loss landscape 3D surface and force-directed graph need care)
  - Whether `notebooks/analysis/` code can be directly adapted or needs rewrite
  - Separation of concerns: ops dashboard (existing) vs research dashboard (new)

## Completed

- Shared PostgreSQL backend for pipeline state: Apptainer-containerized PostgreSQL 16 on SLURM (`scripts/lab-db/pg-server.sbatch`), on-demand launcher (`scripts/lab-db/ensure_pg.sh`), dual-backend PipelineStore (SQLite + PostgreSQL). PGDATA on node-local SSD, NFS backup/restore, idle auto-shutdown, `psycopg[binary]` optional dep. Fixes NFS-unsafe SQLite for concurrent writers. (2026-03-10)
- Scheduler-agnostic orchestration: 5-component system (job/planner/store/executor/driver) replacing monolithic coordinator.py + JSON state.py. UUID-based DAG, SQLite state, SLURM+Flux+DryRun backends, retry scaling, fire-and-forget mode. (2026-03-10, supersedes coordinator from 2026-03-09)
- Memory/batch sizing simplification: memory.py 471→25 lines, batch_sizing.py 169→43 lines. Deleted custom GPU memory estimation (static, measured, trial modes, forward hooks, binary search, budget caching). Batch size now config-driven with safety_factor. `_compute_metrics()` simplified with `classification_report()`. CKA `_save_cka()` deduplicated. tune_config.py batch sizing inlined to `resolve_batch_config()`. ~600 lines removed (2026-03-07)
- Legacy removal: deleted export.py, sweep_export.py, tracking.py, reports/ (Quarto), verify-site skill, SLURM export scripts. Removed Quarto CI jobs. Updated all docs/rules (2026-03-07)
- DQN `from_config()` factory: replaces 3 verbose 15-param construction sites in evaluation.py, serve.py, fusion.py (2026-03-07)
- Inlined `get_memory_summary()` into callbacks.py (was only caller of tracking.py) (2026-03-07)
- Codebase simplification: dedup `_make_conv` to `_utils.py`, extract shared `training_preamble`/`resolve_batch_config`, delete dead `_encode_spatial`, move DQN TODO to `~/plans/fusion-redesign.md`, STAGE_DEPENDENCIES constant, SLURM `log_job_header`/`log_job_footer`, clean bare exceptions in export.py, remove unused `log_path_str`, deprecation warning on legacy flat config (2026-03-07)
- MLflow cleanup: removed datalake/lakehouse/W&B/S3 dead code, rewrote export.py + sweep_export.py to MLflow-only, updated all docs (2026-03-06)
- MLflow consolidation: replaced W&B + lakehouse.py + CSVLogger with MLflow single store (2026-03-06)
- Spec-driven visualization migration: dashboard.qmd 1354→445 lines, 29 YAML specs (2026-03-06)
- Force graph & visualization rework: 7 color modes, edge tooltips, attack_type support (2026-03-04)
- `export_graph_samples()` + v2 JSON schema (2026-03-04)
- `include_attack_type=True` default in preprocessing pipeline (2026-03-04)
- evaluation.py: attack_type capture in embeddings.npz (2026-03-04)
- `rebuild_all_caches.sh` + `build_test_cache.sh` fixes (2026-03-04)
- Feature engineering v2.0.0: 15 new node features (11→26-D), GATv2 switch, paper updated (2026-03-03)
- Loss landscape stage fixes + dashboard tab (2026-03-03)
- SLURM account migration PAS3209→PAS1266 (2026-03-03)
- (older items in git log)
