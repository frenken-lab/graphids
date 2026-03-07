# KD-GAT Session Plan

> Last updated: 2026-03-07

## Priority: Verify Simplification Branch → Tests → Merge

Branch `simplify-codebase` removes legacy code and reduces custom infrastructure by leveraging library features.

### Verification Steps

```bash
# 1. Import check (login node safe)
python -c "from graphids.pipeline.memory import log_memory_state; print('memory OK')"
python -c "from graphids.pipeline.stages.batch_sizing import resolve_batch_config; print('batch OK')"
python -c "from graphids.pipeline.stages.evaluation import evaluate; print('eval OK')"

# 2. Tests via SLURM (required)
bash scripts/slurm/run_tests_slurm.sh
```

## In Progress

- Simplification branch: ready for test + merge (see Completed below)

## Blocked

(none)

## Open Questions

### Is RL justified for fusion?

See `~/plans/fusion-redesign.md` for full analysis.

## Next Up

- Run full pipeline retrain on rebuilt caches
- Run tests: `bash scripts/slurm/run_tests_slurm.sh`
- Fusion method comparison experiment (see Open Questions above)
- Evaluate research questions R1–R3

## Completed

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
