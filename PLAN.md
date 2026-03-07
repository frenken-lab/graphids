# KD-GAT Session Plan

> Last updated: 2026-03-04

## Priority: Wait for Cache Rebuild → Export → Verify

Cache rebuild jobs submitted (SLURM 44485258-63) with `include_attack_type=True`.
Once complete, run export and verify visualizations.

### Post-Cache Steps

```bash
# 1. Check job status
squeue -u $USER

# 2. Verify caches have attack_type metadata
.venv/bin/python -c "
import torch
g = torch.load('data/cache/hcrl_ch/processed_graphs.pt', map_location='cpu', weights_only=False)
g0 = g[0] if isinstance(g, list) else g.data_list[0]
print(f'x: {g0.x.shape}, edge_attr: {g0.edge_attr.shape}')
print(f'attack_type: {g0.attack_type}, node_y: {g0.node_y.shape}')
print(f'id_entropy: {g0.id_entropy}')
"

# 3. Run export (generates graph_samples.json v2)
.venv/bin/python -m graphids.pipeline.export

# 4. Render reports
quarto render reports/

# 5. Verify in browser (WSL only)
quarto preview reports/
```

### Verification Checklist

- [ ] Cache rebuild completes for all 6 datasets (check SLURM logs)
- [ ] Cached graphs have `attack_type`, `node_attack_type`, `node_y`, `id_entropy`, 26-D `x`, 11-D `edge_attr`
- [ ] `python -m graphids.pipeline.export` produces `graph_samples.json` v2 schema
- [ ] `quarto render reports/` succeeds (16/16 pages)
- [ ] Browser: force graph renders with all 7 color modes
- [ ] Browser: edge tooltips show 11-D features
- [ ] Browser: color legend correct per mode
- [ ] Browser: 0 JS console errors
- [ ] Paper figure renders with attack_type coloring

## In Progress

- Cache rebuild (SLURM jobs 44485258-63, CPU partition, ~5-20 min each)

## Blocked

(none)

## Open Questions

### Is RL justified for fusion?

See `~/plans/fusion-redesign.md` for full analysis.

## Next Up

- Run full pipeline retrain on rebuilt caches
- Run tests: `bash scripts/slurm/run_tests_slurm.sh`
- Fusion method comparison experiment (see Open Questions above)
- Loss landscape analysis on retrained models
- Evaluate research questions R1–R3

## Completed

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
