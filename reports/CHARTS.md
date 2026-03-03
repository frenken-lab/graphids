# KD-GAT Dashboard Chart Inventory

Developer reference for all charts in `dashboard.qmd`. Use this to trace data dependencies when column names change in `export.py`.

## Quick Reference

| Page | Chart | Type | Data Source | Key Columns | Library |
|------|-------|------|-------------|-------------|---------|
| Overview | Value Boxes (x4) | valuebox | runs, metrics (Parquet), kd_transfer.json | total_runs, f1, model, metric_name | OJS computed |
| Overview | Leaderboard | table | metrics.parquet | dataset, model, f1, accuracy, precision, recall, auc, mcc | Observable Inputs |
| Overview | Dataset Comparison | barX | metrics.parquet (via leaderboard_data) | dataset, model, [selected metric] | Observable Plot |
| Overview | F1 Dot Plot | dot | metrics.parquet | run_id (split), f1, model | Mosaic vgplot |
| Overview | Model Parameter Counts | barX | model_sizes.json | model_type, scale, param_count_M | Observable Plot |
| Performance | Bubble Chart (F1 vs Accuracy) | dot | metrics.parquet + model_sizes.json | f1, accuracy, model, param_count_M | Observable Plot |
| Performance | Pareto Frontier | dot + line | metrics.parquet + model_sizes.json | f1, model, param_count_M | Observable Plot |
| Performance | F1 vs AUC | dot | metrics.parquet | auc, f1, model | Mosaic vgplot |
| Performance | Runs by Dataset | barX (stacked) | runs.parquet | dataset (from run_id), model_type | Observable Plot |
| Training | Training Duration | barX | runs.parquet | run_id, model_type, duration_seconds, started_at, completed_at | Observable Plot |
| Training | Training Curve Comparison | line | training_curves.parquet | epoch, metric_name, value, run_id | Observable Plot |
| Training | Training Carpet | cell (heatmap) | training_curves.parquet | epoch, metric_name, value, run_id | Mosaic vgplot |
| GAT & DQN | Attention Heatmap | cell (heatmap) | attention_weights.parquet | head, layer, mean_alpha, run_id | Mosaic vgplot |
| GAT & DQN | DQN Alpha Distribution | rectY (histogram) | dqn_policy.parquet | alpha, run_id | Mosaic vgplot |
| GAT & DQN | Training Curves (single) | lineY | training_curves.parquet | epoch, value, metric_name, run_id | Mosaic vgplot |
| GAT & DQN | VGAE Recon Errors | rectY (histogram) | recon_errors.parquet | error, label, run_id | Mosaic vgplot |
| Knowledge Distillation | Teacher vs Student | dot | kd_transfer.json | teacher_value, student_value, dataset, model_type, metric_name | Observable Plot |
| Knowledge Distillation | CKA Similarity | cell (heatmap) | cka_similarity.parquet | student_layer, teacher_layer, similarity, run_id | Mosaic vgplot |
| Knowledge Distillation | UMAP Embeddings (VGAE) | dot | embeddings.parquet | x, y, label, run_id | Mosaic vgplot |
| Knowledge Distillation | UMAP Embeddings (GAT) | dot | embeddings.parquet | x, y, label, run_id | Mosaic vgplot |
| Graph Structure | CAN Bus Force Graph | force-directed | graph_samples.json | nodes (features), edges, dataset, label | D3 (force-graph.js) |
| Datasets | Dataset Catalog | table | datasets.json | name, domain, protocol, source, description | Observable Inputs |
| Staging | Run Timeline | barX (Gantt) | runs.parquet | run_id, model_type, stage, started_at, completed_at | Observable Plot |
| Staging | Model Predictions Summary | table | metrics.parquet | model, f1, accuracy, precision, recall, auc, mcc, n_samples, run_id | Observable Inputs |
| Staging | Confusion Matrix Proxy | dot | metrics.parquet | fpr, fnr, model, dataset (from run_id) | Observable Plot |

## Data Files

| File | Format | Source | Charts Using It |
|------|--------|--------|----------------|
| `data/metrics.parquet` | Parquet | `export.py` (datalake) | Leaderboard, F1 Dot, Bubble, Pareto, F1 vs AUC, Predictions, Confusion |
| `data/runs.parquet` | Parquet | `export.py` (datalake) | Value Boxes, Runs by Dataset, Duration, Timeline |
| `data/training_curves.parquet` | Parquet | `export.py` (merged shards) | Curve Comparison, Carpet, Training Curves (single) |
| `data/attention_weights.parquet` | Parquet | `export.py` (datalake) | Attention Heatmap |
| `data/dqn_policy.parquet` | Parquet | `export.py` (datalake) | DQN Alpha Distribution |
| `data/recon_errors.parquet` | Parquet | `export.py` (datalake) | VGAE Recon Errors |
| `data/cka_similarity.parquet` | Parquet | `export.py` (datalake) | CKA Similarity |
| `data/embeddings.parquet` | Parquet | `export.py` (datalake) | UMAP Embeddings (x2) |
| `data/datasets.json` | JSON | `export.py` (catalog) | Dataset Catalog |
| `data/kd_transfer.json` | JSON | `export.py` (filesystem) | Value Boxes (KD gap), Teacher vs Student |
| `data/model_sizes.json` | JSON | `export.py` (model instantiation) | Parameter Counts, Bubble, Pareto |
| `data/graph_samples.json` | JSON | `export.py` (filesystem) | Force Graph |

## Shared OJS Modules

| Module | File | Exports | Used By |
|--------|------|---------|---------|
| Mosaic setup | `_ojs/mosaic-setup.js` | `vg`, `loadParquetTable`, `safeLoadParquetTable`, `listTables`, `describeTable` | All Mosaic charts, data loading |
| Theme | `_ojs/theme.js` | Color palettes (`MODEL_DOMAIN/RANGE`, `LABEL_DOMAIN/RANGE`, etc.), `MARGIN`, `semanticColorRange` | Color scales across all charts |
| Chart helpers | `_ojs/chart-helpers.js` | `filteredChart`, `colorDirectives` | Mosaic filtered charts (run_id dropdowns) |
| Aggregations | `_ojs/aggregations.js` | `paretoFront`, `bestPerGroup`, `kdGap`, `durationMinutes` | Performance, Overview |
| Force graph | `_ojs/force-graph.js` | `renderForceGraph` | Graph Structure page |

## Patterns

### Filtered Chart (Mosaic)
Most Mosaic charts use the `filteredChart()` helper from `chart-helpers.js`:
```js
filteredChart("table_name", sel => vg.plot(
  vg.mark(vg.from("table_name", { filterBy: sel }), { ... }),
  ...directives
))
```

### Color Scales
Import domain/range arrays from `theme.js` instead of inline hex arrays:
```js
import { MODEL_DOMAIN, MODEL_RANGE } from "./_ojs/theme.js"
// Observable Plot: color: { domain: MODEL_DOMAIN, range: MODEL_RANGE }
// Mosaic: ...colorDirectives(MODEL_DOMAIN, MODEL_RANGE)
```
