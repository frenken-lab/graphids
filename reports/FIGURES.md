# KD-GAT Figure Registry

All figures in the Quarto reports site. Each entry documents the rendering method, data source, and location.

## Rendering Methods

| Method | Description |
|--------|-------------|
| **YAML spec** | Declarative Mosaic spec in `figures/`, rendered by `renderSpec()` |
| **renderTable** | SQL query rendered as Observable `Inputs.table()` via `table-renderer.js` |
| **OJS stub** | Minimal inline OJS (reactive logic that can't be expressed declaratively) |
| **Quarto native** | Quarto value box component |

## Dashboard Figures

| # | Chart | Spec File | Method | Data Source | Tab |
|---|-------|-----------|--------|-------------|-----|
| 1 | Total Runs | - | Quarto native | `summary.json` | Overview |
| 2 | Best F1 Score | - | Quarto native | `summary.json` | Overview |
| 3 | Datasets | - | Quarto native | `summary.json` | Overview |
| 4 | Avg KD Gap | - | Quarto native | `summary.json` | Overview |
| 5 | Leaderboard | - | OJS stub | `metrics` (UNPIVOT + renderTable) | Overview |
| 6 | Dataset Comparison | `fig-dataset-comparison.yaml` | YAML spec | `metrics` (UNPIVOT) | Overview |
| 7 | F1 by Dataset | `fig-f1-by-dataset.yaml` | YAML spec | `metrics` | Overview |
| 8 | Model Params | `fig-model-params.yaml` | YAML spec | `model_sizes` | Overview |
| 9 | Bubble F1 vs Accuracy | `fig-bubble-f1-accuracy.yaml` | YAML spec | `metrics` JOIN `model_sizes` | Performance |
| 10 | Pareto Frontier | `fig-pareto-frontier.yaml` | YAML spec | `metrics` JOIN `model_sizes` (SQL window fn) | Performance |
| 11 | F1 vs AUC | `fig-f1-vs-auc.yaml` | YAML spec | `metrics` | Performance |
| 12 | Runs by Dataset | `fig-runs-stacked.yaml` | YAML spec | `runs` (SQL GROUP BY) | Performance |
| 13 | Training Duration | `fig-training-duration.yaml` | YAML spec | `runs` (SQL CASE) | Training |
| 14 | Training Comparison | `fig-training-comparison.yaml` | YAML spec | `training_curves` (dual menu) | Training |
| 15 | Training Carpet | `fig-training-carpet.yaml` | YAML spec | `training_curves` | Training |
| 16 | Attention Heatmap | `fig-attention-heatmap.yaml` | YAML spec | `attention_weights` | GAT & DQN |
| 17 | DQN Policy | `fig-dqn-policy.yaml` | YAML spec | `dqn_policy` | GAT & DQN |
| 18 | Training Curves | `fig-training-curves-single.yaml` | YAML spec | `training_curves` | GAT & DQN |
| 19 | Recon Errors | `fig-recon-errors.yaml` | YAML spec | `recon_errors` | GAT & DQN |
| 20 | KD Transfer | `fig-kd-transfer.yaml` | YAML spec | `kd_transfer` | KD |
| 21 | CKA Heatmap | `fig-cka-heatmap.yaml` | YAML spec | `cka_similarity` | KD |
| 22 | UMAP VGAE | `fig-umap-vgae.yaml` | YAML spec | `embeddings` | KD |
| 23 | UMAP GAT | `fig-umap-gat.yaml` | YAML spec | `embeddings` | KD |
| 24 | Loss Landscape 2D | `fig-loss-landscape-2d.yaml` | YAML spec | `loss_landscape` | Loss Landscape |
| 25 | Loss Profile 1D | `fig-loss-profile-1d.yaml` | YAML spec | `loss_landscape` (SQL agg) | Loss Landscape |
| 26 | Loss Summary | - | renderTable | `loss_landscape` (SQL agg) | Loss Landscape |
| 27 | Graph Network | `fig-graph-network.yaml` | YAML spec | `graph_nodes` + `graph_edges` | Graph Structure |
| 28 | Degree Distribution | `fig-degree-distribution.yaml` | YAML spec | `graph_stats` | Graph Analysis |
| 29 | Feature Heatmap | `fig-feature-heatmap.yaml` | YAML spec | `graph_stats` (SQL pivot) | Graph Analysis |
| 30 | Adjacency Pattern | `fig-adjacency-matrix.yaml` | YAML spec | `graph_stats` | Graph Analysis |
| 31 | Graph Density Strip | `fig-graph-stats-boxplot.yaml` | YAML spec | `graph_stats` | Statistics |
| 32 | Clustering Coeff | `fig-clustering-coeff.yaml` | YAML spec | `graph_stats` | Statistics |
| 33 | Summary Stats | - | renderTable | `graph_stats` (SQL agg) | Statistics |
| 34 | Graph Feature Overview | `fig-graph-parallel.yaml` | YAML spec | `graph_stats` | Statistics |
| 35 | Dataset Catalog | - | OJS stub | `datasets.json` | Datasets |
| 36 | Run Timeline | `fig-run-timeline.yaml` | YAML spec | `runs` (SQL filter) | Staging |
| 37 | Model Predictions | - | OJS stub | `metrics` (filtered by run) | Staging |
| 38 | Confusion Proxy | `fig-confusion-proxy.yaml` | YAML spec | `metrics` (SQL filter) | Staging |

## Summary

| Method | Count |
|--------|-------|
| YAML spec | 29 |
| renderTable | 3 |
| OJS stub | 6 (4 value boxes + leaderboard + dataset catalog + model predictions) |
| **Total** | **38** |

## File Layout

```
reports/
  figures/             # 29 YAML specs (dashboard + paper)
    fig-*.yaml
  _ojs/
    mosaic-renderer.js  # renderSpec() — YAML/JSON → Mosaic DOM
    mosaic-setup.js     # DuckDB-WASM + coordinator init
    table-renderer.js   # renderTable() — SQL → Inputs.table
    query-utils.js      # safeEq() for SQL injection prevention
  figures/
    _theme.yaml         # Shared color palettes (resolved by renderSpec)
  data/                 # Parquet + JSON from export pipeline
  dashboard.qmd         # ~490 lines (was 1354)
```
