# Resource Profiling — Remaining Phases

> P1 (ResourceProfileCallback + edge margin) and P2 (backward probe, KD teacher,
> fusion pre-flight, compile flag) shipped in session 14. See git history.

## Phase 3: Post-Campaign Analysis (blocked on campaign data)

### 3A. Budget Calibration Analyzer Task

New task in `graphids/core/artifacts/tasks.py` following `run_embeddings()` pattern.
Reads `resource_profile.csv` from a run dir, computes:

- Edge→VRAM Pearson r
- Fragmentation slope (reserved−allocated over steps)
- Peak VRAM p50/p95/p99
- Host RSS max + growth slope
- Empirical backward multiplier (if probe result available)

Writes `{output_dir}/vram_calibration.json`.

Wire into `analyzer.py` via `vram_calibration: bool = False` flag.

### 3B. Cross-Run Aggregation

New `--calibrate` flag on `probe-budget` command. Walks run dirs, reads
`artifacts/vram_calibration.json`, aggregates per (model_type, scale, dataset):

- Recommended `_SAFETY_MARGIN`
- Recommended `backward_multiplier` (median across runs)
- Worker RSS risk flag

Writes `{lake_root}/reference/budget_calibration.csv`.

## Phase 4: Feedback Loop (after P3)

- **Auto-read calibration in `node_budget()`** — read calibrated safety margin
  and backward multiplier from `budget_calibration.csv` if it exists.
- **Per-model worker recommendations** — from callback data, identify optimal
  `num_workers` based on cg_ratio and worker RSS growth rate.
