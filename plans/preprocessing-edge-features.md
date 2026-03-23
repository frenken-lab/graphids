# Preprocessing: Edge Feature Consolidation + Pipeline Fixes

> Created: 2026-03-22

## Context

The `_build_graphs` pipeline in `can_bus.py` has 6 structural inefficiencies, plus `features.py` has reusable helper functions (`edge_features()`, `node_features()`) that are broken/outdated and never called — `can_bus.py` reimplements their logic inline instead. The fix is to make `features.py` the reusable feature library (works for any CAN bus dataset), fix the functions, and have `can_bus.py` call them.

**Current state of `features.py`:**
- `NODE_COL_ORDER` — layout constant ✓ used
- `NODE_STAT_EXPRS` — Polars expressions ✓ used
- `stats_to_tensor()` — assembly function ✓ used by `can_bus.py`
- `node_features()` — per-window function, **dead code** (vectorized path replaced it but kept the shared building blocks above)
- `edge_features()` — per-window function, **dead code** (can_bus.py reimplements inline with a broken 12-slot layout: 3 IAT duplicates, 3 zeros, 1 constant)
- No `EDGE_COL_ORDER`, no `EDGE_STAT_EXPRS`, no `edge_to_tensor()` — edge path has no shared building blocks

**Design principle:** Mirror the node pattern. Nodes have shared building blocks (`NODE_STAT_EXPRS` + `NODE_COL_ORDER` + `stats_to_tensor`) that `can_bus.py` calls. Edges need the same: `EDGE_COL_ORDER` + `EDGE_STAT_EXPRS` + `edge_to_tensor()` fixed to be the single source of truth. Any future CAN bus dataset imports from `features.py`.

## Issues & Fixes

### 1. Bidir join is a separate materialized pass

**File:** `graphids/core/preprocessing/datasets/can_bus.py`

**Problem:** Bidir self-join runs after `collect_all` materializes `edge_df`, causing an extra scan.

**Fix:** Move bidir computation into the `edges_base` lazy frame before `collect_all` so Polars fuses it into the same query plan.

### 2. `_rle_boundaries` computes starts with a Python loop

**File:** `graphids/core/preprocessing/datasets/can_bus.py`

**Problem:** Manual `for c in counts` loop. Polars has `cum_sum()`.

**Fix:** `starts = (rle["len"].cum_sum() - rle["len"]).to_list()`. Delete the function, inline the 3 Polars expressions.

### 3. Edge features assigned column-by-column (7+ `.to_torch()` calls × 600K windows)

**File:** `graphids/core/preprocessing/datasets/can_bus.py`

**Problem:** Per-window loop with 7 separate `.to_torch()` calls = 4.2M FFI crossings.

**Fix:** Build the full edge feature tensor ONCE from `edge_df` using `EDGE_COL_ORDER` + `edge_to_tensor()` before the loop. In the loop, slice with integer indexing (torch view, no copy). Eliminates per-window Polars→torch conversion entirely.

### 4. Edge index uses `np.stack` instead of `.to_numpy().T`

**File:** `graphids/core/preprocessing/datasets/can_bus.py`

**Fix:** `edge_df[es:es+ec].select("src", "dst").to_numpy().T` — one call.

### 5. `features.py` — fix and reuse, don't reimplement

**Files:** `graphids/core/preprocessing/features.py`, `graphids/core/preprocessing/datasets/can_bus.py`

**Problem:** `can_bus.py` imports `edge_features` and `node_features` but calls neither. It reimplements edge assembly inline with a broken layout. Two dead functions in `features.py` that should be the reusable library.

**Fix — make `features.py` the reusable CAN bus feature library:**

1. Add `EDGE_COL_ORDER` constant (like `NODE_COL_ORDER`) — defines the canonical edge feature layout
2. Add `EDGE_STAT_EXPRS` — Polars expressions for vectorized edge computation (like `NODE_STAT_EXPRS`)
3. Fix `edge_features()` — correct feature layout (see issue 6), works per-window for any dataset
4. Fix `node_features()` — verify it matches `NODE_STAT_EXPRS` + `stats_to_tensor` path, keep as the per-window entry point
5. Add `edge_to_tensor()` — assembly function parallel to `stats_to_tensor()`, used by the vectorized path

Then `can_bus.py`:
- Uses `EDGE_STAT_EXPRS` in the lazy `edges_base` query (like it already uses `NODE_STAT_EXPRS` for stats)
- Uses `edge_to_tensor()` for assembly (like it already uses `stats_to_tensor`)
- Removes ALL inline edge feature code
- Any future CAN dataset does the same

### 6. Edge feature layout: 7 of 12 dimensions are dead

**Problem:**
| Slots | Content | Informative? |
|-------|---------|-------------|
| 0, 2, 3 | IAT × 3 | 1 yes, 2 duplicates |
| 1, 4, 5 | Zero | No |
| 6 | 1.0 constant | No |
| 7-10 | byte_0-3 diff | Yes (but only 4 of 8 bytes) |
| 11 | bidir | Yes |

**Fix — new 10-feature layout in `features.py`:**

| Slot | Feature | Source |
|------|---------|--------|
| 0 | IAT | `timestamp.diff()` |
| 1-8 | byte_0-7 diff | `byte_i.diff().abs()` — all 8 payload bytes |
| 9 | bidir | self-join flag |

`N_EDGE_FEATURES: 12 → 10`. Bytes 4-7 already parsed in `_read_raw()` but only 0-3 were used.

**Files to update:**
- `features.py`: `N_EDGE_FEATURES = 10`, add `EDGE_COL_ORDER`, `EDGE_STAT_EXPRS`, `edge_to_tensor()`, fix `edge_features()`
- `can_bus.py`: Remove inline edge assembly, use `features.py` building blocks
- `constants.py`: `EDGE_FEATURE_COUNT = 10`, bump `PREPROCESSING_VERSION` to `"5.0.0"`
- `config/__init__.py`: `VGAEConfig.edge_dim` and `GATConfig.edge_dim` defaults → 10
- `tests/conftest.py`: `EDGE_DIM = 10`
- `tests/test_preprocessing.py`: Update edge shape assertions

## Execution Plan

**Prerequisite:** Commit current working state so code is consistent.

**Task 1 — `features.py` only** (no other files):
Add `EDGE_COL_ORDER`, `EDGE_STAT_EXPRS`, `edge_to_tensor()`. Fix `edge_features()` and `node_features()` to use the correct 10-feature layout. Update `N_EDGE_FEATURES`.
Verify: import check on login node.

**Task 2 — `can_bus.py` only** (depends on task 1):
Rewrite `_build_graphs` edge path to use `features.py` building blocks. Fix issues 1-4 in the process. Remove all inline edge feature code.
Verify: `pytest tests/test_preprocessing.py --collect-only`.

**Task 3 — Constants, config, tests** (depends on task 1):
Update `constants.py`, `config/__init__.py`, `tests/conftest.py`, `tests/test_preprocessing.py`.
Verify: `pytest --collect-only` on all tests.

**Task 4 — Full verification:**
Submit SLURM test job. Resubmit cache rebuild jobs.

## Polars API Notes (verified this session)

- `Series.to_torch()` — no `dtype` arg
- `DataFrame.to_torch(dtype=pl.Float32)` — has `dtype` arg
- `Series.rle()` struct fields: `"value"` and `"len"` (not "values"/"lengths")
- `pl.collect_all([lf1, lf2, lf3])` — parallel execution, shares common scan
- `DataFrame[start:end]` — slice is a view, not a copy
