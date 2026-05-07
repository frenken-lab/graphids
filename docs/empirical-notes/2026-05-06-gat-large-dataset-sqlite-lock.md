# 2026-05-06 — GAT large-dataset ablation: SQLite lock + set_04 CUDA transient

**Prior log:** `docs/empirical-notes/2026-05-06-gat-ablation-first-run-bugs.md`

## Trigger

First submission of `ablations.supervised` on large datasets (set_01, set_02, set_04,
seed 42). 7 of the 12 fit rows failed; hcrl_sa rows from the prior session were unaffected.

---

## Bug 1 — SQLite `database is locked` kills training mid-run (FIXED)

**Root cause:** `DeviceStatsMonitor(cpu_stats=False)` writes ~122 MLflow metrics per
training step via synchronous `client.log_batch()`. With 10 concurrent GPU jobs sharing
a SQLite-backed MLflow DB, lock contention is fatal.

**Fix (two-part):**

- `graphids/plan/compose.py::callbacks_base()`: removed `device_stats` entry entirely.
- `graphids/orchestrate.py::_trainer_kwargs()`: replaced with
  `mlflow.enable_system_metrics_logging()` before `MLFlowLogger()` construction.
  This uses MLflow's built-in async sampler (separate thread, batched writes, no
  per-step lock contention).

**Note:** plan JSON is frozen at submit time. Re-rendering all three supervised plans
was required to propagate the fix to new submissions.

---

## Bug 2 — set_04/focal: CUDA device-side assert on p0240 (TRANSIENT)

**Error:** `torch.AcceleratorError: CUDA error: device-side assert triggered` at
`conv_forward` → `checkpoint.get_device_states(*args)` (async error surfacing at
first CUDA sync after GATv2 scatter).

**Conclusion:** transient hardware issue on p0240. Identical code runs clean on the
same node in a separate variant. Re-submitted fresh to p0227 (new allocation).
