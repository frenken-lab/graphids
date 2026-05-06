# 2026-05-06 — GAT large-dataset ablation: SQLite lock + set_04 CUDA transient

**Prior log:** `docs/empirical-notes/2026-05-06-gat-ablation-first-run-bugs.md`

## Trigger

First submission of `ablations.supervised` on large datasets (set_01, set_02, set_04,
seed 42). 7 of the 12 fit rows failed; hcrl_sa rows from the prior session were unaffected.

---

## Bug 1 — SQLite `database is locked` kills training mid-run (FIXED)

**Affected rows:** set_01/focal (57 ep), set_01/ce (62 ep), set_01/weighted_ce (129 ep),
set_02/ce (33 ep), set_04/ce (1 ep), set_04/weighted_ce (1 ep).

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

**Job:** 47353873 (p0240 V100, 10 epochs, ScatterGatherKernel OOB)

**Error:** `torch.AcceleratorError: CUDA error: device-side assert triggered` at
`conv_forward` → `checkpoint.get_device_states(*args)` (async error surfacing at
first CUDA sync after GATv2 scatter).

**Investigation:**
- Cache metadata: `num_arb_ids=2049`, node_id range [1, 2048] — within embedding bounds.
- Edge index check: 0/122443 graphs with OOB indices in set_04 train cache.
- set_04/lookup ran 300 epochs to completion on the same data, same model, same focal
  loss, same node. Same `vocab_digest` as set_02 (`e990b275...`).

**Conclusion:** transient hardware issue on p0240. Identical code runs clean on the
same node in a separate variant. Re-submitted fresh to p0227 (new allocation).

---

## Re-submission

All 7 failed fit rows re-submitted with fresh renders (plan_ids `019dffad-f4cd...`
through `019dffad-fd4c...`):

| Dataset | Rows | Strategy | JIDs (fit) |
|---------|------|----------|------------|
| set_01 | focal, ce, weighted_ce | resume from `last.ckpt` | 47356087–89 |
| set_02 | ce | resume from `last.ckpt` | 47356099 |
| set_04 | focal, ce, weighted_ce | fresh start | 47356076–80 |

Corresponding test rows submitted with `afterok` chains (JIDs 47356081–83, 47356092–95,
47356100). Set_04/weighted_ce landed on p0240 again — monitor for recurrence.
