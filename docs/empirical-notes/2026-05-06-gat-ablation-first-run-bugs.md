# 2026-05-06 — GAT ablation first-run bugs

First attempt to submit `ablations.supervised` GAT rows (VGAE-independent subset)
on hcrl_sa seed 42. Four distinct failures hit; two fixed, two open.

## All submissions attempted

Plan IDs used:
- `019dff08` — first render (before primitives.py fix)
- `019dff28` — re-render after `LinearRampSchedule` path fix

| Row | Action | Job IDs attempted | Outcome |
|---|---|---|---|
| focal | fit | 47351900, 47351955 | COMPLETED (both — duplicate run) |
| focal-test | test | 47351901, 47351956, 47352210, 47352991 | FAILED ×3, then COMPLETED |
| ce | fit | 47351905, 47351957 | COMPLETED (both — duplicate run) |
| ce-test | test | 47351906, 47351958, 47352211, 47352995 | FAILED ×3, then COMPLETED |
| weighted_ce | fit | 47351960 | COMPLETED |
| weighted_ce-test | test | 47351961, 47352215, 47352996 | FAILED ×2, then COMPLETED |
| none | fit | 47351966 | COMPLETED |
| none-test | test | 47351968, 47352217, 47352997 | FAILED ×2, then COMPLETED |
| id_hash | fit | 47351975 | COMPLETED |
| id_hash-test | test | 47351976, 47352224, 47353001 | FAILED ×2, then COMPLETED |
| id_lookup | fit | 47351973, 47353178 | FAILED ×1 (MLflow race), then COMPLETED |
| id_lookup-test | test | 47352999, 47353182 | FAILED ×1 (no ckpt), then CANCELLED (bad dep) |
| curriculum_random | fit | 47351969, 47353172 | FAILED ×2 (two different bugs) |
| curriculum_random-test | test | 47352221, 47352998, 47353177 | FAILED ×2 (no ckpt), then CANCELLED |

---

## Verified completion state (MLflow + checkpoint)

Verified via `gx plans show 019dff08` + `gx plans show 019dff28` + filesystem ckpt check.

| Row | Fit ckpt | Test MLflow | Verified |
|---|---|---|---|
| focal | ✓ `/gat_loss/focal/seed_42/checkpoints/best_model.ckpt` | FINISHED (21:14) | ✓ |
| ce | ✓ `/gat_loss/ce/seed_42/checkpoints/best_model.ckpt` | FINISHED (21:14) | ✓ |
| weighted_ce | ✓ `/gat_loss/weighted_ce/seed_42/checkpoints/best_model.ckpt` | FINISHED (21:14) | ✓ |
| none | ✓ `/gat_sampling/none/seed_42/checkpoints/best_model.ckpt` | FINISHED (21:14) | ✓ |
| id_hash | ✓ `/id_encoding/hash/seed_42/checkpoints/best_model.ckpt` | FINISHED (21:15) | ✓ |
| id_lookup | ✓ `/id_encoding/lookup/seed_42/checkpoints/best_model.ckpt` | fit FINISHED (21:19) | fit ✓, test NOT RUN |
| curriculum_random | ✗ no ckpt | not logged | ✗ both phases missing |

**5 rows fully done (fit + test). 1 row fit-only (id_lookup). 1 row nothing (curriculum_random).**

---

## Bug 1 — All test jobs: budget probe fails under `torch.inference_mode` (FIXED)

**Jobs:** 47351901, 47351906, 47351956, 47351958, 47351961, 47351968, 47351976,
47352210–47352224, 47352998, 47352999

**Error:**
```
RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
  budget.py:288  _loss(step_fn(warm_batch)).backward()
```

**Root cause:** Lightning's `EvaluationLoop.run()` is decorated `@_no_grad_context`.
When `inference_mode=True` (default), this wraps the entire `run()` — including
`setup_data()` → `test_dataloader()` → `compute_budget()` → `probe()` — in
`torch.inference_mode()`. Unlike `torch.no_grad()`, `torch.inference_mode` cannot
be escaped by `torch.enable_grad()` inside it.

**Fix (two-part):**
- `graphids/core/budget.py` lines 282, 355: `with torch.enable_grad(), _silent_log(model):`.
  Handles the `no_grad` layer once inference_mode is off.
- `graphids/orchestrate.py` `_trainer_kwargs`: `inference_mode=(phase != "test")`.
  Switches the test Trainer from `torch.inference_mode` to `torch.no_grad`,
  which `torch.enable_grad()` can override.

First attempt used only `torch.enable_grad()` — did not work (inference_mode is
non-overrideable). Second attempt added `inference_mode=False` — fixed it.

---

## Bug 2 — curriculum_random fit: wrong module path for `LinearRampSchedule` (FIXED)

**Job:** 47351969

**Error:**
```
AttributeError: module 'graphids.core.data.preprocessing.curriculum'
  has no attribute 'LinearRampSchedule'
```

**Root cause:** `primitives.py` had:
```python
LINEAR_RAMP = "graphids.core.data.preprocessing.curriculum.LinearRampSchedule"
```
`LinearRampSchedule` is in `graphids.core.curriculum`. The preprocessing module
only has `score_random` / `score_vgae`.

**Fix:** `primitives.py` corrected to `"graphids.core.curriculum.LinearRampSchedule"`.
Plan re-rendered → plan_id `019dff28`.

**Status: Path fixed, but uncovered Bug 3 on re-submit.**

---

## Bug 3 — curriculum_random fit: `score_random` is a function, not a class (OPEN)

**Job:** 47353172 (first run with fixed LinearRampSchedule path)

**Error:**
```
TypeError: score_random() missing 1 required positional argument: 'graphs'
```

**Root cause:** `supervised.py` has:
```python
"difficulty": spec(SCORE_RANDOM, seed=seed)
```
`spec()` produces `{class_path: ..., init_args: {seed: seed}}`.
`_instantiate` calls `score_random(seed=seed)`.
But `score_random` is `def score_random(graphs, seed=0) -> Tensor` — a plain
function that needs the full dataset list as its first argument. It cannot be
constructed with just kwargs and called later.

**Fix needed:** Wrap `score_random` / `score_vgae` in callable classes:
```python
class ScoreRandom:
    def __init__(self, seed=0): self.seed = seed
    def __call__(self, graphs): return score_random(graphs, seed=self.seed)
```
Then `spec(SCORE_RANDOM, seed=seed)` + `_instantiate` works as-is.
Update `SCORE_RANDOM` / `SCORE_VGAE` constants in `primitives.py` to point at the classes.

**Status: OPEN. `curriculum_random` and `curriculum_vgae` both blocked until fixed.**

---

## Bug 4 — id_lookup fit: MLflow experiment creation race (TRANSIENT)

**Job:** 47351973

**Error:**
```
mlflow.exceptions.MlflowException: Experiment(name=graphids/hcrl_sa/id_encoding)
  already exists. UNIQUE constraint failed
```

**Root cause:** `id_lookup` and `id_hash` submitted simultaneously. Both raced to
`MLFlowLogger.__init__` → `create_experiment()` before either had written to the
SQLite DB. One won, one hit the UNIQUE constraint.

**Fix:** Resubmit. The experiment exists on retry; `MLFlowLogger` finds it.
Job 47353178 COMPLETED.

**Status: RESOLVED for fit. id_lookup-test not yet run (see below).**

---

## Outstanding work

- **id_lookup-test**: fit ckpt exists, test never ran successfully. Job 47353182
  was submitted with a broken dependency (used `squeue | tail -1` to get dep ID,
  which chained to the wrong job), got silently cancelled — does not appear in
  sacct. Needs a clean `gx submit --plan ... --row-name id_lookup-test` with no
  dependency (fit is done).
- **curriculum_random**: blocked on Bug 3. Fix scorer classes, re-render, resubmit
  both fit and test.
- **curriculum_vgae**: waiting on VGAE ckpt AND blocked on Bug 3 (same scorer issue).
