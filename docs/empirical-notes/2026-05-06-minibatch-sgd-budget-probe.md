# 2026-05-06 — Mini-batch SGD: budget probe was producing full-batch GD on small datasets

**Prior log:** `docs/empirical-notes/2026-05-06-drop-neighborhood-adopt-tam.md`
**Fix:** commit `952d0ad`.

## Problem

VGAE budget probe (`graphids/core/budget.py::probe`) maximises pack size to fill VRAM. On
hcrl_sa the largest benign graph is 34 nodes (~1 MB activations on V100):

```
B1 ≈ headroom / per_node ≈ 12 500 MB / (1 MB / 34) ≈ 430 000 nodes/pack
```

Total benign train nodes ≈ 163 500, so `pack_offline` produced **1–2 packs/epoch** —
full-batch GD, not SGD. Adam accumulated only ~1 200 gradient steps over 600 epochs
(<< the ~1 000 needed for moments to converge); the optimizer was barely warmed up.
Invisible in logs because `pack_offline` exhaustion capped `budget_nodes` to
`total_train_nodes`, so the probe looked "working" — working = 1 pack.

## Fix — `min_steps_per_epoch` knob

Added `min_steps: int | None = None` to `probe()` and `node_budget()`. After VRAM-limited
`B1`, cap to ensure ≥ N steps/epoch:

```python
if min_steps is not None and min_steps > 1:
    step_cap = sum(sizes_list) // min_steps
    if step_cap > B0_nodes:          # never cap below largest single graph
        B1 = min(B1, step_cap)
```

Threaded through `GraphDataModule.__init__` (`min_steps_per_epoch: int = 1`, default
preserves old behaviour) and `BaseModel.compute_budget`. Plan authors pass via the
`**overrides` path in `graph_dm(...)`. `ablations/unsupervised.py` uses
`min_steps_per_epoch=50` for `vgae_data`; `dgi` keeps the plain `data`. Default `=1` ≡
old VRAM-only behaviour — no change for set_02+ where VRAM binds first.

## Validation — hcrl_sa smoke

Fit 47346519 / test 47346524 (plan `019dfe8e-...`), both COMPLETED. Probe log:
`budget_nodes=3271 == 163 550 // 50` exactly — **min_steps cap binding**, not VRAM
(would have been ~430 000). Got ~50 packs/epoch. `val_discrimination_ratio` 4.41 → 7.21
over 166 epochs (early-stop patience=100). Test AUROC: test_01 0.759, test_02 0.739
(fuzzing 0.398 — known VGAE limitation, not a regression), test_03 0.9996, test_04 0.953.
`peak_vram_mb=81` on 16 GB V100 — expected; hcrl_sa is smoke-only.
