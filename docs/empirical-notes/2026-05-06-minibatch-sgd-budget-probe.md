# 2026-05-06 — Mini-batch SGD: budget probe was producing full-batch GD on small datasets

**Prior log:** `docs/empirical-notes/2026-05-06-drop-neighborhood-adopt-tam.md`

## Problem

The VGAE budget probe (`graphids/core/budget.py::probe`) maximises pack size to fill
available VRAM. On hcrl_sa — where the largest single benign graph is **34 nodes** and
the V100 activation cost is ~1 MB at 34 nodes — the VRAM-limited budget was:

```
headroom ≈ 12 500 MB
per_node ≈ 1 MB / 34 = 0.029 MB/node
B1 ≈ 12 500 / 0.029 ≈ 430 000 nodes/pack
```

Total benign train nodes on hcrl_sa ≈ 163 500. With B1 >> total, `pack_offline`
produces **1–2 packs per epoch** — the DataLoader iterates the entire training split in
one or two batches. That is full-batch gradient descent, not SGD. Adam's moment
estimates only accumulate ~1 200 gradient steps over 600 epochs (1 200 << the ~1 000
steps Adam needs for moments to converge). Discrimination-ratio improvement was real but
slow; the optimizer was barely warmed up by the end of training.

This was invisible in the budget log because `budget_nodes` was always capped to
`total_train_nodes` by `pack_offline`'s exhaustion, so the probe appeared to be
"working" — it just happened that working = 1 pack.

## Fix — `min_steps_per_epoch` knob

Added `min_steps: int | None = None` to `probe()` and `node_budget()` in
`graphids/core/budget.py`. After the VRAM-limited `B1` is computed, cap it:

```python
if min_steps is not None and min_steps > 1:
    total_nodes = sum(sizes_list)
    step_cap = total_nodes // min_steps
    if step_cap > B0_nodes:          # never cap below largest single graph
        B1 = min(B1, step_cap)
```

Threaded through `GraphDataModule.__init__` (`min_steps_per_epoch: int = 1`, default
preserves old behaviour) and `BaseModel.compute_budget`. Plan authors pass it via the
existing `**overrides` path in `graph_dm(...)`:

```python
vgae_data = graph_dm(
    source=can_bus(dataset=dataset, seed=seed),
    label_filter="benign",
    min_steps_per_epoch=50,
)
```

`ablations/unsupervised.py` now uses a separate `vgae_data` with `min_steps_per_epoch=50`
while `dgi` keeps the plain `data` (no cap). Default `min_steps_per_epoch=1` ≡ old VRAM-only
behaviour — no change for `set_02`+ where VRAM binds first.

## Validation — hcrl_sa smoke run

Fit job 47346519 / test job 47346524, plan `019dfe8e-c493-7497-94a4-6518d9424631`.
Both COMPLETED, exit 0:0. Fit 5m 26s, test 21s.

**`budget_probed` log (job 47346519 stdout):**

```json
{
  "worst_V": 34, "worst_E": 98,
  "activation_mb": 1, "cudnn_reserve_mb": 784,
  "target_mb": 13341, "budget_nodes": 3271,
  "budget_edges": 10370, "repacked": true,
  "sanity_V": 3262, "sanity_peak_mb": 74,
  "event": "budget_probed"
}
```

`budget_nodes = 3271` matches `163 550 // 50 = 3271` exactly — the **min_steps cap is
binding**, not the VRAM limit. The VRAM-only budget would have been ~430 000 nodes
(1 pack); with the cap we get ~50 packs/epoch.

**Training metrics (166 val epochs, early stopping patience=100 on
`val_discrimination_ratio`):**

| Metric | First epoch | Final |
|---|---|---|
| `val_discrimination_ratio` | 4.41 | 7.21 |
| `train_loss` | 1.40 | 0.37 |

**Test metrics (hcrl_sa, seed 42):**

| Split | AUROC | Notes |
|---|---|---|
| test_01 known/known | 0.759 | DoS 0.999, fuzzing 0.568 |
| test_02 unknown vehicle/known | 0.739 | fuzzing AUROC 0.398 (near-random) |
| test_03 known/unknown | 0.9996 | malfunction |
| test_04 unknown/unknown | 0.953 | malfunction |

Fuzzing weakness on test_02 is a known VGAE limitation (vocab-OOD but low topological
signal), not a regression from this change.

**VRAM:** `graphids.peak_vram_mb = 81` on a 16 GB V100 — 0.5 % occupancy. Expected:
at 34-node graphs the GPU is computationally trivial regardless of batch size. GPU busy
fraction ≈ (8 300 steps × ~10 ms/step) / 283 s ≈ 29 %. The dataset is too small for
meaningful GPU utilisation; hcrl_sa is smoke-only.

## Pending — set_02 throughput check

Fit job 47348301 / test job 47348303, plan `019dfebc-c50f-711d-8c7e-cdc661e0db02`.
set_02 has larger graphs; VRAM budget will bind before `min_steps` cap. Checking that:
1. `budget_nodes` is VRAM-limited (not step-capped) on set_02.
2. `peak_vram_mb` is in the hundreds–thousands, not 81 MB.
3. GPU utilisation is substantially higher than hcrl_sa.

Results to be appended when jobs complete.

## Rendered plan path convention

Previous ad-hoc render paths (`/tmp/plan.json`, no fixed location) made re-submission
require searching. New rule (`slurm-hpc.md` "Rendered plan paths"): render to

```
rendered/{dataset}/{plan_module_as_path}/seed_{seed}.json
```

mirroring the lake-root run structure `{dataset}/ablations/{group}/{variant}/seed_{N}`.
`rendered/` is gitignored at its own `.gitignore` (data artifact). The directory is
tracked so the convention is visible in git.

```bash
# canonical render command
gx run ablations.unsupervised -d hcrl_sa -s 42 \
    -o rendered/hcrl_sa/ablations/unsupervised/seed_42.json

# re-submit from known path, no search required
gx submit --plan rendered/hcrl_sa/ablations/unsupervised/seed_42.json \
    --row-name vgae --cluster pitzer
```
