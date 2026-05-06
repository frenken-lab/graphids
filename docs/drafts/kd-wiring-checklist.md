# KD Pipeline Wiring Checklist

Pre-submission logic check for the knowledge distillation chain:
VGAE → GAT → Extract → Fusion. Each item is a discrete thing to
verify in code before submitting fusion jobs. Status column: ☐ open,
✓ verified, ✗ bug found.

Last updated: 2026-05-06 (all sections 1–4 + 6 complete; 2 bugs found and fixed)

---

## 1. Extract Phase

The `ExtractRow` action calls `extract_states(checkpoints, ...)` which
loads each model ckpt, calls `model.extract_features(batch, device)`,
collects into a TensorDict, and writes to
`{output_dir}/fusion_states/{train,val}_states.pt`.

| # | Check | Status | Notes |
|---|-------|--------|-------|
| 1.1 | **Ckpt paths in fusion plan resolve to real files** — `best_ckpt(dataset, "unsupervised", "vgae", seed)` and `best_ckpt(dataset, "gat_loss", "focal", seed)` point to ckpts that will exist when the extract job runs. Verify the GAT variant used in fusion plan matches the ablation variant that actually trained. | ☐ | GAT is hardcoded to `gat_loss/focal` in `plan/plans/fusion.py` — confirm that's the right choice |
| 1.2 | **Both VGAE and GAT have `extract_features()`** — read both implementations and confirm the method exists and returns a dict with the expected keys. | ✓ | VGAE `vgae.py:482`: returns `errors[N,3], conf[N,1], z_stats[N,4], spike[N,1], affinity[N,1], rq[N,1]` = 11 dims. GAT `gat.py:296`: returns `probs[N,2], conf[N,1], emb_stats[N,4]` = 7 dims. Total = 18 = `_state_dim`. |
| 1.3 | **`extract_features()` called with correct signature** — `extract.py` calls `model.extract_features(batch, device)`. Check both VGAE and GAT accept that exact call (positional device arg). | ✓ | `extract.py:63`: `m.extract_features(batch, device)`. Both models declare `def extract_features(self, batch, device: torch.device)`. Exact match. |
| 1.4 | **`extract_states()` handles the VGAE 6-tuple forward** — VGAE `forward()` returns `(cont_out, canid, nbr, z, kl, edge)`, not logits. `extract_features()` must call `forward()` internally (not be called as `model(batch)`). Verify no mismatch in how extract.py invokes the model. | ✓ | `vgae.py:494`: `extract_features` calls `self._score(batch)` internally — never exposes the 6-tuple. `extract.py:63` calls `m.extract_features(batch, device)`, never `m(batch)`. No mismatch. |
| 1.5 | **Label collection matches graph-level labels** — VGAE trains on benign-only (label_filter='benign'); `extract_states` must use the FULL dataset (both classes) to build fusion states, not the benign-filtered view. Verify `extract.py` constructs the DataLoader from the unfiltered dataset. | ✓ | `extract.py:117-126` uses `GraphDataModule(dataset=source)` with no label_filter — full dataset |
| 1.6 | **TensorDict written with correct CACHE_VERSION** — `extract_states()` stamps a version on the output. `FusionDataModule._load_td()` checks it. Confirm the version constant matches across both files. | ✓ | `fusion.py` imports `CACHE_VERSION` directly from `extract.py` — single source of truth |
| 1.7 | **Idempotency guard skips correctly** — if `{output_dir}/fusion_states/train_states.pt` already exists with the right version, extract skips. If it re-extracts every time, fusion jobs downstream get confused when running in parallel. | ✓ | `extract.py:94-100`: checks both train and val files exist AND version == `CACHE_VERSION`; returns early. Version mismatch or corrupt file falls through to re-extract. |
| 1.8 | **ExtractRow gets MLflow identity tags** — `orchestrate.extract()` dispatches without opening an MLflow run. Confirm this is intentional (pure data transform). If tags are needed for lineage, they're missing. | ☐ | Currently no MLflow run for extract; downstream fusion runs should tag `extractor_ckpts` for lineage. |

---

## 2. Fusion DataModule Wiring

`FusionDataModule` loads pre-extracted TensorDicts and yields `(features, labels)` batches.

| # | Check | Status | Notes |
|---|-------|--------|-------|
| 2.1 | **`cached_states_dir` in plan matches `output_dir` from ExtractRow** — `fusion.py` derives both from `states_dir(dataset, seed)`. Read `paths.py::states_dir()` and confirm the same function is called in both the extract and fusion rows. | ✓ | `paths.py:111`: `states_dir()` returns `{RUN_ROOT}/{dataset}/cached_states/seed_{seed}`. `fusion.py` calls it for `extract_dir`; `primitives.py:134` calls `_states_dir(dataset, seed)` for `cached_states_dir`. Same import, same path. |
| 2.2 | **`{cached_states_dir}/fusion_states/train_states.pt` and `val_states.pt` exist before fusion fit starts** — fusion job must depend (afterok) on extract job. Confirm the plan submission chains them. | ✓ | Fixed via item 6.3 — `submit.py` now tracks `extract_jid` and gates all subsequent fit rows on it. |
| 2.3 | **`state_dim` in fusion model `__init__` matches actual flattened feature dimension** — VGAE: 11 dims, GAT: 7 dims → total 18. Or whatever the real sum is. This is a hardcoded constant in the plan spec that can silently mismatch if `extract_features()` changes. | ✗→✓ | **BUG FIXED**: all 4 model defaults were `state_dim=15`; fusion.py plan now passes `state_dim=18` explicitly. Models updated: mlp.py, bandit.py, dqn.py, weighted_avg.py. |
| 2.4 | **`FusionDataModule` handles both train and val paths** — verify it constructs separate DataLoaders for train and val and that Lightning's `val_dataloader()` hook returns the right one. | ✓ | `fusion.py:97-101`: `train_dataloader()` → `_batches(self.train_td, shuffle=True)`, `val_dataloader()` → `_batches(self.val_td, shuffle=False)`. Distinct. |
| 2.5 | **Label dtype** — fusion models using BCE need `labels.float()`, those using CE need `labels.long()`. Check each fusion model's loss function and compare to what `FusionDataModule` yields. | ✓ | `FusionModuleBase._supervised_loss()`: calls `labels.float()` explicitly before BCE. RL path passes labels to reward calc for comparison — dtype-agnostic. |

---

## 3. Fusion Model Contracts

| # | Check | Status | Notes |
|---|-------|--------|-------|
| 3.1 | **`flatten_features(td)` sort order is stable across extract and train** — keys are sorted lexicographically. If the set of keys ever differs between the two phases, input dim changes silently. | ✓ | `base.py:52`: `sorted(td.keys(include_nested=True, leaves_only=True))`. TensorDict keys are deterministic; set is fixed by extract_features contract. Order: `gat/conf, gat/emb_stats, gat/probs, vgae/affinity, vgae/conf, vgae/errors, vgae/rq, vgae/spike, vgae/z_stats`. |
| 3.2 | **`_derive_scores(td)` in Bandit/DQN extracts anomaly and GAT score from correct keys** — should read `td["vgae"]["conf"]` or similar for anomaly, `td["gat"]["probs"]` for GAT. Read the actual implementation; any hardcoded index into a sorted flat vector is fragile. | ✓ | `reward.py`: `derive_scores` reads `td["vgae","errors"] * self._vgae_weights` (length-3 dot ✓) and `td["gat","probs"][:,1]` (positive-class prob ✓). Named keys, no flat-vector indexing. |
| 3.3 | **`REWARD` constant `vgae_weights` length matches VGAE feature count** — `REWARD["vgae_weights"] = [0.4, 0.3, 0.3]` is 3 weights. If VGAE `extract_features()` returns more or fewer sub-features, the dot product silently underweights or crashes. | ✓ | `REWARD["vgae_weights"] = [0.4, 0.3, 0.3]` (length 3) applied to `errors[N,3]`. Exact match. |
| 3.4 | **DQN: torchrl dependency installed in venv** — DQN uses torchrl. Confirm it's in `pyproject.toml` deps and present in `.venv`. | ✓ | `pyproject.toml:18`: `"torchrl>=0.6"` present. |
| 3.5 | **Bandit: Sherman-Morrison update numerics** — ridge parameter protects against singular matrix. Check there's a positive `lambda_` and that the update clamps or regularizes under near-zero denominators. | ✓ | `bandit.py:141`: denominator is `1.0 + zi @ Az`. `A_inv` starts as `eye(d)/lambda_reg` (PD, `lambda_reg=1.0`); SM rank-1 update preserves PD → `zi @ Az ≥ 0` → denominator ≥ 1.0 always. No singularity. |
| 3.6 | **WeightedAvg: softmax weights initialized to uniform** — if initialized near-zero with a hard-sigmoid, training might stall. Check init. | ✓ | `weighted_avg.py:24`: `self.weight = nn.Parameter(torch.zeros(1))` → `sigmoid(0) = 0.5` → equal weighting at init. Gradient flows through sigmoid from BCE loss. |
| 3.7 | **`automatic_optimization = False` for Bandit/DQN** — Lightning requires `manual_backward()` or no backward at all when this is False. Confirm neither Bandit nor DQN accidentally calls `loss.backward()` via the default Lightning training_step. | ✓ | `FusionModuleBase.automatic_optimization = False` (base class, line 62); `RLFusionBase` re-declares (line 168). `_learn_step()` uses `self.manual_backward(loss)` + `opt.step()`. `training_step` returns `None` for RL path — no loss tensor returned to Lightning. WeightedAvg overrides to `True` and returns loss normally. |

---

## 4. Budget Probe / Test Phase

The probe issue from GAT (inference_mode + `.backward()`) may recur in fusion.

| # | Check | Status | Notes |
|---|-------|--------|-------|
| 4.1 | **FusionDataModule does NOT call `compute_budget()`** — fusion data is pre-extracted flat tensors, not graphs. There's no node-count budget. Verify `FusionDataModule` has no `_budget_result()` or `_pack()` path, and that `dynamic_batching` is not set (or is False) for fusion. | ✓ | `FusionDataModule` has no `dynamic_batching`, no `_budget_result()`, no `_pack()`. All three dataloader methods return `_batches()` generators. |
| 4.2 | **Fusion `test_dataloader()` doesn't trigger budget probe** — verify the fusion test path goes through `FusionDataModule.test_dataloader()` which should be a plain DataLoader over the val TensorDict, no probe. | ✓ | `fusion.py:103-104`: `test_dataloader()` delegates to `self.val_dataloader()` → `_batches(self.val_td, shuffle=False)`. Pure generator, no probe. |
| 4.3 | **Fusion test Trainer inherits `inference_mode=False` fix** — `_trainer_kwargs` now sets `inference_mode=(phase != "test")`. Confirm this path is hit for fusion test rows too (fusion uses the same orchestrate.evaluate() path). | ✓ | `orchestrate.py:165`: `inference_mode=(phase != "test")`; `orchestrate.py:201`: fusion test hits `evaluate()` → `_trainer_kwargs(... "test")` → `inference_mode=False`. |

---

## 5. KD Loss Modules (SoftLabelDistillation / FeatureDistillation)

These are implemented but not yet wired into any current plan. Before any KD plan is authored:

| # | Check | Status | Notes |
|---|-------|--------|-------|
| 5.1 | **Are KD losses in scope for current ablation?** — If no KD plan exists yet, these checks are deferred. Confirm whether a KD plan (large→small VGAE, or large→small GAT) is on the roadmap. | ☐ | If not yet planned, mark 5.x deferred |
| 5.2 | **`SoftLabelDistillation` teacher forward signature** — teacher `forward(batch)` must return `[N, num_classes]` logits. Confirm GAT (when used as teacher) returns that and not a tuple. | ☐ | |
| 5.3 | **`FeatureDistillation` teacher 6-tuple unpack** — teacher VGAE returns `(cont_out, canid, nbr, z, kl, edge)` and distillation unpacks `(t_cont, _, _, t_z, _, _)`. If VGAE forward signature changes (e.g., returns 7 elements), unpack silently reads wrong positions. | ☐ | |
| 5.4 | **Projection layer dimension** — if student latent_dim ≠ teacher latent_dim, `FeatureDistillation` creates an `nn.Linear` to project. Confirm the projection dim is correct and that it's initialized (not None) when dims differ. | ☐ | |
| 5.5 | **Teacher CPU↔device ping-pong overhead** — teacher is moved to GPU each forward, then back to CPU. For large teachers this is expensive. Verify there's a config or fallback to keep teacher on GPU when VRAM allows. | ☐ | Performance, not correctness |
| 5.6 | **`SoftLabelDistillation.last_hard_loss` / `last_soft_loss` logged** — confirm these are emitted via `self.log()` or explicitly logged in `training_step`, otherwise KD component split is invisible in MLflow. | ☐ | |

---

## 6. Plan Wiring (fusion.py)

| # | Check | Status | Notes |
|---|-------|--------|-------|
| 6.1 | **`fusion.py` plan exists and builds without error** — run `gx run ablations.fusion -d hcrl_sa -s 42 --dry-run` (or render to /dev/null) on login node to validate schema. | ✓ | `gx run ablations.fusion -d hcrl_sa -s 42 -o /dev/null` → "wrote 9 rows". Pydantic validated. |
| 6.2 | **`vgae_ckpt` and `gat_ckpt` paths in fusion plan are non-empty** — if either upstream hasn't trained yet, `best_ckpt()` returns a path string regardless of existence. The extract job will then fail at checkpoint load. | ☐ | Render-time path strings are dumb; existence is only checked at exec time |
| 6.3 | **Extract → fusion fit dependency chain in plan submission** — confirm `gx plans submit` chains `extract_fusion` → `{method}_fit` → `{method}_test` via afterok. If the extract row name ends in something other than `-test`, the auto-chain logic won't fire. | ✗→✓ | **BUG FIXED**: submit.py had no extract→fit chaining. Added `extract_jid` tracker; fit rows submitted after an extract row in the same invocation now get `afterok=<extract_jid>`. |
| 6.4 | **`scope_label` not set for fusion DM** — fusion uses TensorDict, not `GraphDataModule`, so curriculum attributes don't apply. Confirm no leftover `scope_label` or `difficulty` kwarg leaks into `FusionDataModule`. | ✓ | `FusionDataModule.__init__` params: `cached_states_dir`, `method`, `batch_size`, `episode_sample_size` only. No `scope_label`/`difficulty`. |

---

## How to use this checklist

Work through items in order 1 → 6. Mark each ✓ (code confirms correct) or ✗ (bug found).
For ✗, open a sub-bullet with the fix. Re-render + submit fusion plan only after all
items in sections 1–4 are ✓.

Section 5 (KD losses) is deferred until a KD plan is authored.
