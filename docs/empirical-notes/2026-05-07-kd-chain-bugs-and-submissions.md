# KD Chain Bug Fixes + Phase 2/3 Submissions — 2026-05-07

## Context

Phase 2 (student KD) training plus artifact pipeline repair. All runs on
hcrl_sa, set_01–04; seed 42; Pitzer V100s.

---

## Bug 1: student_vgae_kd crash — `AttributeError: '_uses_edge_attr'`

**Root cause.** `VGAE`/`GAT` defer architecture construction to `_build()`,
which fires only when `num_ids > 0`. Teacher models instantiated from the
plan spec receive `num_ids=0`, so `_build()` never runs and layer attributes
are never set.

**Fix.** `distillation._load_teacher` now reads `hyper_parameters` from the
checkpoint (available under `weights_only=True`) and calls `_build()` before
loading state dict. Keys pulled: `num_ids`, `in_channels`, `num_classes`.

**File.** `graphids/core/losses/distillation.py::_load_teacher`

---

## Bug 2: student_vgae_kd crash — tensor size mismatch (64 vs 128)

**Root cause.** `FeatureDistillation` aligns student latent `z` (dim 64,
small VGAE) against teacher latent `z` (dim 128, large VGAE) via direct
MSE. No projection in the plan spec.

**Fix.** Added `projection=spec("torch.nn.Linear", in_features=64,
out_features=128)` to the `FEATURE_DISTILLATION` block in `training/main.py`.

**Outcome.** Jobs 47368234–47368238 all COMPLETED or RUNNING after fix.
All prior batches (47367301–47368085) failed.

---

## Bug 3: 84 analyze jobs jammed GPU queue (~3 h each)

**Root cause.** All analyze rows (including fusion_policy, which reads
pre-extracted states from disk) were tagged `gpu`. Loss landscape used
51×51 = 2601 forward passes per checkpoint.

**Fixes.**
- Fusion rows → `mode="cpu"`, `length="short"` (no GPU needed).
- Landscape resolution 51 → 21 (441 passes; fits `gpu/short` 1 h slot).
- All 84 jobs cancelled; analyze plan re-rendered with corrected tags.

**Files.** `graphids/plan/plans/ops/analyze.py`,
`graphids/plan/compose.py` (added `landscape_resolution` param).

---

## Bug 4: `_vgae_loss` / `_dgi_loss` wrong calling convention

**Root cause.** Both called `model(batch.x, batch.edge_index, ...)` with
unpacked tensors; models expect a single `Data` object. `_vgae_loss` also
unpacked a 3-tuple from a 6-tuple return.

**Fix.** Changed to `model(batch)` and unpacked all 6 return values.

**File.** `graphids/core/artifacts/compute.py`

---

## Bug 5: fusion policy artifact path — 4 compounding bugs

All bugs were in the analyze → fusion_policy path:

| # | Location | Bug | Fix |
|---|---|---|---|
| 1 | `compute.py::compute_fusion_policy` | Called `module.agent` (doesn't exist); passed raw `Tensor` to `predict` | Use `module` directly; pass `TensorDict` |
| 2 | `compute.py::compute_fusion_policy` | Wrong key `"norm_states"` (actual: `"td_norm"`) | Corrected key; use `flatten_features(result["td_norm"])` for q-values |
| 3 | `compute.py::PolicyResult` | `q_values: np.ndarray` — crashes for non-DQN models | Changed to `np.ndarray \| None`; guarded in `save_fusion_policy` |
| 4 | `io.py::load_fusion_eval` | Wrong kwargs (`dataset`, `lake_root`, `vgae_ckpt_path`, etc.) — `FusionDataModule` takes only `cached_states_dir` | Rewrote to use `states_dir(dataset, seed)` |
| 5 | `_dispatch.py::_run_fusion_policy` | Passed all wrong kwargs to `load_fusion_eval`; used `module.agent` | Simplified to 3 lines |

---

## Submissions

| Rows | Plan | Datasets | JIDs |
|---|---|---|---|
| student_vgae_kd (fit+test) | training.main | hcrl_sa, set_01–04 | 47368234–47368238 |
| student_gat_kd + nokd (fit+test) | training.main | hcrl_sa, set_01–04 | 47368581–47368602 |
| student_vgae_nokd (fit+test) | training.main | hcrl_sa, set_01–04 | 47368606–47368617 |

student_vgae_kd hcrl_sa confirmed training (job 47368235 running at session
end). All student_gat and student_vgae_nokd pending.
