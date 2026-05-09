# 2026-05-08 — Artifact pipeline: 4 bugs found and fixed during ops.analyze run

**Context:** First full `ops.analyze` run across hcrl_sa + set_01–04, seed 42.
82 jobs submitted (55 GPU / 27 CPU fusion). All bugs surfaced in this session.

---

## Bug 1 — `KeyError: 'alphas'` in `compute_fusion_policy`

**File:** `graphids/core/artifacts/compute.py::compute_fusion_policy`  
**Affected:** mlp, moe, moe_noaux, weighted_avg (all CPU fusion jobs, first pass)  
**Cause:** `FusionModuleBase.predict()` returns `alphas` only on the RL path
(`automatic_optimization=False` — Bandit/DQN). Non-RL models return `fused_scores`
only. `compute_fusion_policy` unconditionally accessed `result["alphas"]`.  
**Fix:** `result.get("alphas", result["fused_scores"])` — fused scores are the
equivalent per-sample signal for non-RL models.

## Bug 2 — `requires_grad` numpy error (immediate follow-on)

**File:** same function, same line  
**Cause:** `fused_scores` from non-RL models retains the autograd graph (output of
`torch.sigmoid`). `.cpu().numpy()` fails on grad-tracked tensors.  
**Fix:** `.detach().cpu().numpy()`.

## Bug 3 — `gx plans submit --length` defaulted to `"long"`

**File:** `graphids/cli/plans/submit.py`  
**Affected:** set_01 GPU jobs — sent to pitzer `gpu` long (90 min walltime)
instead of cardinal `debug` (1 h), causing multi-hour queue wait.  
**Cause:** `--length` CLI option hardcoded `= "long"`, silently overriding each
row's `resources.length` field. The plan encoded `length="short"` correctly;
the CLI stomped it.  
**Fix:** Default `None`; both dry-run print and `submit_row` call use
`length or r.resources.length`.

## Bug 4 — `safe_load_checkpoint` left model on CPU for GPU jobs

**File:** `graphids/core/models/base.py::safe_load_checkpoint`  
**Affected:** all GPU analyzer jobs (device-mismatch crash in embedding lookup)  
**Cause:** `cls(**init_kwargs)` instantiates on CPU. `atomic_load(map_location=device)`
loads state dict tensors to the target device, but `load_state_dict` copies them
*into* the CPU model parameters via `copy_()`, keeping the model on CPU.
`map_location` never moved the module itself. Data batches moved to CUDA via
`batch.clone().to(device)`; embedding index landed on CUDA while weight stayed CPU.  
**Fix:** `module.to(map_location)` after `load_state_dict`.

---

**Net outcome:** CPU fusion: all 27 COMPLETED after bugs 1+2. GPU set_01 resubmitted
to cardinal debug after bugs 3+4; set_02–04 + hcrl_sa pending set_01 validation.

---

## Bug 5 — `compute_cka` silently truncated to corresponding layer pairs

**File:** `graphids/core/artifacts/compute.py::compute_cka`  
**Affected:** all `cka.json` outputs where `n_teacher_layers != n_student_layers`  
**Cause:** Used `n_layers = min(len(teacher_reps), len(student_reps))` then iterated
`range(n_layers)` for both teacher and student. With a 3-layer teacher and 2-layer
student, this produced only 2 `layer_i` keys instead of the full 6-entry cross-matrix.
Teacher layer 2 was silently dropped — no warning, no error.  
**Fix:** Full nested-loop cross-matrix with `teacher_{i}_student_{j}` keys:
```python
return {
    f"teacher_{i}_student_{j}": _linear_cka(teacher_reps[i], student_reps[j])
    for i in range(len(teacher_reps))
    for j in range(len(student_reps))
}
```
Output shape changes from `N` corresponding-pair scalars to `n_teacher × n_student`
scalars. Downstream consumers (`pull_data.py`, `cka/App.svelte`) updated accordingly.  
**Requires:** Re-run `ops.analyze` for all datasets/variants/seeds to regenerate
`cka.json` with the corrected key format before the paper figure reflects the fix.
