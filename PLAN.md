# GraphIDS Session Plan

> PLAN.md is **current-session work only**. Historical changelogs live in
> `git log`; durable verdicts in `docs/decisions/README.md`; living
> architecture in `docs/reference/`; cross-project plans in `~/plans/`.

### #43 closed — VGAE 5-bug-fix verified

All 5 enumerated bugs (sigmoid decoder, scoring weights, scaler
benign-only, mean-pool, masking) are confirmed fixed in main code.
Verified by post-fix `set_01` seed 42 fit (MLflow run
`8e06fc6e903045fbb94376e9ef5266d5`): `val_loss = 0.113` vs ~2800
plateau pre-fix; `test_03_known_vehicle_unknown_attack/auc = 0.76`;
aggregate MCC flipped from −0.05 → +0.12. Closing comment posted
to GitHub. Remaining test_01 known-attack F1=0 and test_06 masquerade
AUC=0.27 are NOT regressions of #43 — different bug surfaces, should
file as separate issues if pursued.

### Ablation surface — 3 new groups, score_weights walked back

- **`scaler/`** — `z_benign` (default), `robust_benign`. Tests scaler
  fitting population. `z_joint` was added then removed: principled-out
  for IDS (OOD-attack is dominant deployment risk; joint fitting bakes
  the training-attack distribution into the input frame). Both code
  path (`scaler.py:STRATEGIES`) and ablation variant deleted.
- **`curriculum_direction/`** — `low_to_high` (1.0→10.0, current code
  default) and `high_to_low` (10.0→1.0, DCL/MID literature direction).
  Random within-tier ordering — isolates direction from scorer choice.
- **`score_weights/`** — added then removed in same session. The
  proposed (α, β, γ) sweep was answering the wrong question; the
  research-backed approach is per-component z-normalization on benign
  val (BWGNN/DCOR/AutoGraphAD/GAD-NR pattern), not fixed weights. Left
  as separate code-change work.

`configs/plans/ofat.jsonnet` extended with the 4 new variants in
Stage 1 (parallel, no upstream deps). Render verified — 39 JSONL rows
(was 31).

### kd-gat-paper methodology subsection added

Subsection `### Feature Standardization on Benign Training Rows` in
`paper/content/methodology.md`. 13 new BibTeX entries across 3 .bib
files. Defends `z_benign` choice with NIDS-context citations
(ADBench, PyOD, Sommer & Paxson), one-class precedent (Donut,
Deep SVDD, USAD, GANomaly), and test-time normalization analog
(AdaBN, TENT, RevIN). Empirical magnitude on `set_01`: median 2.4%
mean shift across 35 features, max 37%. Uncommitted in
`~/kd-gat-paper`; not synced to Curvenote.

### Compute-efficiency analysis (`~/plans/compute-efficiency-recalibration.md`)

Three-layer evidence pass on the 13 post-#43 fits:

1. **System telemetry** — GAT GPU util **median 11%** with mem at 99%;
   VGAE 17% / 98%. Fusion is **0% GPU** across all 4 methods.
   Conclusion: dataloader-bound, not compute-bound. The "GAT is
   compute-bound (cg_ratio≈0.21)" comment in
   `configs/models/supervised.libsonnet:4` is empirically false.

2. **Existing 2026-03-23 plan** (`~/plans/gpu-utilization-and-training-efficiency.md`)
   — most items still unapplied: `compile_model: true` is still
   `false`; `exclude_keys=['attack_type']` not used; `set_to_none=True`
   only in budget probe.

3. **Literature review** (22 primary sources) — published GAT/VGAE
   recipes use **Adam constant-LR, no scheduler, fixed-epoch or
   patience=50 early stopping**. Our cosine over 300-1200 epochs is
   outside published practice. OneCycleLR / Lion-Sophia / linear-LR
   scaling for GNNs have **no published validation** — earlier
   recommendations along those lines retracted.

Recommendations summary in the doc, with HIGH-confidence path:
num_workers bump (4→8 GAT, 2→6 VGAE), fusion → CPU partition,
constant-LR Adam replacing cosine, epoch budgets aligned with
Veličković 2018 / Kipf 2016 (200 VGAE, 150 GAT, 100 fusion),
`compile_model: true`. Estimated 4-6× per-seed compute reduction.

## Open issues

- **#32** Add WaDi dataset module.
- **Recalibration application** — apply Tier 1.5 (epoch budgets +
  constant-LR scheduler) and Tier 1.1 (num_workers) from
  `~/plans/compute-efficiency-recalibration.md`. Verify with one
  smoke fit before applying to full ablation sweep.
- **VGAE per-component telemetry gap** — `train_recon`, `train_canid`,
  `train_nbr`, `train_kl` are not flowing through to MLflow. Log calls
  in `vgae_module.py:_training_step_inner` are gated on
  `task_loss.last_recon is not None` — that gate isn't firing.
  Investigation needed before deeper VGAE diagnosis is possible.

## Reference

- Architecture: `docs/reference/`
- Decisions: `docs/decisions/README.md`
- Rules: `.claude/rules/`
- Cross-project plans: `~/plans/`
- Issues: `gh issue list`
