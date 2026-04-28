# GraphIDS Session Plan

> PLAN.md is **current-session work only**. Historical changelogs live in
> `git log`; durable verdicts in `docs/decisions/README.md`; living
> architecture in `docs/reference/`; cross-project plans in `~/plans/`.

## Current state

### Shipped this session (2 commits on main, unpushed — 16 unpushed total)

- `7f23df4` **VGAE mask-recon synthesis** — bundles Phase 0 pre-clean
  (lr=0.006 override drop, score_difficulty relocate to curriculum.py)
  + the atomic synthesis from `~/plans/vgae-mask-recon-and-latent-density.md`.
  Cuts variational/canid/nbr heads (-193 LOC), adds 15% random-mask
  training + 7-round round-robin test + Mahalanobis on mu (diag Σ,
  eps=1e-3 floor) + KL in score (+160 LOC). Forward 5-tuple→3-tuple.
  Net -38 LOC across 16 files. Tests: 205 collect, ruff clean.
  Synthetic-batch instantiation smoke confirms training_step → finite
  loss, validation_step OK, test_step raises pre-calibration error.
  No backward-compat fallthroughs; old ckpts loaded by new code raise
  in `_per_graph_errors`. Fusion cache version bumps 1→2.
- `0e74e3f` **GPU val/test dynamic batching** — `_build_eval_loader`
  now reuses the train-time probe to dynamically pack val/test on
  GPU; falls back to fixed batch_size=32 on CPU. Symptom that drove
  this: jid 47126749 hit 30% GPU util / 95% VRAM (val ran ~945
  batch_size=32 batches/epoch through 2 NFS workers, ~36× more
  forwards than train). Per-example metrics are
  batch-boundary-invariant, so the change only moves throughput.

### Pending — Commit 3 smoke verification (the actual `~/plans/...` Commit 3)

The synthesis hasn't run on real data yet. Smoke must terminate
before the synthesis can be called passing.

**Submit command:**
```bash
cd ~/graphids && source .venv/bin/activate && \
python -m graphids submit configs/ablations/unsupervised/vgae.jsonnet \
    --dataset set_01 --seed 42 --smoke
```

**Pass criterion (single, hard):** `val_discrimination_ratio` peak ≥ 1.5.

**Reference:** old code on seed=43 (jid 47126749) peaked at 1.540 —
that's the no-regression bar. Plan threshold (1.5) is more permissive
since old smoke was on a slightly different seed and still had
lr=0.006.

**Fail action** (verbatim from plan): "**revert Commit `7f23df4`.**
Don't ablate, don't tune mask_rate, don't try without KL or without
Mahalanobis. The synthesis stands or falls together." Eval-loader
fix (`0e74e3f`) stays regardless.

**Sanity checks** (logged but not gating):
- `train_recon_masked > train_recon_unmasked` per-batch (mask training
  is firing, encoder isn't echoing v back via some non-feature path)
- All three score components (recon, mahal, kl) finite + non-zero
  z-normed values across val (verified post-`fit_score_norm`)
- GPU util should now be **substantially > 30%** courtesy of `0e74e3f`
  — if it's still ≤ 50%, something else is starving the dataloader
  beyond the val-batch-size issue

### After the smoke (regardless of pass/fail)

- Run `graphids test` against the best.ckpt — exercises the
  `fit_score_norm` two-pass calibration (mu_mean/std, then component
  norms) and the round-robin scoring path on real data. ~5 min wall.
- If the smoke passes: open issue or commit to remove the warm-allocator
  budget-regression note (`6490eb7`) — it may have been the val-batch
  problem all along, now fixed by `0e74e3f`.

## Open issues — short list

- **#32** Add WaDi dataset module.
- **Tier 1.4 A/B** — cosine→constant LR shipped in two places (base.py
  and vgae_module.py) without controlled validation. One focal GAT pair
  on set_01 seed=42, both at max_epochs=200, would settle whether it
  was the right call. ~50 min/run × 2.
- **#6 design discussion** — `canid` head is now deleted; the prior
  question about canid_weight tuning is moot. Closing implicitly via
  `7f23df4`.
- **Warm-allocator budget regression** — `budget_utilization_pct`
  dropped 101 → 78.6 between historical 4.8 s/ep VGAE run and current
  code (`6490eb7`). May have been masked by the val-batch issue — wait
  for the post-`0e74e3f` smoke before re-investigating.

## Reference

- Architecture: `docs/reference/`
- Decisions: `docs/decisions/README.md`
- Rules: `.claude/rules/`
- Cross-project plans: `~/plans/`
- Issues: `gh issue list`
