# GraphIDS Session Plan

> PLAN.md is **current-session work only**. Historical changelogs live in
> `git log`; durable verdicts in `docs/decisions/README.md`; living
> architecture in `docs/reference/`; cross-project plans in `~/plans/`.

## Current state

### Shipped this session (3 commits on main, unpushed)

- `0d20b9c` uv.lock regen (post-OTel rip drift)
- `fdfc518` **GAT recalibration**: max_epochs 300→200, patience 100→30,
  added `min_delta=0.001` to EarlyStopping (callback gained the field),
  dropped CosineAnnealingLR from base.py.
- `95bb8df` **VGAE recalibration** — five coupled changes:
  1. monitor: `val_discrimination_gap` → `val_discrimination_ratio` (gap
     shrinks monotonically as both losses converge → useless as max-mode
     signal; ratio grows as discrimination strengthens). Both ckpt + ES
     track the ratio. Gap kept as diagnostic only.
  2. `validation_step` 3 forwards → 1 forward (per-class via masking, not
     re-batching). val_loss now excludes KL — magnitudes shift, curve
     shape comparable.
  3. Tier B z-norm scoring: 6 scalar buffers + fitted flag on the module.
     `fit_score_norm()` calibrates against benign-val at test-start
     (trainer.test hook mirrors OCGIN's calibrate_svdd_center pattern).
     Falls back to fixed-weight composite when buffers empty (older ckpts
     keep working).
  4. `extract_features` deduplicated against `_per_component_errors`
     (now returns z too). Column order preserved for cached fusion states.
  5. `autoencoder.jsonnet`: max_epochs 1200→600, precision 16-mixed→32-true
     (fp16 was numerically safe but no throughput win measured); cosine
     scheduler dropped from VGAEModule.build_optimizers.

### In flight — smoke jid 47125867

Single-forward val + ratio monitor + z-norm calibration **launched but
not finished**. User is watching the Monitor stream personally. Expected
~30 min wall on gpudebug. What to verify when it terminates:

- val_discrimination_ratio trajectory grows ~1.1 → ~1.3 over 50 epochs
  (matches the prior smoke's same numbers, just on the right metric).
- Per-epoch time DROPS from 41 s/ep baseline (the goal of the val 3→1
  refactor). The historical 4.8 s/ep run is the lower bound; likely
  lands somewhere in between (warm-allocator probe budget regression
  is still in effect, separately).
- ModelCheckpoint saves a late-training epoch (not ep 0 like the broken
  monitor did). Best-epoch should be the highest-ratio epoch.
- All metrics finite (back on fp32, expected stable).

### What's not yet covered by the smoke

- Z-norm calibration end-to-end: only fires during `graphids test`, not
  during `submit ... vgae.jsonnet`. After the smoke fit terminates,
  running `graphids test --ckpt-path {best.ckpt} --dataset set_01` will
  exercise the trainer hook + fit_score_norm + branched scoring. ~5 min
  wall.
- `extract_features` dedup: only matters when `extract-fusion-states`
  CLI runs. Verified by import + structural test, not exercised on real
  data this session.

## Open issues — short list

- **#32** Add WaDi dataset module.
- **Audit #3** — move `score_difficulty` (vgae.py:291-328, ~38 LOC)
  out of the model class to `core/data/curriculum.py` where the
  curriculum scorer interface lives. Pure relocation, no behavior change.
- **Audit #4** — inline + delete `from_config` (vgae.py:265-282).
  cfg→kwargs translator with no logic; caller can build inline.
- **Audit #5** — `build_encoder_stack` should return the targets it
  picks so the decoder reverses *those*, not a duplicate derivation
  (vgae.py:101-104). Drift risk between encoder/decoder construction.
- **Audit #6 cleanup** — stale module docstring in vgae_module.py:22-35
  (references `score_*_weight` "read back off self.loss_fn" no longer
  true), unused `import torch.nn.functional as F` inside extract_features,
  `ModelType` noqa is wrong (it IS used in annotation).
- **Tier 1.4 A/B** — cosine→constant LR shipped in two places (base.py
  and vgae_module.py) without controlled validation. One focal GAT pair
  on set_01 seed=42, both at max_epochs=200, would settle whether it
  was the right call. ~50 min/run × 2.
- **#6 design discussion** — `canid` head was at random-baseline
  cross-entropy (~0.94) at ep 20 in the prior smoke; either
  canid_weight=0.1 is too small to drive the head or the head
  architecture is undersized. User wants to discuss after this batch
  of changes lands.
- **`lr: 0.006` stale workaround** in
  `configs/ablations/unsupervised/vgae.jsonnet:23`. The comment claims
  the bump escapes a "~2800 floor for ~50% of epochs" — that plateau
  was the pre-#43 bug and no longer exists post-fix. Override is now
  unjustified; should drop or rejustify on current data.
- **Warm-allocator budget regression** — `budget_utilization_pct`
  dropped 101 → 78.6 between the historical 4.8 s/ep VGAE run and
  current code (commit `6490eb7`). VGAE got smaller batches without
  the GAT-specific util benefit. Independent of the val refactor.

## Reference

- Architecture: `docs/reference/`
- Decisions: `docs/decisions/README.md`
- Rules: `.claude/rules/`
- Cross-project plans: `~/plans/`
- Issues: `gh issue list`
