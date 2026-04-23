# GraphIDS Session Plan

> PLAN.md is **current-session work only**. Historical changelogs live in
> `git log`; durable verdicts in `docs/decisions/README.md`; living
> architecture in `docs/reference/`; cross-project plans in `~/plans/`.

## This session — ablation analysis substrate + eval demarcation

Phases from `~/plans/ablation-analysis-substrate.md`, plus a follow-up
profile collapse for the SLURM surface.

- **Phase 0 — Bouthillier 2021 §5 review** (no code). Agent extracted the
  paper; summary at `~/plans/bouthillier-2021-section-5.md`. Recommended
  N = 29 per variant under the paper's γ=0.75 / α=β=0.05 decision rule.
  Researcher constraint: N ≤ 3 (OSC allocation) — documented in
  `~/plans/phase-3-revised.md`.
- **Phase 1 — per-class test metrics.** Replaced `binary_test_metrics`
  for classifier-flavor models (GAT + all fusion) with unified
  `classification_test_metrics` (Multiclass* + ClasswiseWrapper). Test
  step buffers `(N, K)` probabilities; VGAE/DGI keep binary + threshold.
  New keys: `accuracy`, `mcc`, `ece`, `{f1,precision,recall,specificity,
  auc,ap}_{macro,weighted,per_class/<name>}`.
- **Phase 2 — parent/child MLflow runs.** Added
  `start_parent_run(group, variant, dataset)` in `_mlflow.py`; children
  link via `MLFLOW_PARENT_RUN_ID` env var → `mlflow.parentRunId` tag. CLI:
  `python -m graphids mlflow-start-parent`. Launcher opens 16 parents
  upfront; `_fit()` injects the env var per-variant.
- **Phase 3 — `graphids.analysis.compare`.** Four MLflow-driven functions:
  `leaderboard`, `tie_candidates`, `effect_size` (Cohen's d + bootstrap
  CI, no p-values at N≤3), `expected_max`. CLI: `python -m graphids compare
  {leaderboard|ties|effect-size|expected-max} <group> <dataset>`.
- **Phase 5 — SLURM profile collapse + CLI tighten.** Two profiles only
  (`gpu`, `cpu`). `scripts/run` absorbs `scripts/slurm/submit.sh`; takes
  `<preset.jsonnet>` (training) or `--mode {gpu|cpu} --command "..."`
  (ops). `submit-profile` CLI deleted; MLflow walltime-history moved to
  `graphids.slurm.sizing`. Eval demarcation falls out naturally: test
  commands submit with `--mode cpu` — no GPU allocation, no separate
  profile needed.

## 2026-04-22 session — Stages 1+2 OOV handling + pluggable IdEncoder

Driven by the 2026-04-21 Cardinal DAG failure: all 10 fit-then-test
jobs crashed with `IndexError: index out of range in self` at the
CAN-ID embedding lookup. Diagnosed as per-split vocab drift — each
split built its own `arb_id → index` mapping, so test subdirs with
novel arb_ids over-flowed the train-sized embedding table. Research
pass, design decision, and implementation in one session.

- **Research.** `~/plans/oov-embedding-handling.md` (v2, 2021–2026
  source gate, >20 cites per source). Verdict: industrial recsys has
  converged on hash-bucketed embeddings for dynamic/OOV vocabs; CAN
  IDS literature has no standard treatment. Three-stage plan.
- **Stage 1 — shared vocab (fix the crash).** New
  `graphids/core/data/vocab.py` + `CANBusSource.build()` scans union
  of all splits' source_dirs, persists `vocab.json`, digest stamped
  into `cache_metadata.json`. `METADATA_SCHEMA_VERSION` bumped 2→3.
  Rebuild verified jid 47041299 (14 min, 6 datasets; set_01 has 2046
  unique arb_ids + UNK@0 = 2047).
- **Pluggable `IdEncoder` architecture.** New
  `graphids/core/models/id_encoding/` with `IdEncoder` base,
  `LookupIdEncoder` (Stage 3 `p_unk_drop` plumbing), `HashIdEncoder`
  (Stage 2 primary). `InputEncoder` takes `id_encoder: IdEncoder`;
  VGAE/GAT/DGI Module wrappers gained `id_encoder_class_path` +
  `id_encoder_kwargs` config surface. `from_vocab_size` classmethod
  bridges `datamodule.num_ids` → encoder-native params.
- **Stage 2 ablation presets.** Three jsonnet presets under
  `configs/ablations/id_encoding/`: `lookup.jsonnet` (baseline),
  `learned_unk.jsonnet` (Stage 3 `p_unk_drop=0.1`), `hash.jsonnet`
  (Stage 2 HashIdEncoder, `k=2`, `num_buckets_factor=4`).
  All three validate through the Pydantic gate.
- **Paper.** `~/kd-gat-paper/paper/content/methodology.md` gained a
  new §"Handling Out-of-Vocabulary Arbitration IDs" subsection with
  9 new 2021–2026 bib entries (Monolith, Unified Embedding, Yan hash,
  recsys surveys, CAN-IDS survey, ROAD, CAN-MIRGU, CAN-BERT).
- **State-dict break.** `input_encoder.id_embedding.weight` →
  `input_encoder.id_encoder.embedding.weight`. Ckpts from the 2026-04-21
  Cardinal DAG are formally unloadable by design.

## 2026-04-23 session — launcher generalization + seed 42 launch

Preconditions from the 2026-04-22 session were all met before launch:
caches rebuilt (jid 47041299), state-dict refactor + pluggable
IdEncoder landed, ckpt compat guard in place, stale 2026-04-21 ckpts
deleted. SLURM sanity pytest (jid 47045029) green; Pitzer V100 VGAE
smoke (jid 47045030) walltimed at gpudebug's 1h limit with
checkpoints written (expected compute-tiny behavior per commit
6522722).

- **Launcher generalized.** `scripts/ablation/launch_set_01.sh` →
  `launch_ofat.sh` (git mv). Added `--dataset <name>` flag (default
  `set_01`). DAG shape unchanged.
- **id_encoding folded into Stage 1.** Three variants
  (`lookup`/`learned_unk`/`hash`) added to the parallel standalone
  pool. Stage 1 count 10 → 13. Removes the manual for-loop from
  PLAN's prior Step 2.
- **Cardinal fit wall bumped to 1:30:00.** `configs/resources/
  submit_profiles.json` — the prior 1:15:00 cut one pre-rebuild gat
  fit to 6 minutes of margin (69.3 min elapsed). 90 min gives 20+
  min safety margin across all 19 historical variants; VGAE on
  Cardinal has zero prior FINISHED data so this is the one-line
  insurance.
- **Docs synced.** 6 doc references updated to the new launcher
  filename + Stage 1 variant list (README, config-system rule,
  config-architecture reference, copilot-instructions,
  PLAN).

### Seed 42 launched (39 SLURM jobs on Cardinal)

```
Stage 0: 2 jobs  (vgae fit+test)              JIDs 8724431–8724432
Stage 1: 26 jobs (13 fits + 13 tests)          JIDs 8724433–8724460
Stage 2: 2 jobs  (curriculum_vgae fit+test)    JIDs 8724461–8724462
Stage 3: 1 job   (extract-fusion-states)       JID  8724464
Stage 4: 8 jobs  (4 fusion fits + 4 tests)     JIDs 8724465–8724472
```

**Dep race hit again** — Stage 3 first submission failed with "Job
dependency problem"; re-submitted manually after a few-second pause.
Launcher bug (PLAN's open #12) is still live; must be fixed before
seed 123 / 777 mirror launches.

**MLflow parents skipped** — all 19 `python -m graphids
mlflow-start-parent` calls returned nonzero ("mlflow unavailable");
children will log to MLflow but won't link via `MLFLOW_PARENT_RUN_ID`.
Recoverable post-hoc by stamping `mlflow.parentRunId` tags across the
child runs by (group, variant, dataset).

## Next session — triage seed 42 + fix launcher race

### Step 1 — inspect seed 42 results

```bash
squeue -M cardinal -u $USER -o '%.10i %.22j %.2t %.10M %.10l %R' | head -50
sacct -M cardinal -u $USER --starttime=2026-04-23 --format=JobID,JobName%30,State,Elapsed,ExitCode -P
```

Check each stage:
- Stage 0 (vgae 8724431): did it finish within 90min? Post-rebuild
  VGAE on Cardinal H100 is untested.
- Stage 1 standalone: 13 fits — all COMPLETED? Which ones bumped the
  90min wall vs completed cleanly?
- Stage 3 (extract-fusion-states 8724464): needs both upstream ckpts
  to exist, so this is the end-to-end dep-resolution test.
- Stage 4 (fusion × 4): each runs afterok states — same dep-race
  class, but only one dep edge so less brittle than Stage 3.

### Step 2 — fix the launcher dep-race (blocks seed 123/777)

Two options, smaller is better:
1. **Sleep 3s between stage boundaries** in `launch_ofat.sh`
   (between Stage 1 submission loop and Stage 3's `SBATCH_DEP=`
   call). Cheapest.
2. **Retry-with-backoff on "Job dependency problem"** in
   `scripts/run`. Cleaner but touches the hot path.

### Step 3 — fix MLflow parent-run failure

Diagnose why `python -m graphids mlflow-start-parent` returned
nonzero during the 2026-04-23 launch. Likely either (a) the launcher
sourced `.env` but `MLFLOW_TRACKING_URI` wasn't exported, or (b) the
SQLite lock from the system-metrics sampler. If parents can't be
recovered at submit time, add a post-hoc one-shot to stamp
`mlflow.parentRunId` on existing child rows keyed by (group,
variant, dataset).

### Step 4 — scale to N=3 + compare

Once seed 42 is green (or failures are understood), mirror launches
for `--seed 123` and `--seed 777`:

```bash
scripts/ablation/launch_ofat.sh --dataset set_01 --seed 123 --cluster cardinal
scripts/ablation/launch_ofat.sh --dataset set_01 --seed 777 --cluster cardinal
```

Then:
```bash
python -m graphids compare effect-size id_encoding set_01
python -m graphids compare leaderboard id_encoding set_01
```

### Step 5 — paper data pipeline

Once N=3 is in, run `graphids/analysis/export_hf.py` (design at
`~/plans/hf-dataset-design.md`) to push the leaderboard +
effect-size tables to HF, then `~/kd-gat-paper/data/pull_data.py`
pulls them so paper rendering is device-agnostic.

## Still-open follow-ups

None blocking the next session. All Stage 1+2 correctness work is
landed; the remaining items are verification (SLURM jobs above) and
paper-pipeline integration.
- **Retroactive eval on existing fit-only ckpts.** One-shot sweep script
  to populate MLflow test rows for historical runs. Sizable compute
  (~20 fit-only ckpts × 5 min CPU each) but straightforward.
- **Phase 4 — seed-expansion launcher wrapper.** Bash: take `--seeds
  1,2,3` and loop. Low priority given N ≤ 3 screening workflow already
  uses the existing `--seed` loop.
- **HF dataset export pipeline.** Design drafted at
  `~/plans/hf-dataset-design.md`. Implementation starts once seed 42
  completes cleanly: `graphids/analysis/export_hf.py` reads MLflow +
  run_dirs, assembles the bucket tree, pushes versioned revision.
  Paper build (`~/kd-gat-paper/data/pull_data.py`) consumes from HF
  so rendering is device-agnostic.
- **fp16 overflow — hcrl_sa only.** Confirmed NOT a regression:
  set_01 fp16 ran clean on V100 past the crash window (job 46974907).
  hcrl_sa has outlier features under the train-only scaler (cache v9)
  that push something in VGAE's forward past fp16 range. Low
  priority — smoke tests on hcrl_sa can use `--set
  trainer.precision="32-true"`; real ablation uses set_01.
- **Launcher retry on cross-stage dep race (task #12, FIRED AGAIN
  2026-04-23).** Stage 3's first submission failed with "Job
  dependency problem" because Cardinal's scheduler hadn't registered
  the upstream focal jid yet. This has now fired twice — elevated
  priority. Must be fixed before seed 123 / 777 launches. Fix:
  sleep 3s between stage boundaries in `launch_ofat.sh` OR
  retry-with-backoff on that specific error in `scripts/run`.
- **MLflow parent-run creation silently fails at submit time
  (2026-04-23).** All 19 `python -m graphids mlflow-start-parent`
  calls returned nonzero during the seed 42 launch — children logged
  but aren't grouped. Cause unknown (possibly env var not exported
  under the launcher's subshell, or SQLite lock from system-metrics
  sampler). Either fix at source, add `set -x` / stderr capture, or
  add a post-hoc stamp-by-tag script.

## Open issues

- **#32** Add WaDi dataset module.

## Reference

- Architecture: `docs/reference/`
- Decisions: `docs/decisions/README.md`
- Rules: `.claude/rules/`
- Cross-project plans: `~/plans/`
- Issues: `gh issue list`
