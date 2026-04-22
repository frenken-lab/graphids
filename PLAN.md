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

## Next session — validate then relaunch

Preconditions are all met: caches rebuilt (jid 47041299, 2026-04-22),
state-dict refactor + pluggable IdEncoder landed, ckpt compat guard
in `orchestrate/stage._check_ckpt_compat`, stale 2026-04-21 ckpts
deleted. 15 local-safe tests pass.

### Step 1 — SLURM sanity (two jobs, parallel)

```bash
# (a) full test suite — catches GPS conv, curriculum, fast-dev-run regressions
scripts/run --mode cpu --length short --command "python -m pytest tests/ -x"

# (b) single fit smoke on gpudebug — proves Stages 1+2 end-to-end on real data
scripts/run configs/ablations/unsupervised/vgae.jsonnet --dataset set_01 --seed 42 --smoke
```

Both should be green before Step 2. If (b) passes, training works
against the new shared-vocab caches; if (a) passes, no refactor
regression in the model layers. Expected wall: (a) ~20 min, (b) ~1h.

### Step 2 — full set_01 DAG + id_encoding ablation

```bash
# Existing ablation axes (conv_type, gat_loss, gat_sampling, fusion, unsupervised)
scripts/ablation/launch_set_01.sh --seed 42 --cluster cardinal

# New id_encoding axis: 3 presets side-by-side on one seed first
for variant in lookup learned_unk hash; do
  scripts/run configs/ablations/id_encoding/$variant.jsonnet \
      --dataset set_01 --seed 42 --cluster cardinal
done
```

### Step 3 — scale to N=3 + compare

Once seed 42 is green across all axes, mirror the launches for
`--seed 123` and `--seed 777`. Then:

```bash
python -m graphids compare effect-size id_encoding set_01
python -m graphids compare leaderboard id_encoding set_01
```

`compare effect-size` reports Cohen's d + bootstrap CI across the
three arms on every per-attack-category metric — the paper's
Stage-2 evidence table.

### Step 4 — paper data pipeline

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
- **Launcher retry on cross-stage dep race (task #12).** Stage 3's
  first submission failed with "Job dependency problem" because
  Cardinal's scheduler hadn't registered the upstream focal jid yet.
  Either sleep 2-5s between stage boundaries or retry with backoff
  on that specific error.

## Open issues

- **#32** Add WaDi dataset module.

## Reference

- Architecture: `docs/reference/`
- Decisions: `docs/decisions/README.md`
- Rules: `.claude/rules/`
- Cross-project plans: `~/plans/`
- Issues: `gh issue list`
