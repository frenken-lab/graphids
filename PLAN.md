# GraphIDS Session Plan

> Last updated: 2026-04-04 (session 19 — config stack migration plan)

## Next session — Phase 1 jsonnet migration

Full migration from YAML chain + `merge_yaml_chain` to jsonnet composition.
Single PR, no shadow path. `jsonargparse` / `LightningCLI` stay; Phases 3–4
strip those later.

**Read first:**
- `docs/migration_plan.md` — 7-phase overview
- `docs/phase1_jsonnet.md` — commit-by-commit migration order, shim, gotchas
- `docs/reference/3-chain.md` — current 3-handoff config flow (what we're replacing)

**Execute in order (see `docs/phase1_jsonnet.md` §7 for detail):**
1. Install go-jsonnet 0.20.0+ to `~/.local/bin/` (both login + via dotfiles for all machines)
2. Commit 1: `configs/` skeleton + `graphids/config/jsonnet.py` + ADR 0010
3. Commit 2: **spike** — port exactly `autoencoder + vgae/small`, add one-chain parity test. STOP if it fails.
4. Commit 3: port remaining models/stages/fusion methods; extend parity test to all ~100 recipe chains
5. Commit 3.5: temporary dual-carry of `config_files + jsonnet_path` on `StageConfig` (one commit only)
6. Commit 4: rewrite contracts/planning/resolver/entrypoint/_lightning/cli. Point of no return.
7. Commit 5: **delete** YAML files, `merge_yaml_chain`, `deep_merge`, `apply_dotted_overrides`, `to_override_dict`, `resolve_config_files`, parity test. Add tiny `test_jsonnet_render.py` as permanent guard.
8. Commit 6: update `docs/reference/`, `.claude/rules/`, `CLAUDE.md`, `PLAN.md`

**Exit gates (from `phase1_jsonnet.md` §11):**
- `grep -r merge_yaml_chain graphids/ tests/` → empty
- `ls graphids/config/stages/` → no such directory
- `python -m graphids.orchestrate validate` passes on ablation/smoke_test/final_eval
- `dg launch smoke_test` runs one asset per stage to COMPLETED
- `python -m graphids fit --config configs/stages/autoencoder.jsonnet --trainer.max_epochs 1` runs on gpudebug
- **Net LOC negative** (~−200 to −300). Positive net means a shadow path leaked.

**First actions next session:**
- Verify go-jsonnet install on Pitzer login: `jsonnet --version`
- Sketch `configs/_lib/defaults.libsonnet` + `configs/stages/autoencoder.jsonnet` + `configs/models/vgae.libsonnet` (scales.small only)
- Wire `graphids/config/jsonnet.py::render_config` (implementation in `docs/phase1_jsonnet.md` §6)
- Write the spike parity test (one parametrization only — `autoencoder + vgae/small + hcrl_ch + seed 42`)
- Submit via `scripts/submit.sh tests -k jsonnet_parity` on a CPU SLURM job
- If green → proceed to Commit 3. If red → debug on the 1-chain surface, do NOT port more files.

**Known traps (see `phase1_jsonnet.md` §9):**
- `num_workers: null` must survive as literal null (jsonargparse footgun)
- `+` vs `+:` in jsonnet — always use `+:` for nested dict keys
- String-coercion tolerance in parity comparator (`str(a)==str(b)` at leaves)
- `defaults/trainer.yaml` is silently injected by `parser_kwargs.default_config_files` today — bake into jsonnet, delete the reference
- KD overlay conditional: `if std.length(auxiliaries) > 0 then vgae.kd else {}`
- Dev path needs `.jsonnet` → temp YAML preprocessor in `cli.run_lightning`

## What this session did (2026-04-04, session 19 — config stack migration plan)

Wrote the 7-phase migration roadmap (`docs/migration_plan.md`) and detailed
Phase 1 implementation plan (`docs/phase1_jsonnet.md`). Initial draft was a
shadow-path approach (keep both YAML and jsonnet running in parallel until
Phase 2); rewrote to full-migration-in-one-PR after pushback — git history is
the rollback, no dual-write. No code changes this session; pure planning.

## What this session did (2026-04-04, session 18 — fusion refactor)

Audited `graphids/core/models/` with a focus on fusion complexity. Found a
runtime-crash bug in KD training, 4 tiers of dead code, and 5 reward-shaping
coefficients exposed as kwargs that the paper fixes as methodological
constants. Executed all cuts in one pass. **Net −200 LOC across source,
tests, configs, and docs.**

### Bug fix (Tier 1)

**`teacher_on_device` was missing `@contextlib.contextmanager`** — plain
generator function used as `with teacher_on_device(self, device):` in
`vgae.py:427` and `gat.py:269`. Any VGAE/GAT KD training crashes on the
first step with `TypeError: 'generator' object does not support the context
manager protocol`. Confirmed at runtime. One-line fix. Explains part of
issue #25 (KD pipeline never tested end-to-end).

## Current state

Pipeline converges at LightningCLI (`train_entrypoint.py` → `run_lightning()`).
`ConfigResolver` handles cross-field validation + audit trail; override chain
collapsed 9→3 per ADR 0009 (commit `0837c04`). SLURM submission via
`scripts/submit.sh`. Dagster orchestrator runs as a CPU SLURM job, not login.

Each model config is **one dagster asset = one SLURM job** running
train → test → analyze sequentially. Training under `set -euo pipefail`;
test/analyze best-effort. Per-phase marker files (`.train_complete`,
`.test_complete`, `.analyze_complete`) written on success.

Asset checks split: `checkpoint_complete` (blocking) gates downstream,
`analysis_complete` (non-blocking) informational. Phase status surfaced in
dagster metadata (`phase_train`/`phase_test`/`phase_analyze`).

**Observability:** three layers.

- `turm` — live SLURM queue + log tailing (`PYTHONUNBUFFERED=1` gives real-time)
- Orchestrator JSONL — structured events per run at
  `{SLURM_LOG_DIR}/orchestrator_{job_id}.jsonl`
- `pipeline-status` — dagster + sacct + phase markers aggregate, with
  `--log [FILTER]` and `--follow`

Run records: `run_record.json` sidecar per run (atomic write, Pydantic
schema) → DuckDB catalog rebuildable via `rebuild-catalog`.

## Active

## What this session did (2026-04-04, session 17 — tests audit)

### test-writing rules file

New `.claude/rules/test-writing.md` — auto-loaded into every future session
in this project. Codifies the three-question framework (project-vs-framework,
formula-mirror-vs-property, cite-the-bug), marker discipline, and
anti-patterns from this audit.

## What this session did (2026-04-03, session 16 — lake audit + fusion CPU pipeline)

### Lake artifact audit (set_01)

Audited all 49 run directories under `set_01`. Key findings:

| Finding                              | Scope                         | Action                                                                                            |
| ------------------------------------ | ----------------------------- | ------------------------------------------------------------------------------------------------- |
| Train val_acc 96% but test acc 17%   | All GAT normal/curriculum     | Not a bug — test aggregates 6 subdirs including OOD + excluded attack types. Tracked as GH issue. |
| No analysis artifacts for fusion/DGI | All fusion + DGI runs         | `ANALYSIS_SUPPORTED_MODELS` had only vgae/gat. Added `dgi`. Fusion blocked on deeper issues.      |
| No `best_model.ckpt` for Bandit/DQN  | 2 RL fusion runs              | `automatic_optimization=False` breaks `ModelCheckpoint` silently.                                 |
| Fusion only got 50 epochs            | All fusion runs               | Was `max_epochs: 50`; fixed to 1500, patience 200.                                                |
| ~12 stale orphan directories         | Pre-`DeviceStatsMonitor` runs | Safe to clean up.                                                                                 |

## Key references

Work items live in GitHub issues now, not `docs/backlog/` (deleted
wholesale). Use `gh issue list` or the `/gh` skill.
