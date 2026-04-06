# GraphIDS Session Plan

> Last updated: 2026-04-05 (session 26 — Typer CLI + config reorg)

## What this session did (2026-04-05, session 26 — Typer CLI + config reorg)

Two pieces of work:

### Config reorg: stage/family name migration + broken import fixes

Completed the remaining Phase 2 orchestration + Phase 3 test updates from
the config reorg that consolidated model families (vgae/dgi/gat →
unsupervised/supervised) and stage names (normal/curriculum → supervised):

- `orchestrate/{contracts,resolve,cross_field,planning,recipes}.py` —
  stage name updates, family checks, dead curriculum code removed
- `orchestrate/logger.py`, `core/models/{gat_module,vgae_module}.py` —
  stale `"normal"`/`"curriculum"` refs
- ~15 files with broken imports from an earlier incomplete refactor:
  `graphids.config.shared` → `orchestrate.shared`/`slurm.resources`,
  `graphids.core.contracts` → `orchestrate.contracts`/`core.analysis.schemas`,
  `graphids.config.contracts` → `orchestrate.recipes`,
  `SLURM_ACCOUNT`/`SLURM_LOG_DIR` → `slurm.env`
- Fixed `dep["model"]` → `dep["family"]` bug in `planning.py`
  (topology.json uses `"family"` key)
- Fixed pre-existing `DGIModule`/`VGAEModule`/`GATModule` `__init__.py`
  re-exports, `fusion_reward.py` indentation error
- All 6 test files updated (stage names, family names, imports)

### Typer CLI: replace argparse + jsonargparse with principled grouping

Replaced the accumulated argparse shim collection (`graphids/commands/`,
12 files) and jsonargparse dependency with a Typer CLI following the
responsibilities doc chain:

```
Typer parses CLI args → jsonnet renders → Pydantic validates → code runs
```

- Created `graphids/cli/` package: `app.py` (shared options, parse_tla,
  apply_overrides), `_training.py`, `_analysis.py`, `_data.py`,
  `_orchestrate.py`, `_slurm.py`
- Created `graphids/core/train_entrypoint.py` — the missing module that
  3 files referenced. Shared render→validate→instantiate→trainer chain
  for both dev path and pipeline path.
- Rewrote `__main__.py` for Typer
- Deleted `graphids/commands/` (12 files)
- Replaced jsonargparse with typer in `pyproject.toml`
- Fixed `pipeline-status` broken import (`SLURM_LOG_DIR`)
- Fixed `fusion.py` stale import from old `commands/extract_fusion_states`
- Rewrote `test_cli_routing_smoke.py` for Typer CliRunner

Commands grouped by `rich_help_panel`: Training (4), Analysis (1),
Data (3), Orchestration (3), SLURM (2), plus hidden `_finalize-record`.

## Next session — SLURM smoke test

Verify end-to-end via `scripts/slurm/submit.sh tests`. The Typer CLI,
jsonnet render, Pydantic validation, and instantiate chain are all
wired but only import-tested on login node.

**Known deferred items:**

- `instantiate.py` still has broken imports (`graphids.callbacks`,
  `CurriculumEpochCallback` without import). These fire at training
  time, not import time.
- `orchestrate/entrypoint.py` imports `run_training_from_spec` /
  `run_test_from_spec` from `core.train_entrypoint` — now exists.
- `analyze` command interface changed: `--analyzer.ckpt_path` →
  `--tla 'ckpt_path="..."'` (jsonnet TLA instead of jsonargparse
  dotted override)

## Active
with zero LightningCLI / jsonargparse involvement. `_lightning.py` and
`cli.py` are gone; `GraphIDSCLI`, `build_cli`, `schema_parser`,
`CLI_KWARGS`, `WandbSaveConfigCallback`, `patch_config_paths`, and
`validate_cli_chain` are all deleted.

**Read first before starting Phase 4:**

- `docs/migration_plan.md` — Phase 4 "Jsonargparse Retooling" section
  (ActionJsonnet path vs stdlib argparse for `commands/analyze.py`)
- `graphids/commands/analyze.py` — jsonargparse retooling (jsonnet parser_mode)
- `graphids/core/artifacts/analyzer.py` — `Analyzer.__init__` signature
  that the parser reads

**Known Phase 3 deferred items:**

- `commands/analyze.py` still imports `jsonargparse.ArgumentParser` to
  build an auto-parser over `Analyzer.__init__`. Phase 4 keeps
  jsonargparse and retools it for Jsonnet-backed configs + type-hint
  validation per `docs/migration_plan.md §Phase 4`.
- Dotted-key typos in `trainer_overrides` / `stage_overrides` used to be
  caught at planning time by the deleted `validate_cli_chain` (via
  `parser.parse_object`). Post-Phase-3 they now fail at instantiation
  (Trainer / Model `__init__` kwarg mismatch). Consider re-adding
  stage-libsonnet override validation in `ValidatedConfig` if this
  becomes a pain point — but let the next production sweep expose it
  first.
- Fusion stage still absorbs `auxiliaries=[]` and `vgae_ckpt_path=null`
  as ignored TLAs (Phase 1 deferred, unchanged in Phase 3).

## What this session did (2026-04-05, session 22 — Phase 3 strip LightningCLI)

LightningCLI + `jsonargparse` (for the training path) deleted. The
rendered jsonnet dict is now consumed directly by
`graphids.core.instantiate.instantiate` which imports class_paths via
`importlib`, applies signature-filtered link_arguments, constructs the
forced callback set explicitly, and returns a wired `(trainer, model,
datamodule)` triple. The dev-path argparse entrypoint moved under
`commands/` to comply with the project convention that argparse-based
CLI tools live there.

### Bug fix (Tier 1)

**`teacher_on_device` was missing `@contextlib.contextmanager`** — plain
generator function used as `with teacher_on_device(self, device):` in
`vgae.py:427` and `gat.py:269`. Any VGAE/GAT KD training crashes on the
first step with `TypeError: 'generator' object does not support the context
manager protocol`. Confirmed at runtime. One-line fix. Explains part of
issue #25 (KD pipeline never tested end-to-end).

**Observability:** three layers.

- `turm` — live SLURM queue + log tailing (`PYTHONUNBUFFERED=1` gives real-time)
- Orchestrator JSONL — structured events per run at
  `{SLURM_LOG_DIR}/orchestrator_{job_id}.jsonl`
- `pipeline-status` — dagster + sacct + phase markers aggregate, with
  `--log [FILTER]` and `--follow`

Run records: `run_record.json` sidecar per run (atomic write, Pydantic
schema) → DuckDB catalog rebuildable via `rebuild-catalog`.

## Active

## What this session did (2026-04-05, session 25 — Jsonnet recipe expansion)

- Moved recipe expansion logic into Jsonnet (`configs/_lib/recipes.libsonnet`) with a
  `configs/recipes/_expand.jsonnet` entrypoint, keeping Python as validation + wrapper.
- Simplified `recipe_expand.py` to delegate expansion to Jsonnet and updated tests/docs
  to reference the new expansion path.

## What this session did (2026-04-05, session 23 — Phase 4 jsonargparse retooling)

- Switched analyzer configs to Jsonnet (`graphids/config/stages/analyze_*.jsonnet`) and
  updated usage docs/refs to point at the new paths.
- Retooled `commands/analyze.py` to use jsonargparse `parser_mode="jsonnet"` with
  `--config`, keeping CLI overrides intact.
- Upgraded jsonargparse dependency to full extras (`all,shtab,argcomplete`) and
  aligned the Phase 4 docs + config-system notes with the new direction.

## What this session did (2026-04-05, session 24 — config reorg execution)

- Moved `StageConfig` and `ResourceSpec` into `graphids/config/shared.py` and
  re-routed imports across orchestrate/SLURM/tests.
- Replaced orchestrate cross-field checks with Pydantic validation in
  `graphids/config/schemas.py` + `graphids/config/cross_field.py`.
- Migrated resource profiles to `configs/resources/job_profiles.json` and
  removed the legacy YAML profile files; updated `slurm/resources.py` loader.
- Removed the legacy `orchestrate validate` CLI and updated docs/refs.
- Added reward-default resolution from `configs/fusion/reward.libsonnet`
  for DQN/Bandit instantiation and added safe defaults for missing
  GAT config fields in test fixtures.

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
