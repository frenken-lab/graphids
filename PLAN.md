# KD-GAT Session Plan

> Last updated: 2026-04-01 (session 7b — observability: tools + phase markers + pipeline-status CLI)

## Current State

Pipeline converges at LightningCLI (`train_entrypoint.py` → `run_lightning()`). ConfigResolver
handles cross-field validation + audit trail. SLURM submission via `scripts/submit.sh` works.
Dagster orchestrator runs as CPU SLURM job (not login node).

Each model config is **one dagster asset = one SLURM job** running train→test→analyze
sequentially. Training runs under `set -euo pipefail`; test/analyze are best-effort
(`set +euo pipefail`). Per-phase marker files (`.train_complete`, `.test_complete`,
`.analyze_complete`) written on success.

Asset checks are split: `checkpoint_complete` (blocking) gates downstream assets,
`analysis_complete` (non-blocking) is informational only. Checkpoint checks report
per-phase status (`phase_train`, `phase_test`, `phase_analyze`) in dagster metadata.

Observability: `python -m graphids pipeline-status` shows aggregated asset status
(dagster + phase markers + sacct). DeviceStatsMonitor logs CUDA memory stats per step.
nvitop, reportseff, SlurmTUI installed as complementary tools.

## What this session did (2026-04-01, session 7b — observability)

### Observability layer — existing tools + thin glue code

Researched SLURM/HPC monitoring ecosystem. Decided against turm fork in favor of
installing existing tools + ~200 lines of custom glue.

**Tools installed:**
- nvitop 1.6.2 — real-time GPU monitoring TUI + Python API (in .venv)
- reportseff 2.10.3 — multi-job efficiency reports, replaces seff (in .venv)
- SlurmTUI 0.6.3 — sacct history TUI with Textual (standalone tool)

**Phase markers** (`generate_script()` in `slurm/slurm.py`):
- `.train_complete`, `.test_complete`, `.analyze_complete` written on phase success
- Uses `if cmd; then touch; fi` pattern (not `$?` — immune to line insertions)
- `PHASE_MARKERS` constant in `config/runtime.py`, re-exported via `__init__.py`
- Checkpoint check in `checks.py` reports `phase_train/test/analyze` in dagster metadata
- Legacy runs (no markers) show `-` instead of false failures

**DeviceStatsMonitor** (`_lightning.py`):
- Added as forced callback (like checkpoint/early_stopping)
- Logs CUDA memory stats per step to WandB/CSV automatically
- nvitop's Lightning callback was deprecated + incompatible with pytorch_lightning 2.6.1

**Pipeline status CLI** (`graphids/commands/pipeline_status.py`):
- `python -m graphids pipeline-status [--limit N] [--json]`
- Uses `DagsterInstance.get_asset_records()` batch API (no server needed)
- Deduplicates `get_run_by_id()` calls by run_id
- Derives `run_dir` from checkpoint path using `CKPT_SUBPATH` depth (not hardcoded)
- Rich table output with per-asset status + phase markers + wall time + job ID
- JSON output mode for scripting

**Plan:** `~/plans/observability-plan.md` — full research + architecture + deferred items

### Issues updated

| Issue | Status |
|-------|--------|
| `issues/pipeline-observability.md` | **Partially resolved** — CLI status done, alerting deferred |
| `issues/slurm-phase-reporting.md` | **Resolved** — phase markers implemented (session 7) |

## What this session did (2026-04-01, session 7)

### Wiring audit and contract hardening

Categorized 17 bugs from sessions 5-6: config resolution (6), device management (3),
checkpoint/IO (2), SLURM/infra (3), model contract (1), dagster orchestration (2).
Identified the config→checkpoint→IO wiring path as the dominant bug source.

Audited the wiring path and found 12 fragility points. Fixed 9 (3 critical, 3 high,
3 medium):

**Critical fixes:**
- `merge_yaml_chain` raises `FileNotFoundError` on missing config files (was silent skip)
- `to_override_dict` warns on unmapped upstream assets, raises `KeyError` on unmapped
  model families (was silent drop of upstream checkpoints)
- `checkpoint_path()` delegates to `PathContext` (was parallel path derivation that could diverge)

**High fixes:**
- `to_override_dict` warns on `runtime_overrides` key conflicts (was silent last-write-wins)
- `touch_complete` uses `os.open` + `fsync` on file and directory for NFS durability (was bare `touch`)
- `_flatten_dict` rejects non-scalar values with `TypeError` (was silent `str()` on lists/dicts)

**Medium fixes:**
- `generate_script` uses `set -euo pipefail` before preamble + training
- `lake_root`/`user` read at materialization time, not `build_defs` time (was baked at import)
- Config snapshot applies `LINK_TARGETS` for manual replay reproducibility

### Fusion checkpoint fallback

`best_model.ckpt` never created for fusion RL (bandit/DQN) — only `last.ckpt` exists.
Fixed `run_test_from_spec`, `checks.py`, and `assets.py` to prefer `best_model.ckpt`,
fall back to `last.ckpt`.

### Issues updated

| Issue | Status |
|-------|--------|
| `issues/slurm-phase-reporting.md` | **Resolved** — phase markers implemented |
| `issues/config-system-overhaul.md` | W4 (collision detection) resolved, validation hardened |
| `issues/override-pipeline-consolidation.md` | 5 mitigations added, `OverrideChain` proposal still open |

## Blocking — Must do before ablation

1. **Run clean smoke test** — resubmit with current code (includes all session 6+7 fixes):
   ```bash
   # Clear seed 99 markers first
   for d in /fs/ess/PAS1266/kd-gat/dev/rf15/hcrl_sa/*/seed_99; do
     rm -f "$d/.complete"; rm -rf "$d/artifacts"
   done
   KD_GAT_RECIPE=graphids/config/recipes/smoke_test.yaml \
     scripts/submit.sh ablation --assets '*' --partition 'hcrl_sa|99'
   ```

2. **Run config/override tests on SLURM**:
   ```bash
   scripts/submit.sh tests -k "test_overrides or test_config or test_merge_parity or test_submit_sh or test_cli_routing or test_recipe_expand_kd"
   ```

## Next

1. Clean smoke test (blocking item 1)
2. Run tests (blocking item 2)
3. Launch ablation (`plans/experiment-sweep-plan.md`)

## Key References

| Doc | Purpose |
|-----|---------|
| `plans/architecture/slurm-job-consolidation.md` | **Implemented** — bundle train+test+analyze in one SLURM job |
| `issues/config-system-overhaul.md` | Config overhaul tracker — mostly complete, W4 resolved session 7 |
| `issues/per-stage-recipe-overrides.md` | Global vs stage-specific overrides (open) |
| `issues/slurm-phase-reporting.md` | **Resolved** session 7 — phase markers implemented |
| `issues/analyzer-manifest-lifecycle.md` | Manifest ownership (open) |
| `issues/override-pipeline-consolidation.md` | 4-hop override flow — 5 mitigations applied, architecture open |
| `plans/experiment-sweep-plan.md` | 17-config ablation matrix |
| `plans/open_issues.md` | All deferred items |
