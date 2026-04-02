# Open Issues

> Audited: 2026-04-02. Resolved items removed (see git history).

## Performance (from 2026-03-30 profiling)

- **CurriculumDataModule rebuilds DataLoader every epoch** → kills persistent workers →
  ~40 min spawn overhead over 300 epochs. Fix: create DataLoader once, update
  `CurriculumSampler.set_epoch()` only. ~10-line fix.
- **PSS verification on GPU node** — RSS double-counts shared mmap pages. Submit job with
  `smaps_rollup` in `worker_init_fn` to confirm. If confirmed, reduce `--mem` in profiles.
- **VRAM probe validation** — compare `probe_bytes_per_node` against `DeviceStatsMonitor`
  peak. If >40% conservative, split `_GRAD_MULTIPLIER` for KD.

## Ablation / Experiment (from runs 001-005)

- **GPS `batch_size` right-sizing** — O(N^2) global attention OOMs on V100. Need
  GPS-specific cap (~256-384) or `attn_type="performer"`.
- **Dataset-scoped data staging** — `stage_data.sh` copies entire 86GB cache to TMPDIR.
  Should copy only needed dataset (4-6GB).
- **Scratch cache cleanup** — 64GB of stale versioned dirs (v3-v7) on scratch.
- **ESS stale run dirs** — dirs at `/fs/ess/PAS1266/kd-gat/dev/rf15/set_01/` without
  `.complete` markers. Need cleanup policy.

## Code Cleanup (from preprocessing + models consolidation)

- **`edge_to_tensor` in `features.py`** — zero production callers. Delete candidate.
- **Broken test imports in `test_features.py`** — `edge_features`, `_assemble_chunk_numpy`,
  `_numpy_to_data` don't exist in `features.py`.
- **`prepare_data()` / `setup(stage)` separation** — missing on all DataModules. `setup()`
  ignores `stage` param, loads everything unconditionally. Blocks DDP.
- **No `predict_dataloader()`** on any DataModule.
- **`lr` / `weight_decay` params in GAT/DGI `__init__`** — dead code (saved to hparams
  but never read). Cleanup candidate.
- **`T_max: 300` in stage YAMLs is static** — old code used `self.trainer.max_epochs`
  dynamically. A `link_arguments` could wire this.

## Orchestration (from dagster rebuild)

- **Dagster testing layers 0-3 missing** — no unit tests for pure Python, dagster unit
  with mock SLURM, dagster integration with real IOManager.

## Observability (deferred from 2026-04-01)

- **`--watch` mode for `pipeline-status`** — auto-refresh like `watch squeue` but with
  DAG context.
- **Structured JSON logs** — `structlog.processors.JSONRenderer()` when `SLURM_JOB_ID`
  is set, enables `jq` parsing.

## Config (from codex-refactor audit)

- **Orphaned YAML files** — `config/schema/*.yaml` (4 files), `config/overrides/` (4 files),
  `config/matrix/allowed_combinations.yaml`. No Python loads them. Delete unless wired.
- **`experimentruns` fallback** — `LAKE_ROOT` defaults to relative in-repo path when
  `KD_GAT_LAKE_ROOT` unset. `.env.example` should include it.

## Scripts (from scripts-refactor)

- **`smoke-test` subcommand** — not implemented. Dagster ablation on hcrl_sa serves as smoke test.
- **`run_tests.sh`** — not created, `scripts/submit.sh tests` handles it. Low priority alias.
