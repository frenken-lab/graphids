# Open Issues

> Audited: 2026-04-02. Resolved items removed (see git history).

## Performance (from 2026-03-30 profiling)

- **PSS verification on GPU node** — RSS double-counts shared mmap pages. Submit job with
  `smaps_rollup` in `worker_init_fn` to confirm. If confirmed, reduce `--mem` in profiles.
- **VRAM probe validation** — compare `probe_bytes_per_node` against `DeviceStatsMonitor`
  peak. If >40% conservative, split `_GRAD_MULTIPLIER` for KD.

## Ablation / Experiment (from runs 001-005)

- **Scratch cache cleanup** — 64GB of stale versioned dirs (v3-v7) on scratch.
- **ESS stale run dirs** — dirs at `/fs/ess/PAS1266/kd-gat/dev/rf15/set_01/` without
  `.complete` markers. Need cleanup policy.

## Code Cleanup (from preprocessing + models consolidation)

- **`prepare_data()` / `setup(stage)` separation** — missing on all DataModules. `setup()`
  ignores `stage` param, loads everything unconditionally. Blocks DDP.
- **No `predict_dataloader()`** on any DataModule.

## Orchestration (from dagster rebuild)

- **Dagster testing layers 0-3 missing** — no unit tests for pure Python, dagster unit
  with mock SLURM, dagster integration with real IOManager.

## Observability (deferred from 2026-04-01)

- **`--watch` mode for `pipeline-status`** — auto-refresh like `watch squeue` but with
  DAG context.
- **Structured JSON logs** — `structlog.processors.JSONRenderer()` when `SLURM_JOB_ID`
  is set, enables `jq` parsing.

## Config (from codex-refactor audit)

- **`experimentruns` fallback** — `LAKE_ROOT` defaults to relative in-repo path when
  `KD_GAT_LAKE_ROOT` unset. `.env.example` should include it.

## Scripts (from scripts-refactor)

- **`smoke-test` subcommand** — not implemented. Dagster ablation on hcrl_sa serves as smoke test.
- **`run_tests.sh`** — not created, `scripts/submit.sh tests` handles it. Low priority alias.
