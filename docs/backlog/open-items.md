# Open Issues

> Audited: 2026-04-03. Resolved items removed (see git history).

## Performance

- **PSS verification on GPU node** — RSS double-counts shared mmap pages. Submit job with
  `smaps_rollup` in `worker_init_fn` to confirm. If confirmed, reduce `--mem` in profiles.
- **VRAM probe validation** — compare `probe_bytes_per_node` against `DeviceStatsMonitor`
  peak. If >40% conservative, split `_GRAD_MULTIPLIER` for KD.

## Ablation / Experiment

- **Scratch cache cleanup** — 64GB of stale versioned dirs (v3-v7) on scratch.
- **ESS stale run dirs** — dirs at `/fs/ess/PAS1266/kd-gat/dev/rf15/set_01/` without
  `.complete` markers. Need cleanup policy.

## Code Cleanup

- **`prepare_data()` / `setup(stage)` separation** — missing on all DataModules. `setup()`
  ignores `stage` param, loads everything unconditionally. Blocks DDP.
- **No `predict_dataloader()`** on any DataModule.

## Orchestration

- **Dagster testing layers 0-3 missing** — no unit tests for pure Python, dagster unit
  with mock SLURM, dagster integration with real IOManager.

## Config

- **`KD_GAT_LAKE_ROOT` missing from `.env.example`** — `LAKE_ROOT` defaults to relative
  in-repo path when unset. Add to `.env.example`.
