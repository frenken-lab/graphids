# Plan: Refactor `scripts/` to Option C — Thin Shells, Python Does the Work

> Status: **ready** | Created: 2026-03-31

## Context

The `scripts/` directory has 4 stale scripts using the pre-flatten CLI (`stage=autoencoder model_type=vgae`), 1 script calling a nonexistent Python file, 1 disposable single-use script, ghost `.pyc` files, and a ghost reference to `run_tests_slurm.sh` (doesn't exist, referenced in 4 docs). The root cause: shell scripts encode CLI invocations that drift when Python code changes.

Option C target: shell scripts should only be `source _preamble.sh && python -m graphids <subcommand>`. Python CLI is the single source of truth for how to invoke itself.

## Changes

### 1. Delete dead files

| File | Reason |
|------|--------|
| `scripts/slurm/_wait_and_resubmit.sh` | Hardcoded Run 004 job IDs. Single-use, already executed. |
| `scripts/slurm/profile_test.sh` | Subsumed by `profile_training.sh` with env vars. Uses old CLI. |
| `scripts/slurm/invalidate_cache_marker.sh` | Absorbed into `rebuild-caches` subcommand. |
| `scripts/data/__pycache__/` | Ghost `.pyc` for deleted `export_paper_data.py` + `paper_sync.py` |

### 2. Add `rebuild-caches` subcommand to Python CLI

**Why**: `rebuild_caches.sbatch` has 35 lines of inline Python using deleted `resolve()` / `from_cfg()` APIs.

**New file**: `graphids/orchestrate/rebuild_caches.py` (~40 lines)
- Accepts `--dataset` (one or more), `--delete-existing`
- Instantiates `CANBusDataModule` with flat kwargs, calls `.setup("fit")`
- Invalidates scratch staging marker after rebuild (absorbs `invalidate_cache_marker.sh`)

**Dispatch**: Add `elif cmd == "rebuild-caches":` to `graphids/__main__.py`

**Shell wrapper**: Rewrite `rebuild_caches.sbatch` to ~15 lines (was 70). Mem bumped to 128G for set_03/set_04.

### 3. Add `smoke-test` subcommand to Python CLI

**Why**: `test_pipeline_stages.sbatch` uses old CLI, wrong checkpoint paths (`.pt` not `.ckpt`), wrong path structure. Dagster `smoke` is documented but not implemented.

**New file**: `graphids/orchestrate/smoke.py` (~50 lines)
- Runs 3 stages sequentially via `subprocess.run(["python", "-m", "graphids", "fit", "--config", ...])`
- Checks checkpoint exists between stages
- Tiny architectures, 2 epochs, hcrl_ch

**Shell wrapper**: Rewrite `test_pipeline_stages.sbatch` to ~10 lines (was 85).

### 4. Fix `test_preprocessing.sbatch`

Rewrite inline Python to use current `CANBusDataModule` constructor with flat kwargs. Small enough that a subcommand isn't warranted.

### 5. Rewrite `loss_landscape.sbatch`

Calls nonexistent `scripts/profiling/loss_landscape.py`, uses `--partition=serial` (invalid). `Analyzer` class already supports `landscape=True` + `landscape_resolution`. Rewrite to use `python -m graphids analyze --analyzer.landscape=true`.

### 6. Create `run_tests.sh`

4 files reference `scripts/slurm/run_tests_slurm.sh` but it doesn't exist. Create `run_tests.sh` (~8 lines), update references.

## Files to modify

| File | Action |
|------|--------|
| `graphids/__main__.py` | Add `rebuild-caches` and `smoke-test` dispatch |
| `graphids/orchestrate/rebuild_caches.py` | **NEW** — ~40 lines |
| `graphids/orchestrate/smoke.py` | **NEW** — ~50 lines |
| `scripts/slurm/rebuild_caches.sbatch` | Rewrite to thin shell |
| `scripts/slurm/test_pipeline_stages.sbatch` | Rewrite to thin shell |
| `scripts/slurm/test_preprocessing.sbatch` | Fix inline Python |
| `scripts/slurm/loss_landscape.sbatch` | Rewrite to use Analyzer |
| `scripts/slurm/run_tests.sh` | **NEW** — ~8 lines |
| `scripts/slurm/_wait_and_resubmit.sh` | **DELETE** |
| `scripts/slurm/profile_test.sh` | **DELETE** |
| `scripts/slurm/invalidate_cache_marker.sh` | **DELETE** |
| `scripts/data/__pycache__/` | **DELETE** |
| `.claude/rules/slurm-hpc.md` | Update `run_tests_slurm.sh` → `run_tests.sh` |
| `.claude/rules/critical-constraints.md` | Update `run_tests_slurm.sh` → `run_tests.sh` |
| `plans/architecture/write-paths.md` | Update reference |
| `plans/memory-profiling/vram-probe-kd-aware.md` | Update reference |

## Execution order

1. Delete dead files (3 scripts + __pycache__)
2. Create `rebuild_caches.py` + wire in `__main__.py` + rewrite `.sbatch`
3. Create `smoke.py` + wire in `__main__.py` + rewrite `test_pipeline_stages.sbatch`
4. Fix `test_preprocessing.sbatch` inline Python
5. Rewrite `loss_landscape.sbatch` to use Analyzer
6. Create `run_tests.sh` + update 4 doc references
7. Verify on login node

## Verification

- `python -m graphids rebuild-caches --help` (login node safe)
- `python -m graphids smoke-test --help` (login node safe)
- `python -c "from graphids.orchestrate.rebuild_caches import rebuild_caches; print('OK')"`
- `python -m graphids rebuild-caches --dataset hcrl_ch` (tiny dataset, ~2 min)
- `grep -r 'run_tests_slurm' .` returns 0 matches after reference updates
