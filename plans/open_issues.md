## Problem statement

latest dagster job runs either failed or were cancelled due to poor checkpoint writing

this has raised a larger issue with the codebase, and discord is only larger due to claude limiations
(see my entire working history)

This makes code fixes difficult, as claude will inject malicious shortcuts that kill the viability
of the codebase working

## No structure enforcement

- Claude ignore implicit structure of codebase in favor of going rogue with quirk fixes that
  cause long-term damage to codebase

## No Documentation to enforce

- No github wiki to track the true state of the codebase
- Claude will likely ignore this, but will at least be a reference point and a proxy to measure
  drift

## Config issues

- There is terrible configuration setting up, tracking, and overriding
- many different configurations with no defined responsibilities
- complicated merge mechanics in CLI
- dagster or user overrides or ignores on ad-hoc basis, no policy

## Experiment Tracking

- no assurance that everything that needs to be written (slurm logs, metrics, yaml, model artifacts)
  is actually being logged.
- Training jobs ran a whole pipeline and didnt bother to actually save checkpoint or final model weights
- poor communication between lighting, wandb, and dagster. Zero awareness, everything is a black box
- lightning logs go by a version counter, dagster uses a unique path file with hash, wandb unsure

## No writing enforcement

- Claude ignores writing outputs to the share file location in favor of writing outputs to repo.
- This poisons code conventions, causing drift and untracked errors

## Half wired up metrics

- some wandb logs populate
- dagster UI broken
- No defined read and write roles
- database in share file is ad-hoc, buried code everywhere

## No Cleanup or re-reads

- Claude would rather write 100 plans and re-read none of them than referring to documentation
  or previously established plans
- This causes document slop to pile up, which further drowns out useful context
- claude HEAVY bias to write through issues than think through issues. Zero socratic questioning.
- Errors tend to have a common pattern, but for claude it is a constant stream of information with
  no pattern
- Have memory tool for claude to refer to, but it is fundamentally broken. Claude only writes
  memories never recalls memories. This makes it worse than nothing, as it only grabs the tool when
  caught not checking, and decides writing memories as a greedy fix instead of a true optimum.

## Deferred from preprocessing consolidation (2026-03-27)

- Delete `_temporal.py` (175 lines, zero production callers) + clean temporal entries from `__init__.py`
- Delete `edge_to_tensor` from `features.py` (zero callers in production or tests)
- Broken test imports in `test_features.py`: `edge_features`, `_assemble_chunk_numpy`, `_numpy_to_data` don't exist in `features.py`
- `SimpleNamespace` cfg reconstruction in `CANBusDataModule.setup` and `FusionDataModule.setup` — refactor `load_datasets` to keyword args
- Move `cache_predictions` staticmethod from `FusionDataModule` to `fusion_features.py` (model-layer logic in data-layer class)
- Generalize `atomic_write` in `utils.py` — `_write_cache_metadata` reimplements tmpfile→fsync→rename inline
- `prepare_data()` / `setup(stage)` separation missing on all 3 DataModules (blocks DDP)
- `setup()` ignores `stage` param — loads everything unconditionally
- No `predict_dataloader()` on any DataModule

## Deferred from observability (2026-04-01)

- `--mail-type=END,FAIL` in submit.sh — zero-effort job failure emails (skipped, user said "just fail")
- `--watch` mode for `pipeline-status` — auto-refresh like `watch squeue` but with DAG context
- Structured JSON logs — `structlog.processors.JSONRenderer()` when `SLURM_JOB_ID` is set, enables `jq` parsing
- `pipeline_status.json` — machine-readable status file for cron-based alerting
- dagster-slack sensor — requires `dagster-daemon run` as persistent process
- turm fork — sacct + GPU GRES in Rust TUI (deferred in favor of existing tools: SlurmTUI, nvitop, reportseff)

## Deferred from dagster orchestration (2026-03-29)

- `dagster-slurm>=1.12.0` in `pyproject.toml` is unused — zero imports. Remove.
- Dagster testing layers 0-3 have no test files (Layer 4 smoke exists). Need: pure Python unit tests, dagster unit with mock SLURM, dagster integration with fake SLURM + real IOManager, IOManager sidecar unit tests.

## Open from ablation runs 001/004 (2026-03-24 to 2026-03-30)

- GPS `batch_size` right-sizing: O(N^2) global attention OOMs on V100. Need GPS-specific cap (~256-384) or `attn_type="performer"`.
- Dataset-scoped data staging: `stage_data.sh` copies entire 86GB cache/ to TMPDIR. Should copy only needed dataset (4-6GB).
- Scratch cache cleanup: 64GB of stale versioned dirs (v3-v7) on scratch.
- KD wall time unverified: retry path (base 4h → 6h) exists but never tested for KD workloads.
- ESS stale run dirs: 11 dirs at `/fs/ess/PAS1266/kd-gat/dev/rf15/set_01/`, only 2 have `.complete` markers. Need cleanup policy.

## Deferred from performance analysis (2026-03-30)

- CurriculumDataModule rebuilds DataLoader every epoch → kills persistent workers → ~40 min spawn overhead over 300 epochs. Fix: create DataLoader once, update `CurriculumSampler.set_epoch()` only.
- PSS verification on GPU node: RSS double-counts shared mmap pages. Submit job with `smaps_rollup` in `worker_init_fn` to confirm. If confirmed, reduce `--mem` in resource profiles.
- VRAM probe validation: compare `probe_bytes_per_node` against `DeviceStatsMonitor` peak from Run 005. If >40% conservative, split `_GRAD_MULTIPLIER` for KD.

## Deferred from models consolidation (2026-03-30)

NOTE: SOME OF THESE HAVE BEEN LOOKED OVER, AGAIN POOR TRACKING

- VGAE `configure_optimizers` kept — projection param handling needs verification before switching to CLI-wired optimizer
- `lr` / `weight_decay` params in GAT/DGI `__init__` are dead code (saved to hparams but never read). Cleanup candidate.
- `T_max: 300` in stage YAMLs is static — old code used `self.trainer.max_epochs` dynamically. A `link_arguments` could wire this.
- No DGI stage YAML exists yet — DGI is placeholder in ablation recipe only.
- Full SLURM test run of consolidated models not yet executed.
