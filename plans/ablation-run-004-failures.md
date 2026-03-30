# Ablation Run 004 — Failure Log

> Date: 2026-03-30 | Branch: dagster | Audited: 2026-03-30
> Recipe: `config/recipes/ablation.yaml` (18 configs, set_01/set_02, seed 42)

## Run History

| Job | Run | State | Duration |
|-----|-----|-------|----------|
| 46152801 | dagster orchestrator (run 1) | CANCELLED | 2h10m |
| 46156620 | dagster orchestrator (run 2) | FAILED | 3h17m |

## Resolved Issues

### 1. SLURM RAM OOM (6 jobs, run 1)
All `vgae/small/autoencoder` and `gat/small/normal` jobs OOM'd at 24G on `set_01`.
**Fix:** Bumped `resources.yaml` to flat 36G / 4 CPUs for small/medium vgae, gat, dgi.

### 2. Dagster subprocess crash (6 steps, run 2)
`context.log.warning("stale_checkpoint", path=..., reason=...)` passed structlog-style kwargs
to dagster's `DagsterLogManager` (inherits `logging.Logger` — only accepts `exc_info`, `extra`,
`stack_info`, `stacklevel`). TypeError crashed subprocesses before dagster could emit
STEP_FAILURE events, causing silent cascading failures.
**Fix:** Switched to f-string: `context.log.warning(f"Stale checkpoint ...: {ckpt_file}")`.

### 3. Large GAT CUDA OOM on V100 — RESOLVED
`curriculum_cbe06f3a` (large_reference curriculum) hit `torch.OutOfMemoryError` during
sanity check val loop. Root cause: `vram_node_budget()` was model-blind (hardcoded
`_BYTES_PER_NODE = 32768`).

**Fix:** `vram_node_budget()` (`preprocessing/datamodule.py:80-128`) now accepts `model` and
`train_dataset` params. When both are available + CUDA, calls `_probe_bytes_per_node()`
(`:36-77`) which runs one `_step()` call, measures `max_memory_allocated()`, derives real
bytes/node. The `_BYTES_PER_NODE` constant is now a CPU/no-model fallback only (`:120`).
KD-aware: probe uses `model._step` (captures teacher inference cost) when available (`:117`).

### Teacher VRAM — RESOLVED
Lightning auto-moved `self.teacher` to GPU via `nn.Module._modules`. Fix: teacher stored via
`self.__dict__["teacher"]` to bypass registration. `teacher_on_device()` (`_training.py:57-73`)
unconditionally moves teacher CPU<->GPU for inference. See `plans/memory-profiling/vram-probe-kd-aware.md`.

### 5. profile_jobs.py broken for dagster pipeline — RESOLVED
`scripts/profile_jobs.py` replaced by `graphids/orchestrate/profiler.py` (252 lines). Fixes:
1. RSS=0.0G — sacct now queries `.batch` step, not `.0` (`profiler.py:120-121`)
2. `gpu_stats.csv` dependency removed — GPU metrics come from wandb/DeviceStatsMonitor
3. Log path mismatch — `_JOB_NAME_RE` (`profiler.py:160-169`) parses dagster naming format
   (`stage_identityhash_dataset_sseed`)

Verified: `python -m graphids profile 46152810 46152812` returns correct RSS=48G/24G, CPU%=28%/23%.

## Open Issues

### 4. KD autoencoder wall time (partially resolved)

`autoencoder_8e6b9f70_kd` hit 2h wall still in epoch 1. Three compounding factors:
1. set_01 is 26x larger than hcrl_sa
2. ~~Teacher VRAM~~ (RESOLVED)
3. 2h wall insufficient — non-KD large barely fit 2h (epoch 157/299). KD needs 6-8h.

**Current state:** `resources.yaml:196-204` has `TIMEOUT: scale_time: 1.5, max_retries: 1`
(base 4h → retry at 6h). No separate KD resource profiles exist — KD runs share base
profiles and rely on the TIMEOUT retry path for extra time. This may be sufficient given
Run 004 sacct data (large VGAE completed in 1h58m of 4h), but unverified for KD workloads.

### 6. Dagster-side observability (partially resolved)

**Training-side: DONE.** wandb + CSVLogger + DeviceStatsMonitor + sacct profiler all wired.
See `plans/research/profiling-and-observability.md` for full stack.

**Dagster-side: mostly done.**

| Item | Status | Where |
|------|--------|-------|
| `dagster.yaml` with `python_logs` file handler | Done | `/fs/scratch/PAS1266/dagster/dagster.yaml` — captures `dagster` + `graphids.orchestrate` loggers to `logs/python.log` |
| `context.add_output_metadata()` — job_id, wall time, RSS | Done | `component.py:381-395` — parses sacct parent row (Elapsed) + `.batch` row (MaxRSS) |
| `AssetObservation` in poll loop — SLURM state transitions | Done | `component.py:357-361` callback → `slurm.py:92` `on_state` param. slurm.py stays dagster-free. |
| **Dagster Pipes** — epoch/loss/GPU from training script | Not done | Significant effort. wandb already covers same metrics. |

### 7. Dagster testing strategy (mostly not done)

**Layer 4 (smoke) exists:** `python -m graphids.orchestrate smoke` (`orchestrate/__main__.py:91-176`).
Verified on gpudebug (3-stage chain, hcrl_sa, 3 epochs) during dagster rebuild.

**Layers 0-3 have no test source files.** Pycache artifacts (`test_pipes_slurm`, `test_slurm_primitives`)
suggest tests existed but were deleted. Layers needed:

- **Layer 0 — Pure Python (login node):** `compute_identity_hash()`, `run_dir()`, config resolution, CLI arg building
- **Layer 1 — Dagster unit (login node):** Direct-invoke assets with `mock.Mock(spec=SlurmTrainingResource)`
- **Layer 2 — Dagster integration (login node):** `materialize_to_memory()` with fake SLURM + real IOManager
- **Layer 3 — IOManager unit (login node):** `build_output_context()` / `build_input_context()` sidecar read/write

## ESS Disk State

Data at `/fs/ess/PAS1266/kd-gat/dev/rf15/set_01/`: 11 run dirs, most have
`config.yaml` + `checkpoints/best_model.ckpt`. Only 2 have `.complete` markers.
IOManager sidecars only for 2 completed assets.
