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

### Teacher VRAM — RESOLVED (2026-03-30)
Lightning auto-moved `self.teacher` to GPU via `nn.Module._modules`. Fix: teacher stored via
`self.__dict__["teacher"]` to bypass registration. `teacher_on_device()` (`_training.py:57-73`)
unconditionally moves teacher CPU<->GPU for inference. See `plans/memory-profiling/vram-probe-kd-aware.md`.

## Open Issues

### 3. Large GAT CUDA OOM on V100

`curriculum_cbe06f3a` (large_reference curriculum) hit `torch.OutOfMemoryError` during
sanity check val loop.

**Root cause:** `vram_node_budget()` (`datamodule.py:80`) is model-blind. Hardcoded
`_BYTES_PER_NODE = 32,768` calibrated on small-model conv passes. Ignores `hidden_channels`,
`num_layers`, JK-LSTM aggregation cost.

- Large GAT got same budget as small: 506K nodes
- JK-LSTM requested 24.67 GiB for workspace over 506K nodes (input `[3, 506632, 256]`)
- Correct budget: ~73 KiB/node → 221K nodes for large GAT on V100

**Fix needed:** `vram_node_budget()` must accept model architecture params.

### 4. KD autoencoder wall time (SIGUSR1 after 2h)

`autoencoder_8e6b9f70_kd` hit 2h wall still in epoch 1. Three compounding factors:
1. set_01 is 26x larger than hcrl_sa
2. ~~Teacher VRAM~~ (RESOLVED — see above)
3. 2h wall insufficient — non-KD large barely fit 2h (epoch 157/299). KD needs 6-8h.

**Fix needed:** KD variants need separate resource profiles (`--time=06:00:00` minimum).

### 5. profile_jobs.py broken for dagster pipeline

`scripts/profile_jobs.py` returned empty data. Three broken assumptions:
1. RSS=0.0G — OOM-killed jobs don't flush sacct batch records
2. No `gpu_stats.csv` — `_epilog.sh` doesn't write it (relies on Lightning DeviceStatsMonitor)
3. Log path mismatch — profiler expects `slurm_logs/{jid}/{jid}_0_log.out`, dagster produces
   `slurm_logs/{jobname}_{jid}.out`

**Fix needed:** Update profiler for dagster log layout.

### 6. Observability Gaps

**Dagster side:** `poll()` (`slurm.py:79`) queries sacct every 60s in silent loop. No
intermediate events. Asset shows "materializing" for entire duration.

**SLURM job side:** Generated script (`slurm.py:38-43`) is preamble + training + epilog.
No background GPU monitoring.

**Lightning:** `trainer.yaml` has WandbLogger + CSVLogger configured (lines 9-16).
DeviceStatsMonitor is a callback (line 23). However, training progress is only visible as
tqdm in raw SLURM `.out` files during the run — no real-time dagster integration.

**profile_jobs.py:** 4 of 5 data sources broken (see issue 5). Only `sacct` works.

**Net result:** Zero observability beyond `squeue` without manually tailing SLURM `.out` files.

#### Dagster improvements (prioritized)

**Config-only (no code):**
1. `python_logs` file handler in `dagster.yaml` — would have caught the TypeError traceback
2. `dagster debug export <run_id>` — already available, never used

**Small code changes:**
3. `AssetObservation` in poll loop — emit SLURM state transitions (~10 lines)
4. `context.add_output_metadata()` — attach job_id, wall time, peak RSS (~5 lines)

**Significant effort, highest value:**
5. **Dagster Pipes** — training script reports epoch/loss/GPU metrics back to dagster event log
   via `dagster-pipes` package (no torch dependency). Single biggest observability gap.

### 7. Dagster Testing Strategy

**Layer 0 — Pure Python (login node):** Test `compute_identity_hash()`, `run_dir()`, config
resolution, CLI arg building.

**Layer 1 — Dagster unit (login node):** Direct-invoke assets with
`mock.Mock(spec=SlurmTrainingResource)`. Verify CLI commands, partition parsing, config selection.

**Layer 2 — Dagster integration (login node):** `materialize_to_memory()` with fake SLURM
resource + real `CheckpointPathIOManager(base_dir=tmp_path)`. Tests cross-stage checkpoint handoff.

**Layer 3 — IOManager unit (login node):** `build_output_context()` / `build_input_context()`
for sidecar read/write.

**Layer 4 — Smoke (SLURM gpudebug):** `execute_in_process()` with real SLURM resource.
Already exists as `python -m graphids.orchestrate smoke`.

Key: `execute_in_process()` forces serial (ignores executor config). `materialize_to_memory()`
auto-uses `mem_io_manager()`. Both are test-only.

## ESS Disk State

Data at `/fs/ess/PAS1266/kd-gat/dev/rf15/set_01/`: 11 run dirs, most have
`config.yaml` + `checkpoints/best_model.ckpt`. Only 2 have `.complete` markers.
IOManager sidecars only for 2 completed assets.
