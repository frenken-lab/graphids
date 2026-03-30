# Ablation Run 004 — Failure Log

**Date:** 2026-03-30
**Branch:** dagster
**Recipe:** ablation.yaml (18 configs, set_01/set_02, seed 42)

## Run History

| Job | Run | State | Duration |
|-----|-----|-------|----------|
| 46152801 | dagster orchestrator (run 1) | CANCELLED | 2h10m |
| 46156620 | dagster orchestrator (run 2) | FAILED | 3h17m |

## Resolved Issues

### 1. SLURM RAM OOM (6 jobs, run 1)
All `vgae/small/autoencoder` and `gat/small/normal` jobs OOM'd at 24G on `set_01`.
**Fix:** Bumped resources.yaml to flat 36G / 4 CPUs for small/medium vgae, gat, dgi.

### 2. Dagster subprocess crash (6 steps, run 2)
`context.log.warning("stale_checkpoint", path=..., reason=...)` passed structlog-style kwargs to dagster's `DagsterLogManager`, which inherits Python's `logging.Logger._log()` and only accepts `exc_info`, `extra`, `stack_info`, `stacklevel`. Raised `TypeError` — crashed subprocess before dagster could emit STEP_FAILURE event, causing silent cascading failures.
**Fix:** Switched to f-string: `context.log.warning(f"Stale checkpoint ...: {ckpt_file}")`.

## Open Issues

### 3. Large GAT CUDA OOM on V100

`curriculum_cbe06f3a` (large_reference curriculum, job 46156626) hit `torch.OutOfMemoryError` during **sanity check val loop** (before epoch 1).

**Root cause:** `vram_node_budget()` (`datamodule.py:28-59`) is model-blind. It uses a hardcoded `_BYTES_PER_NODE = 32,768` calibrated on small-model conv passes. It takes `conv_type` and `heads` but ignores `hidden_channels`, `num_layers`, and JK-LSTM aggregation cost.

**Evidence:**
- Large GAT got same budget as small: **506,632 nodes** (`slurm_logs/...46156626.out:13`, `free_vram_gb=16.6`)
- Conv passes consumed 12.50 GiB, leaving 2.83 GiB free
- JK-LSTM requested **24.67 GiB** for workspace over 506K nodes: input shape `[3, 506632, 256]` (`gat.py:167 → jumping_knowledge.py:91 → rnn.py:1124`)
- Large GAT overlay: `hidden=64, heads=4` → conv output = 256 channels (`overlays/large_gat.yaml:6-8`)
- JK-LSTM constructed with `channels=hidden_channels * heads = 256` (`gat.py:81-83`)

**Correct budget:** ~73 KiB/node (vs 32 KiB constant) → **221K nodes** for large GAT on V100.

**Fix needed:** `vram_node_budget()` must accept model architecture params (at minimum `hidden_channels`, `num_layers`, `jk_mode`) or use a per-model-type constant.

### 4. KD autoencoder wall time (SIGUSR1 after 2h)

`autoencoder_8e6b9f70_kd` (job 46156625) hit the 2h wall limit still in epoch 1.

**Three compounding factors:**

1. **set_01 is 26x larger than hcrl_sa** (4.4 GB cache vs 171 MB, 3.4x hcrl_ch)
2. **~~Teacher consumes ~11 GiB VRAM on GPU~~** (**RESOLVED** — `dagster` branch, 2026-03-30).
   Teacher stored via `self.__dict__["teacher"]` to bypass `nn.Module._modules` registration.
   `teacher_on_device()` now unconditionally moves teacher CPU→GPU for inference, then back.
   Budget should recover to ~506K nodes (teacher weights ~3 MB, activation memory freed after each step).
   VRAM probe also updated: runs `model._step()` instead of `forward()` to capture teacher footprint
   during budget estimation. See `plans/memory-profiling/vram-probe-kd-aware.md`.
3. **2h wall time insufficient.** Non-KD large autoencoder (bf355e79, 745K params) barely fit 2h (epoch 157/299 on set_01). KD small with 3x batch reduction + teacher overhead needs **6-8h**.

**Fix needed:** KD variants need separate resource profiles (time ≥ 6h). ~~Also investigate teacher VRAM~~ (resolved).

### 5. profile_jobs.py broken for dagster pipeline

The profiler returned empty data for all jobs. Three broken assumptions:

1. **RSS = 0.0G** — OOM-killed jobs don't flush sacct `.0` batch records
2. **No gpu_stats.csv** — `_epilog.sh` doesn't write it; profiler expects `slurm_logs/{jid}/gpu_stats.csv` which never exists
3. **No metadata** — profiler looks for structlog events in `slurm_logs/{jid}/{jid}_0_log.out` but dagster SLURM logs are flat: `slurm_logs/{jobname}_{jid}.out`

**Fix needed:** Either update `_epilog.sh` to write `gpu_stats.csv` and update profiler path expectations, or rewrite profiler for dagster log layout.

## ESS Disk State

Data IS being written to `/fs/ess/PAS1266/kd-gat/dev/rf15/set_01/`:
- 11 run dirs exist
- Most have `config.yaml` + `checkpoints/best_model.ckpt`
- Only 2 have `.complete` markers (8b158266 conv_gps, bf355e79 large_reference)
- IOManager sidecars at `/fs/ess/PAS1266/kd-gat/.dagster/io/{asset}/set_01|42.json` — only for 2 completed assets

## 6. Zero Observability During Jobs

**Dagster side:** `poll()` (`slurm.py:78`) queries `sacct -j {id} --format=JobID,State` every 60s in a silent sleep loop. No intermediate events logged. Dagster asset shows "materializing" for entire duration with nothing in between.

**SLURM job side:** Generated script is 4 lines (`slurm.py:38-43`): preamble → `python -m graphids fit` → epilog. No background GPU monitoring. `_preamble.sh` claims "Lightning handles DeviceStatsMonitor" (false). `_epilog.sh` prints sacct summary, no GPU stats.

**Lightning:** `trainer.yaml:9` sets `logger: false` — no CSVLogger, no metrics.csv, no loss curves. `DeviceStatsMonitor` not in callbacks list. Even if it were, no logger to write to. Training progress only visible as tqdm in raw SLURM `.out` file.

**profile_jobs.py broken for dagster:** 4 of 5 data sources don't exist:

| Expected | Source | Status |
|----------|--------|--------|
| sacct stats | `sacct` query | **Works** |
| `gpu_stats.csv` at `slurm_logs/{jid}/` | nvidia-smi polling | **Nothing writes this** |
| structlog `training_complete` event | Training code | **No code emits this** |
| `metrics.csv` + DeviceStatsMonitor | Lightning CSVLogger | **`logger: false`** |
| `--since` job discovery | Filters on `"submitit"` | **Dagster names don't match** |

**Net result:** Running ablation has zero observability beyond `squeue` showing RUNNING. No epoch progress, GPU util, loss curves, or memory pressure without manually `tail`-ing SLURM `.out` files.

### Dagster capabilities we're not using

**Sources:** dagster docs (https://docs.dagster.io/guides/log-debug/logging/python-logging, https://docs.dagster.io/deployment/oss/dagster-yaml, https://docs.dagster.io/guides/operate/run-executors), context7 dagster docs, `dagster instance info` on our install.

**Instance state:** `DAGSTER_HOME=/fs/scratch/PAS1266/dagster` already persists run history in SQLite (5 historical runs verified). Compute logs configured at `/fs/scratch/PAS1266/dagster/compute_logs/`. But no Python logging config — STEP_FAILURE tracebacks lost when subprocesses crash.

#### Easy wins (config-only, no code changes)

1. **`python_logs` file handler in dagster.yaml** — captures all dagster events + managed loggers to a file. Would have caught the TypeError traceback that silently killed 6 subprocesses.
   ```yaml
   python_logs:
     python_log_level: DEBUG
     managed_python_loggers: [structlog]
     dagster_handler_config:
       handlers:
         fileHandler:
           class: logging.FileHandler
           level: INFO
           filename: /fs/scratch/PAS1266/dagster/logs/dagster_orchestrator.log
           mode: a
   ```
   Source: https://docs.dagster.io/guides/log-debug/logging/python-logging

2. **`dagster debug export <run_id>`** — dumps failed run for offline inspection. Already available, never used.

#### Small code changes

3. **`context.log_event(AssetObservation(...))`** in poll loop — emit SLURM state transitions (PENDING→RUNNING→COMPLETED) as dagster events with metadata (job_id, elapsed time). ~10 lines in `poll()`.
   Source: https://docs.dagster.io/guides/build/ops/op-events

4. **`context.add_output_metadata()`** in asset body — attach SLURM job_id, wall time, peak RSS to materialization event. ~5 lines per asset.

#### Significant effort, highest value

5. **Dagster Pipes** — lightweight `dagster-pipes` package (no torch dependency) installed in training venv. Training script calls `open_dagster_pipes()` to report epoch/loss/GPU metrics back to dagster event log. This is the **single biggest observability gap** — without it, dagster has zero visibility into what the training job is doing.
   ```python
   # In training script (no dagster import needed):
   from dagster_pipes import open_dagster_pipes
   with open_dagster_pipes() as ctx:
       ctx.report_asset_materialization(metadata={"epoch": 5, "loss": 0.03})
   ```
   Source: https://docs.dagster.io/guides/build/external-pipelines/using-dagster-pipes

#### Not applicable to our setup

- `@run_failure_sensor` / `@run_status_sensor` — require persistent `dagster-daemon` (we run ephemeral SLURM jobs)
- `dagster-slurm` community package — designed for SSH-based remote submission, we're already on-cluster
- Dask/K8s/Celery executors — overkill, our custom sbatch+poll is simpler for our use case

#### Executor note

We use multiprocess executor (each dagster step = subprocess). Supports `max_concurrent` for parallelism. **Caution:** `execute_in_process()` ignores executor config and forces serial — verify `__main__.py` uses `dg.materialize()` for real runs.
Source: https://docs.dagster.io/guides/operate/run-executors

### What would have saved us in Run 004

| Fix | Would have caught |
|-----|-------------------|
| `python_logs` file handler | TypeError traceback from `context.log.warning()` — the 6 silently crashing subprocesses |
| `compute_logs` per-step capture | OOM tracebacks, training progress for all jobs |
| `AssetObservation` in poll loop | Which SLURM jobs were PENDING vs RUNNING, how long in queue |
| Dagster Pipes | Epoch progress, batch budget collapse (506K→168K), teacher VRAM consumption — all mid-job |

### Teacher VRAM investigation — RESOLVED (2026-03-30)

**Root cause:** Lightning auto-moved `self.teacher` to GPU via `nn.Module._modules` registration.

**Fix applied:** Option 1 — teacher stored via `self.__dict__["teacher"]` (bypasses `_modules`).
`teacher_on_device()` rewritten: unconditionally moves teacher CPU→GPU for inference, CPU after.
Deleted broken `offload_teacher_to_cpu` flag and `_teacher_on_cpu` state. Files changed:
`_training.py:57-73`, `vgae.py:378-405`, `gat.py:231-260`.

**Remaining concern:** KD resource profiles still need `--time=06:00:00` minimum (factor 2 above).
Per-step CPU↔GPU transfer adds ~0.5 ms (3 MB weights over PCIe 3.0) — negligible.

## 7. Dagster Testing Strategy

**Sources:** https://docs.dagster.io/guides/test, https://docs.dagster.io/guides/test/unit-testing-assets-and-ops, https://docs.dagster.io/guides/build/external-resources/testing-configurable-resources, https://docs.dagster.io/api/dagster/execution

### Key dagster test utilities

| Utility | Purpose |
|---------|---------|
| Direct function call | Unit test asset logic — `@asset` preserves callability |
| `materialize_to_memory()` | In-process execution with in-memory IO, no disk writes |
| `execute_in_process()` | Full job test, replaces executor with in-process, returns `ExecuteInProcessResult` |
| `build_asset_context(partition_key=...)` | Create context for testing partitioned assets |
| `build_output_context()` / `build_input_context()` | Test custom IOManager methods directly |
| `mock.Mock(spec=SlurmTrainingResource)` | Mock resource for unit tests |
| `validate_run_config(job, config)` | Validate config without executing |
| `instance_for_test()` | Ephemeral persistent instance for integration tests |

### Recommended testing layers for KD-GAT

**Layer 0 — Pure Python (no dagster, login node):**
Test `compute_identity_hash()`, `run_dir()`, config resolution, `enumerate_assets()`, CLI arg building. These are pure functions.

**Layer 1 — Dagster unit (login node):**
Direct-invoke asset functions with `mock.Mock(spec=SlurmTrainingResource)`. Verify CLI command strings, partition key parsing, config overlay selection. Pattern:
```python
mock_slurm = mock.Mock(spec=SlurmTrainingResource)
mock_slurm.submit_and_wait.return_value = "COMPLETED"
result = my_asset(context, slurm=mock_slurm)
assert "--config stages/autoencoder.yaml" in mock_slurm.submit_and_wait.call_args[...]
```

**Layer 2 — Dagster integration (login node):**
`materialize_to_memory()` with `FakeSlurmResource` + real `CheckpointPathIOManager(base_dir=tmp_path)`. Tests cross-stage checkpoint handoff and the full asset graph without hitting SLURM.

**Layer 3 — IOManager unit (login node):**
`build_output_context()` / `build_input_context()` to test `CheckpointPathIOManager` sidecar read/write in isolation.

**Layer 4 — Smoke (SLURM gpudebug):**
`execute_in_process()` with real `SlurmTrainingResource` on `gpudebug` partition, tiny configs. Already exists as `python -m graphids.orchestrate smoke`.

### Key findings from docs

- `execute_in_process()` **ignores the configured executor** and forces serial in-process — fine for testing but must not be used for real runs (source: https://docs.dagster.io/api/dagster/execution)
- `materialize_to_memory()` auto-uses `mem_io_manager()` — no disk IO manager setup needed (source: https://docs.dagster.io/api/dagster/execution)
- Dagster docs explicitly note unit testing isn't ideal when "most business logic is in an external system" — matches our SLURM pattern. Mock the resource, test the orchestration logic. (source: https://docs.dagster.io/guides/test/unit-testing-assets-and-ops)
- No built-in "dry run" mode, but `validate_run_config()` + `execute_in_process()` with mocks serves the same purpose
- Environment-aware resource swapping is the recommended pattern for test vs prod (source: https://docs.dagster.io/deployment/dagster-plus/ci-cd/branch-deployments/testing)
