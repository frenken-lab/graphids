# Profiling, Optimization, and Observability — Consolidated Plan

> Date: 2026-03-30
> Context: KD-GAT Run 004 failures (VRAM OOM, zero observability), dagster branch
> Environment: OSC Pitzer, V100 (16 GB, compute capability 7.0), CUDA 12.6, PyTorch 2.8, PyG 2.7, SLURM

## Goals

1. **Batch size**: generalize VRAM budget calculation across model × scale × GPU combos
2. **Training speed**: identify and remove bottlenecks (CPU↔GPU, data loading, kernel efficiency)
3. **Observability**: per-step GPU metrics, memory telemetry, throughput, experiment tracking

## Priority Actions

| Pri | Action | Effort | Impact | Status |
|-----|--------|--------|--------|--------|
| P0 | Enable CSVLogger + WandbLogger in `trainer.yaml` | 30 min | Solves issue #5 (zero observability). GPU util/temp/mem/power every 15s. | **Done** (2026-03-30). wandb 0.25.1 installed, `trainer.yaml` updated. `orchestrate validate` passes. |
| P0 | Add DeviceStatsMonitor callback | 1 line YAML | Per-step CUDA allocator stats. Catches OOM precursors. Requires logger. | **Done** (2026-03-30). Added to `trainer.yaml` callbacks. |
| P1 | Dagster UI via SSH tunnel | 30 min | Asset catalog, run history, Gantt charts, log tailing for orchestration layer. | Pending |
| P1 | Run 1 nsys profiling job | 1 SLURM job | System-wide CPU↔GPU timeline. Answers "is training data-bound or compute-bound?" | Pending |
| P1 | Run 1 memory snapshot job | 1 SLURM job | Per-allocation trace with stack traces. Diagnoses 13G vs 22G bimodal worker memory. | Pending |
| P2 | ThroughputMonitor callback | ~20 LOC | Samples/sec + batch latency for SLURM resource right-sizing (54G requested vs 23G peak). | Pending |
| P2 | Benchmark `torch.compile` on 1 VGAE job | 1 SLURM job | PyG 2.5+ claims up to 300% speedup. GATv2Conv should work with `dynamic=True`. | Pending |
| P3 | Feed sacct output into DuckDB | Script | Cross-job resource analysis for right-sizing. | Deferred |

---

## Proposed Logging Inventory

Everything that would be logged under the new plan, who produces it, where it lands, and risks.

### Per-Step Training Metrics (inside SLURM job, via Lightning `self.log()`)

| Metric | Producer | Models | Destination | Risk |
|--------|----------|--------|-------------|------|
| `train_loss` | `self.log()` in training_step | All 6 (VGAE, GAT, DGI, MLP, WeightedAvg, RL) | wandb + CSV | None — universal |
| `val_loss` | `self.log()` in validation_step | All 6 | wandb + CSV | None — universal |
| `train_acc` | `self.log()` in training_step | GAT, Temporal, MLP, WeightedAvg, RL | wandb + CSV | **Missing from VGAE, DGI** (unsupervised — no accuracy metric). Not a bug, but ablation comparison must account for it. |
| `val_acc` | `self.log()` in validation_step | GAT, Temporal, MLP, WeightedAvg, RL | wandb + CSV | Same gap as train_acc |
| `alpha` | `self.log()` in training_step | WeightedAvg only | wandb + CSV | Fusion-specific. Only 1 of 18 configs. |
| `avg_reward`, `accuracy` (RL) | `self.log()` in training_step | RLFusionModule only | wandb + CSV | RL-specific dynamic keys from `_derive_scores()`. Other fusion methods don't log these. |
| Test metrics (AUROC, F1, etc.) | `self.log_dict(test_metrics.compute())` in on_test_epoch_end | All 6 | wandb + CSV | Only at test time, not during training. |
| `threshold` | `self.log_dict(metrics)` in on_test_epoch_end | VGAE, DGI | wandb + CSV | Anomaly detection threshold — not in supervised models. |

### Per-Step GPU / System Metrics (inside SLURM job, automatic)

| Metric | Producer | Frequency | Destination | Risk |
|--------|----------|-----------|-------------|------|
| `allocated_bytes.all.peak` | DeviceStatsMonitor callback | Every step | wandb + CSV | **Not yet in trainer.yaml.** Requires logger ≠ false. |
| `reserved_bytes.all.peak` | DeviceStatsMonitor callback | Every step | wandb + CSV | Same — dead until callback added. |
| `num_alloc_retries` | DeviceStatsMonitor callback | Every step | wandb + CSV | Leading OOM indicator. Same dependency. |
| `num_ooms` | DeviceStatsMonitor callback | Every step | wandb + CSV | Same. |
| `inactive_split_bytes` | DeviceStatsMonitor callback | Every step | wandb + CSV | Fragmentation indicator. Same. |
| GPU utilization % | wandb system metrics (pynvml) | 15s | wandb only | **Not in CSV.** wandb-only metric. If wandb fails, this data is lost. |
| GPU temperature (°C) | wandb system metrics (pynvml) | 15s | wandb only | Same — wandb-only. |
| GPU power (W) | wandb system metrics (pynvml) | 15s | wandb only | Same. |
| GPU memory used | wandb system metrics (pynvml) | 15s | wandb only | Same. Overlaps with DeviceStatsMonitor but measured differently (driver vs allocator). |
| CPU %, RSS, disk I/O | wandb system metrics | 15s | wandb only | Same — no CSV fallback. |

### Per-Step Custom Metrics (proposed callbacks, inside SLURM job)

| Metric | Producer | Frequency | Destination | Risk |
|--------|----------|-----------|-------------|------|
| `throughput/samples_per_sec` | ThroughputMonitor callback | Every step | wandb + CSV | **Not yet written.** ~20 LOC. Needs `torch.cuda.synchronize()` — adds ~4μs overhead per step. |
| `throughput/batch_ms` | ThroughputMonitor callback | Every step | wandb + CSV | Same. |

### Per-Job Metadata (inside SLURM job, automatic)

| Metric | Producer | Destination | Risk |
|--------|----------|-------------|------|
| Hyperparameters | `save_hyperparameters()` + WandbSaveConfigCallback | wandb config tab | **Callback not yet written** (~10 LOC). Without it, full jsonargparse config not forwarded to wandb (Lightning #19728). `save_hyperparameters()` alone logs flat init_args but misses trainer/data config. |
| Git commit + diff | wandb auto-capture | wandb | Set `WANDB_DISABLE_GIT=true` on NFS for perf. Acceptable to skip. |
| Environment (hardware, OS, Python) | wandb auto-capture | wandb | None. |
| stdout/stderr | wandb auto-capture + SLURM log files | wandb + ESS `slurm_logs/` | Redundant (both capture). Good — two copies. Logs write to `$KD_GAT_SLURM_LOG_DIR` (default: `/fs/ess/PAS1266/kd-gat/slurm_logs/`). |

### Per-Job Post-Hoc (after SLURM job completes)

| Metric | Producer | Destination | Risk |
|--------|----------|-------------|------|
| Elapsed, MaxRSS, MaxVMSize | `sacct` in `_epilog.sh` | stdout → ESS log file | Captured in SLURM stdout log on ESS. No structured persistence (DuckDB) yet. |
| Checkpoint file | ModelCheckpoint callback | `{run_dir}/checkpoints/best_model.ckpt` | None — already works. |
| `config.yaml` (expanded) | SaveConfigCallback (Lightning default) | `{run_dir}/config.yaml` | Already works via `overwrite: True` in CLI_KWARGS. |

### structlog Events (inside SLURM job, application-level)

| Event | Producer | Destination | Risk |
|-------|----------|-------------|------|
| `probe_bytes_per_node` | `_probe_bytes_per_node()` in datamodule.py:70 | stdout (structlog console) | Captured in SLURM log file + wandb stdout. `profile_jobs.py` parses this from log files. |
| `vram_node_budget` | `vram_node_budget()` in datamodule.py:103 | stdout | Same. |
| `oom_batch_skipped` | OOMSkipMixin in _training.py:36 | stdout | Same. Critical safety event — should be visible in wandb. |
| `submitted`, `dry_run` | slurm.py:62,74 | stdout (dagster process) | Dagster layer, not training job. Not in wandb. |

### Orchestration Layer (dagster on login node)

| Metric | Producer | Destination | Risk |
|--------|----------|-------------|------|
| Asset materialization status | dagster SlurmTrainingComponent | dagster UI (SQLite) | **UI not yet running.** Requires `dagster-webserver` in tmux + SSH tunnel. |
| Run timing (Gantt charts) | dagster run tracking | dagster UI | Same dependency. |
| Asset lineage DAG | dagster Definitions | dagster UI | Same. |
| Checkpoint path handoff | CheckpointPathIOManager | JSON sidecar files on ESS | Works today (verified in smoke test). |

### Consumers (parsers that read the above)

| Consumer | Reads | Status |
|----------|-------|--------|
| `profile_jobs.py` | DeviceStatsMonitor columns from CSV, `gpu_stats.csv`, structlog events from SLURM logs | **Broken** — DeviceStatsMonitor not in callbacks, gpu_stats.csv never written. Structlog parsing works if log files exist. |
| wandb dashboard | All wandb-destined metrics above | **Not yet configured.** |
| DuckDB catalog | Intended: final metrics + config per run | **Does not exist.** No code writes to it. |
| dagster UI | Asset status, run logs | **Not yet running.** |

### Risk Summary

| Risk | Severity | Mitigation |
|------|----------|------------|
| wandb network failure during SLURM job | Medium | CSVLogger backup captures all `self.log()` metrics. wandb auto-retries with backoff. `WANDB_MODE=offline` as env fallback. GPU system metrics (util/temp/power) lost — no CSV equivalent. |
| ~~DeviceStatsMonitor not added~~ | ~~Blocker~~ | **Resolved** (2026-03-30). Added to trainer.yaml callbacks. |
| ~~`logger: false` not changed~~ | ~~Blocker~~ | **Resolved** (2026-03-30). Changed to WandbLogger + CSVLogger list. |
| WandbSaveConfigCallback not written | Medium | Partial config logged via `save_hyperparameters()`. Full trainer/data config missing from wandb. Ablation comparison by config fields degraded. |
| Unsupervised models missing accuracy | Low | Expected — VGAE/DGI don't have accuracy. wandb comparison panels must filter by model type. Not a code bug. |
| RL fusion dynamic keys | Low | `avg_reward`, `accuracy` only logged by RLFusionModule. Other fusion methods log `val_acc` instead. wandb handles sparse columns gracefully. |
| sacct output not persisted (structured) | Low | Raw output now captured in SLURM log on ESS. No structured DuckDB ingest yet. |
| structlog + wandb stdout interleaving | **None** | structlog→stdout, wandb progress→stderr. No stdlib bridge installed. No conflict (verified). |

---

## Current Logging State (as of 2026-03-30)

**What works (updated 2026-03-30):**
- All 6 LightningModules call `self.log()` → now routed to WandbLogger + CSVLogger
- DeviceStatsMonitor callback active → `profile_jobs.py` VRAM parsing will receive data
- `ModelCheckpoint` + `EarlyStopping` + `CurriculumEpochCallback` active
- `_epilog.sh` runs `sacct` at job end (captured in SLURM log on ESS)
- SLURM logs write to `$KD_GAT_SLURM_LOG_DIR` (default: `/fs/ess/PAS1266/kd-gat/slurm_logs/`)

**Remaining gaps:**

| Gap | Evidence | Impact |
|-----|----------|--------|
| `gpu_stats.csv` never written | `profile_jobs.py:294` expects nvidia-smi polling output. No script produces it. | wandb system metrics now cover this (GPU util/temp/power). `profile_jobs.py` parser for this file is dead code — consider removing. |
| DuckDB catalog doesn't exist | `kd_gat.duckdb` described in rules but no code writes to it. | No cross-run experiment database. wandb dashboard partially replaces this need. |
| sacct output unstructured | `_epilog.sh` prints to stdout on ESS. | Grepable but no DuckDB ingest. |
| `wandb login` not yet run | One-time setup on login node. | wandb will fail on first SLURM job without API key in `~/.netrc`. |
| `_preamble.sh` env vars not set | `WANDB_DIR`, `WANDB_DISABLE_GIT`, `WANDB_SILENT` not yet added. | wandb will write to cwd (possibly TMPDIR) and probe git on NFS (slow). |
| WandbSaveConfigCallback not written | ~10 LOC needed in cli.py for full config forwarding. | Partial config logged via `save_hyperparameters()`. Full trainer/data config missing from wandb. |

---

## Tool Evaluations

### Adopt

#### Weights & Biases (wandb) — Primary Logger

- Online mode works on OSC (compute nodes have outbound HTTPS, confirmed via curl)
- `WandbLogger` built into Lightning, zero training code changes
- Automatic system metrics: GPU util %, memory, temperature, power — every 15s
- Academic tier: 200 GB free with .edu email
- Fallback: `WANDB_MODE=offline` + `wandb sync` in epilog
- **Only tool that provides GPU compute utilization %** — DeviceStatsMonitor only logs memory

#### DeviceStatsMonitor — Memory Telemetry

- 1 line of YAML, built into Lightning
- Logs ~30 CUDA allocator keys per step: `allocated_bytes.all.peak`, `num_alloc_retries`, `num_ooms`, fragmentation (`inactive_split_bytes`)
- `profile_jobs.py` already has code to parse this output from CSVLogger
- **Requires a logger to be active** (fails silently if `logger: false`)

#### CSVLogger — Always-On Backup

- Zero-dependency fallback alongside wandb
- NFS-friendly (append-only CSV files)
- Guarantees metrics survive even if wandb has issues
- Lightning supports multiple loggers natively

#### Dagster UI (OSS) via SSH Tunnel — Pipeline Observability

- `dagster-webserver -h 127.0.0.1 -p 3000` on login node in tmux
- `ssh -L 3000:localhost:3000 pitzer.osc.edu` from local machine
- ~1-2 GB RAM, SQLite storage at `$DAGSTER_HOME`, zero cost
- Asset catalog, run history, Gantt charts, log tailing, partition status
- Binding to `127.0.0.1` = SSH key is auth (no inbound network exposure)
- Start on-demand, kill when done. No always-on requirement unless using schedules/sensors.
- **Dagster Cloud not needed.** Solo tier is $10/mo with no academic program. Only adds Insights (aggregate trends) and multi-user auth — neither matters for single-researcher use. DuckDB catalog already covers cross-run analytics.

#### Nsight Systems (nsys) — System-Wide Profiling

- Available: `module load nvhpc/25.1` (v2024.7.1)
- `nsys profile --pytorch=autograd-shapes-nvtx python -m graphids fit ...`
- Auto-annotates PyTorch ops with zero code changes
- Results exportable to SQLite/CSV/Parquet via `nsys export` — fully scriptable
- Shows: CUDA kernel timeline, CPU↔GPU transfer gaps, data loading vs compute split
- **Use for**: one-off diagnostic profiling jobs, not production training

#### torch.cuda Memory APIs — Programmatic Profiling

- `max_memory_allocated()` + `reset_peak_memory_stats()`: already used in `_probe_bytes_per_node()`. Best tool for batch sizing.
- `memory_stats()`: ~80 keys covering fragmentation, retries, pool breakdowns. Use when diagnosing *why* memory is high.
- `_record_memory_history()` + `_dump_snapshot()`: per-allocation tracing with stack traces. Viewable at pytorch.org/memory_viz. **Would directly help diagnose 13G vs 22G bimodal worker memory.**

#### ThroughputMonitor Callback — Resource Right-Sizing

```python
class ThroughputMonitor(pl.Callback):
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._start = torch.cuda.Event(enable_timing=True)
        self._end = torch.cuda.Event(enable_timing=True)
        self._start.record()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._end.record()
        torch.cuda.synchronize()
        elapsed_ms = self._start.elapsed_time(self._end)
        n_graphs = batch.num_graphs if hasattr(batch, 'num_graphs') else len(batch)
        pl_module.log("throughput/samples_per_sec", n_graphs / (elapsed_ms / 1000))
        pl_module.log("throughput/batch_ms", elapsed_ms)
```

~20 lines, ~4us overhead per step. Answers: data-bound vs compute-bound?

### Benchmark (test before committing)

#### torch.compile + PyG

- PyG 2.5+ claims full compatibility with up to 300% speedup
- `dynamic=True` already set in project
- GATv2Conv should work. Known graph breaks: `global_mean_pool()` without batch_size, `remove_self_loops()`, SplineConv, RGCNConv
- V100 gets fusion benefits but NOT `reduce-overhead` mode (CUDA graph workspace caching increases memory)
- **Run 1 VGAE job** with `compile_model: True` to measure real impact before enabling broadly

### Skip (not worth integrating)

#### NVIDIA Tools

| Tool | Why Skip |
|------|----------|
| **nvprof** | Deprecated since CUDA 11.0, removed in 13.0. Strict subset of nsys+ncu. |
| **Nsight Compute (ncu)** | Kernel-level roofline analysis. 10-100x slower than normal. Only useful after nsys identifies a problematic kernel. Low priority — bottlenecks are CPU-side. |
| **DCGM** | Binary exists on OSC but `nv-hostengine` not running. Requires admin SLURM prolog/epilog setup. wandb system metrics are the user-accessible alternative. |
| **NVTX (manual)** | `nsys --pytorch` flag auto-annotates PyTorch ops. Manual NVTX unnecessary unless profiling custom CUDA kernels. |

#### RAPIDS Stack

| Tool | Why Skip |
|------|----------|
| **cuDNN Frontend** | PyTorch 2.8 already uses cuDNN 9.10.2 internally. Frontend API targets framework devs writing custom fused kernels. Zero benefit for GNN sparse ops (scatter/gather). |
| **cuGraph / cugraph-pyg** | Primary speedup is GPU neighbor sampling on single large graphs. KD-GAT does graph classification on many small batched graphs — no sampling needed. GPU already 83-90% utilized; bottleneck is CPU-side worker memory. Pulls in entire RAPIDS stack + RMM memory manager. |
| **kvikIO / GDS** | OSC lacks GDS infrastructure (no `nvidia-fs.ko`, no NFSoRDMA). Would fall back to POSIX with zero benefit. Data loads once into RAM then iterates from memory — storage I/O isn't the bottleneck. |

#### PyTorch Features (wrong workload)

| Tool | Why Skip |
|------|----------|
| **`cudnn.benchmark`** | CNN-only ("for convolutional networks, other types currently not supported"). |
| **`channels_last`** | 4D image tensor layout, irrelevant for 2D graph data. |
| **`float32_matmul_precision("medium")`** | TF32 is Ampere+ only. No effect on V100. |
| **CUDA Graphs** | Require static tensor shapes. Fundamentally incompatible with variable-size PyG batches. |
| **`torch.compile` `reduce-overhead`** | Increases memory (CUDA graph caching). Use default mode only. |

#### Experiment Trackers

| Tool | Why Skip |
|------|----------|
| **MLflow** | SQLite on NFS has locking issues under concurrent SLURM writes. Filesystem backend deprecated in v3.6. No auto GPU metrics. More overhead than wandb for less. |
| **Aim** | NFS locking issues with RocksDB. No concurrent writers to same repo. |
| **Neptune.ai** | Dead — acquired by OpenAI, winding down by March 2026. |
| **DVC** | Duplicates existing data staging protocol. DVCLive less mature. Acquired by lakeFS. |
| **pytorch_memlab** | Abandoned (last release 2023). Use native `_record_memory_history` instead. |

#### Cluster Monitoring

| Tool | Why Skip |
|------|----------|
| **Prometheus + nvidia-smi exporters** | Requires cluster admin access to install on compute nodes. |
| **ReFrame** | System regression testing, not ML observability. |

---

## Already Configured (no action needed)

| Tool | Status | Evidence |
|------|--------|----------|
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True,garbage_collection_threshold:0.8` | `_preamble.sh:26` |
| AMP (mixed precision) | `precision: 16-mixed` | `trainer.yaml:6` |
| Gradient checkpointing | `use_reentrant=False` | `_conv.py:195-224` |
| `_probe_bytes_per_node()` | Custom VRAM probe for DynamicBatchSampler | `datamodule.py:36-72` |
| sacct + `_epilog.sh` | Coarse per-job GPU utilization | `scripts/slurm/_epilog.sh` → ESS log |
| SLURM log routing to ESS | `$KD_GAT_SLURM_LOG_DIR` → `/fs/ess/PAS1266/kd-gat/slurm_logs/` | `constants.yaml`, `orchestrate/slurm.py`, `scripts/lib/slurm.sh` |
| `torch.compile` with `dynamic=True` | Configured but not benchmarked | project config |

---

## V100 Deprecation Warning

cuDNN 9.11+ drops V100 (Volta, compute capability 7.0). PyTorch 2.8 ships cuDNN 9.10.2 (last Volta-supporting version). **PyTorch 2.9+ will likely break V100 support.** Pin `torch<2.9` when it ships, or plan migration to A100/H100.

Source: [PyTorch issue #162574](https://github.com/pytorch/pytorch/issues/162574), [cuDNN 9.11.0 release notes](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.11.0/release-notes.html)

---

## Detailed Research (individual files)

- `plans/research/nvidia-gpu-profiling-tools.md` — nsys, ncu, DCGM, NVTX, torch.cuda APIs
- `plans/research/wandb-research.md` — capabilities, jsonargparse conflict, adoption history, implementation checklist
- `plans/research/lightning-profiler-vram-research.md` — why Lightning profilers can't replace `_probe_bytes_per_node()`
