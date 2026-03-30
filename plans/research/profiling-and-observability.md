# Profiling, Optimization, and Observability

> Updated: 2026-03-30 | Environment: OSC Pitzer, V100 (16 GB), CUDA 12.6, PyTorch 2.8, PyG 2.7

## What's wired (all done)

| Layer | Tool | Where |
|-------|------|-------|
| Training metrics | WandbLogger + CSVLogger | `trainer.yaml` loggers |
| Full config logging | WandbSaveConfigCallback | `cli.py:9-17` (Lightning #19728) |
| GPU memory telemetry | DeviceStatsMonitor | `trainer.yaml` callbacks |
| GPU system metrics | wandb pynvml (util%, temp, power) | Automatic, 15s interval |
| Env vars | `WANDB_DIR`, `WANDB_DISABLE_GIT`, `WANDB_SILENT` | `_preamble.sh:25-27` |
| VRAM probe | `_probe_bytes_per_node()`, KD-aware | `datamodule.py` (runs `_step()` not `forward()`) |
| Orchestration UI | dagster webserver + daemon | `scripts/dev/dagster-ui.sh` (port 3000, SSH tunnel) |
| Checkpoint handoff | CheckpointPathIOManager | JSON sidecars at `{lake_root}/.dagster/io/` |
| SLURM resource tracking | sacct in `_epilog.sh` | ESS log files |
| CUDA alloc config | `expandable_segments:True,garbage_collection_threshold:0.8` | `_preamble.sh` |
| Mixed precision | `precision: 16-mixed` | `trainer.yaml` |
| Gradient checkpointing | `use_reentrant=False` | `_conv.py:195-224` |

## Remaining work

| Pri | Action | Effort | Purpose |
|-----|--------|--------|---------|
| P1 | Run 1 nsys profiling job | 1 SLURM job | CPU↔GPU timeline — data-bound or compute-bound? |
| P1 | Run 1 memory snapshot job | 1 SLURM job | Diagnose 13G vs 22G bimodal worker memory |
| P2 | ThroughputMonitor callback | ~20 LOC | samples/sec for resource right-sizing |
| P2 | Benchmark `torch.compile` on 1 VGAE job | 1 SLURM job | PyG 2.5+ claims 300% speedup. V100 gets fusion, not `reduce-overhead`. |
| P3 | Feed sacct output into DuckDB | Script | Cross-job resource analysis |

## Remaining gaps

| Gap | Impact |
|-----|--------|
| `gpu_stats.csv` parser in `profile_jobs.py` | Dead code — wandb covers GPU util/temp/power. Remove parser. |
| DuckDB catalog (`kd_gat.duckdb`) | No code writes to it. wandb partially replaces. |
| sacct unstructured | Grepable in ESS logs but no DuckDB ingest. |

## Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| wandb network failure | Medium | CSVLogger backup for `self.log()` metrics. GPU system metrics (util/temp/power) wandb-only — lost if wandb fails. |
| Unsupervised models missing accuracy | Low | Expected — VGAE/DGI have no accuracy. Filter by model type in wandb panels. |
| RL fusion dynamic keys | Low | `avg_reward`/`accuracy` only from `DQNFusionModule`/`BanditFusionModule` (was `RLFusionModule`, deleted). wandb handles sparse columns. |

## Tool decisions (don't re-investigate)

**Adopt**: wandb (primary logger), DeviceStatsMonitor (memory), CSVLogger (backup), dagster UI (orchestration), nsys (one-off profiling), torch.cuda memory APIs (batch sizing)

**Skip** (with reasons — don't revisit):
- **nvprof**: deprecated. **ncu**: 10-100x slower, only after nsys finds bad kernel. **DCGM**: needs admin.
- **cuGraph/cugraph-pyg**: graph classification, not sampling. **kvikIO/GDS**: no OSC infra.
- **cudnn.benchmark**: CNN-only. **channels_last**: image tensors. **TF32**: Ampere+ only. **CUDA Graphs**: variable-size batches.
- **MLflow**: NFS locking. **Aim**: RocksDB NFS issues. **Neptune**: dead. **DVC**: duplicates staging. **pytorch_memlab**: abandoned.
- **torch.compile `reduce-overhead`**: increases memory (CUDA graph caching). Use default mode only.

## V100 deprecation warning

cuDNN 9.11+ drops V100 (Volta, compute 7.0). PyTorch 2.8 ships cuDNN 9.10.2 (last Volta version). **Pin `torch<2.9` when it ships.** Sources: [PyTorch #162574](https://github.com/pytorch/pytorch/issues/162574), [cuDNN 9.11.0 notes](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.11.0/release-notes.html)

## Detailed research (individual files)

- `nvidia-gpu-profiling-tools.md` — nsys, ncu, DCGM, NVTX, torch.cuda APIs
- `wandb-research.md` — decisions, adoption history, jsonargparse conflict
- `lightning-profiler-vram-research.md` — why Lightning profilers can't replace `_probe_bytes_per_node()`
