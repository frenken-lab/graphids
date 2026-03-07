# Memory Optimization Guide

This guide documents the memory optimization features in the KD-GAT pipeline, designed to prevent OOM (Out of Memory) errors on both CPU and GPU when processing large datasets.

## Overview

The pipeline processes CAN bus data as graphs, which can consume significant memory:
- **hcrl_sa**: ~9K graphs (works on most systems)
- **set_01-04, hcrl_ch**: ~50-100K+ graphs (requires optimization)

## Memory Optimizations

### 1. Memory-Mapped Graph Loading

**File**: `graphids/core/training/datamodules.py`

Uses PyTorch's `mmap=True` parameter to memory-map cache files. Graphs are loaded on-demand from disk rather than all at once.

**Requirements**: PyTorch 2.1+

### 2. Teacher CPU Offloading

**File**: `graphids/pipeline/stages/modules.py` (VGAEModule, GATModule)

Offloads teacher model to CPU after each forward pass during KD training, freeing GPU memory between batches.

```bash
python -m graphids.pipeline.cli autoencoder --model vgae --scale small \
    --auxiliaries kd_standard --teacher-path path/to/teacher.pt \
    -O training.offload_teacher_to_cpu true
```

### 3. Chunked Difficulty Scoring

**File**: `graphids/pipeline/stages/training.py`

Curriculum learning processes graphs in chunks and clears GPU cache between chunks to prevent memory accumulation.

### 4. Config-Driven Batch Sizing

**File**: `graphids/pipeline/stages/batch_sizing.py`

Batch size is determined from config values: `batch_size * safety_factor`. Dynamic batching uses PyG's `DynamicBatchSampler` to pack variable-size graphs to a node budget.

### 5. Lightning DeviceStatsMonitor

**File**: `graphids/pipeline/stages/trainer_factory.py`

Lightning's built-in `DeviceStatsMonitor` callback automatically tracks GPU/CPU memory stats every training step.

## Configuration Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gradient_checkpointing` | `True` | Trade compute for memory in model layers |
| `offload_teacher_to_cpu` | `False` | Move teacher to CPU between batches during KD |
| `use_teacher_cache` | `True` | Cache teacher outputs (increases memory) |
| `clear_cache_every_n` | `100` | Clear CUDA cache every N steps |
| `safety_factor` | `0.5` | Fraction of configured batch_size to use |
| `batch_size` | `4096` | Base batch size (scaled by safety_factor) |
| `dynamic_batching` | `True` | Use DynamicBatchSampler for variable-size graphs |
| `precision` | `16-mixed` | Mixed precision training (halves activation memory) |

## Troubleshooting OOM Errors

### CPU OOM
1. Reduce `num_workers` (`-O num_workers 4`)
2. Increase SLURM memory (`--mem=256G`)

### GPU OOM
1. Lower `safety_factor` (`-O training.safety_factor 0.3`)
2. Enable teacher offloading (`-O training.offload_teacher_to_cpu true`)
3. Reduce batch size (`-O training.batch_size 2048`)
4. Enable gradient checkpointing (on by default)

### Monitoring
```bash
watch -n 1 nvidia-smi          # GPU memory
seff <jobid>                   # Post-job memory usage
```

MLflow + DeviceStatsMonitor log memory metrics at each step.
