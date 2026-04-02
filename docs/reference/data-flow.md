# Pipeline Data Flow — Ground Truth

Timeless reference for how data moves through preprocessing and training.

## Phase 1: Preprocessing (runs once per dataset, CPU only)

```
Raw CSVs (NFS)                        Cache artifacts (NFS → scratch → TMPDIR)
─────────────────                     ──────────────────────────────────────────

data/automotive/set_02/
├── normal_traffic.csv
├── dos_attack.csv
├── fuzzing_attack.csv
└── ...

        │
        ▼  1. pl.scan_csv() — lazy, no data loaded yet
        ▼  2. Column normalization (arbitration_id → arb_id, data_field → payload)
        ▼  3. Hex payload parsing → 8 Float32 byte columns
        ▼  4. Shannon entropy per message
        ▼  5. Attack type tagging from filename
        ▼  6. Sort by timestamp, .collect()  ← FIRST MATERIALIZATION
        │     set_02: ~5M rows × 15 columns
        │
        ▼  7. Vocabulary — unique arb_ids → dense int IDs (~30-50 unique)
        │
        ▼  8. sliding_window_graphs()
        │
        │   8a: Window assignment (window_size=100, stride=100)
        │       set_02: ~50K windows
        │
        │   8b: Three parallel lazy frames:
        │     ┌─ stats_lf: group_by(_wid, node_id) → 35 node features
        │     ├─ edges_base: shift(-1).over(_wid) → 11 edge features
        │     └─ labels_lf: group_by(_wid) → y, attack_type
        │
        │   8c: Sequential .collect() (saves ~20-30 GB peak)
        │   8d: Local ID remapping (bulk Polars join)
        │   8e: Polars → torch bulk handoff (.to_torch Float32)
        │   8f: Zero-copy collation — bulk tensors ARE the collated format.
        │       RLE boundaries from group_by become the slices dict directly.
        │       No per-window Data objects, no list[Data], no collate() call.
        │       Peak memory: ~1x final tensor size (was ~3x before 2026-03-31 fix).
        │
        ▼  9. Returns (Data, slices_dict, num_graphs) directly from bulk tensors
        ▼ 10. atomic_save() → torch.save + fsync + rename
        │
        ├──→ {lake}/cache/v7.0.0/{dataset}/processed/data_train.pt  (set_02 ≈ 5.9 GB)
        ├──→ {lake}/cache/v7.0.0/{dataset}/processed/num_arb_ids.txt
        ├──→ {lake}/cache/v7.0.0/{dataset}/cache_metadata.json
        └──→ {lake}/cache/v7.0.0/{dataset}/processed/.complete
```

## Phase 2: Training Data Loading (every epoch, every batch)

```
Storage hierarchy:
  NFS (~50ms/read) → Scratch/GPFS (~5ms/read) → TMPDIR/local SSD (~0.1ms/read)
  (staged by _preamble.sh)

┌─────────────────────────────────────────────────────────────────────┐
│                     TRAINING LOOP (per epoch)                      │
│                                                                    │
│  1. torch.load(data_train.pt, mmap=True)                          │
│     Memory-mapped tensors, pages fault on access.                  │
│     Done ONCE at setup(), not per epoch.                           │
│                                                                    │
│  2. Train/val split — torch.randperm(seed=42)                     │
│     80/20 split. Both share same mmap'd data. Done ONCE.          │
│                                                                    │
│  3. DynamicBatchSampler (MAIN process)                            │
│     Walks graph indices, accumulates node counts until budget      │
│     (batch_size × p95_nodes). Cheap integer arithmetic.            │
│         │                                                          │
│         ▼ sends index lists to worker queues                       │
│                                                                    │
│  4. WORKER PROCESSES (spawn, num_workers=2, persistent_workers)   │
│     a) dataset[i] → InMemoryDataset.__getitem__ → tensor slicing  │
│     b) Batch.from_data_list(data_list) — collation                │
│        Cold (epoch 1, _data_list=None): ~166ms/batch               │
│        Warm (epoch 2+, _data_list cached): ~52ms/batch             │
│     c) Return Batch to main via IPC (file_system sharing)          │
│         │                                                          │
│         ▼ pin_memory=True: page-locked RAM                         │
│                                                                    │
│  5. batch.to(device, non_blocking=True) — async DMA (~1-2ms)      │
│                                                                    │
│  6. GPU forward + backward                                        │
│     VGAE: ~10ms  |  GAT: ~25ms                                    │
│                                                                    │
│  WRITES (async): CSVLogger, WandbLogger, DeviceStatsMonitor,      │
│                  ModelCheckpoint (on val improvement)               │
└─────────────────────────────────────────────────────────────────────┘
```

### Steady-state bottleneck

```
Time (ms) →  0    10    20    30    40    50    60
             │     │     │     │     │     │     │
Worker 0:    │▓▓▓▓▓▓▓▓▓▓▓▓▓▓ warm collate ▓▓▓▓▓│▓▓▓▓▓▓▓▓
Worker 1:    │  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓ warm collate ▓▓▓│▓▓▓▓▓▓▓▓▓
             │     │     │     │     │     │     │
GPU:         │█fwd█│█bwd█│  gap  │█fwd█│█bwd█│  gap  │
             │ 5ms │ 5ms │ ~30ms │ 5ms │ 5ms │       │

With persistent_workers + warm cache (T_c≈52ms):
  2 workers produce at 2/52ms ≈ 38 batches/sec
  GPU consumes at 1/10ms = 100 batches/sec (VGAE)
  Predicted util: 38% — but measured 83-90% in Run 003
  (prefetch buffer smoothing + batch size variance close the gap)
```

### Collation call sites

| # | Location | Input type | Hot loop? | Notes |
|---|----------|-----------|-----------|-------|
| 1 | `datamodule._build_loader` | CANBusDataset | Yes — every batch | Standard PyG DataLoader |
| 2 | `curriculum.py` train loader | CANBusDataset | Yes — every batch | Same path |
| 3 | `curriculum.py` val loader | CANBusDataset | Yes — val epoch | Same path |
| 4-12 | Various (score_difficulty, encode_dataset, etc.) | list[Data] | No — once | Cold path, no optimization needed |
