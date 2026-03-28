## The Full Pipeline — Ground Truth

### Phase 1: Preprocessing (runs once per dataset, CPU only)

```
Raw CSVs (NFS)                        Cache artifacts (NFS → scratch → TMPDIR)
─────────────────                     ──────────────────────────────────────────

data/automotive/set_02/
├── normal_traffic.csv
├── dos_attack.csv
├── fuzzing_attack.csv
└── ...

        │
        ▼  Step 1: pl.scan_csv() — lazy, no data loaded yet
        │
        ▼  Step 2: Column normalization (Polars lazy exprs)
        │           arbitration_id → arb_id
        │           data_field → payload
        │
        ▼  Step 3: Hex payload parsing (Polars lazy)
        │           "0A1B..." → byte_0=0x0A, byte_1=0x1B, ... byte_7
        │           8 Float32 columns
        │
        ▼  Step 4: Shannon entropy per message (Polars lazy)
        │           sum(-p * log(p)) over 8 bytes
        │
        ▼  Step 5: Attack type tagging from filename
        │           dos_attack.csv → attack_type=1
        │
        ▼  Step 6: Sort by timestamp, .collect()  ← FIRST MATERIALIZATION
        │           One big Polars DataFrame in RAM
        │           set_02: ~5M rows × 15 columns
        │
        ▼  Step 7: Vocabulary — unique arb_ids → dense int IDs
        │           ~30-50 unique CAN IDs → {0: OOV, 1: "0x123", ...}
        │
        ▼  Step 8: sliding_window_graphs()
        │
        │   ┌──────────────────────────────────────────────────┐
        │   │  8a: Window assignment                           │
        │   │      window_size=100, stride=100 (non-overlapping)
        │   │      _wid = row_idx // 100                       │
        │   │      set_02: ~50K windows                        │
        │   │                                                  │
        │   │  8b: Three parallel lazy frames:                 │
        │   │                                                  │
        │   │  ┌─ stats_lf ─────────────────────────────────┐  │
        │   │  │ group_by(_wid, node_id).agg(               │  │
        │   │  │   byte mean/std/range × 8 = 24 features    │  │
        │   │  │   msg_count, entropy_mean                   │  │
        │   │  │   skewness (clamp ±10), kurtosis (clamp ±10)│ │
        │   │  │   split_half_ratio, change_rate             │  │
        │   │  │   node_iat_mean, node_iat_std               │  │
        │   │  │   in_degree=0, out_degree=0 (placeholders)  │  │
        │   │  │ ) → 35 node features (NODE_FEATURE_COUNT)   │  │
        │   │  └─────────────────────────────────────────────┘  │
        │   │                                                  │
        │   │  ┌─ edges_base ───────────────────────────────┐  │
        │   │  │ shift(-1).over(_wid) → temporal adjacency   │  │
        │   │  │ IAT = timestamp.diff()                      │  │
        │   │  │ byte_diffs = abs(byte_i.diff()) × 8         │  │
        │   │  │ edge_freq, bidir_flag                       │  │
        │   │  │ → 11 edge features (EDGE_FEATURE_COUNT)     │  │
        │   │  └─────────────────────────────────────────────┘  │
        │   │                                                  │
        │   │  ┌─ labels_lf ────────────────────────────────┐  │
        │   │  │ group_by(_wid).agg(                         │  │
        │   │  │   y = max(attack) > 0,                      │  │
        │   │  │   attack_type = mode(attack_type where >0)  │  │
        │   │  │ )                                           │  │
        │   │  └─────────────────────────────────────────────┘  │
        │   │                                                  │
        │   │  8c: Sequential .collect() (saves ~20-30 GB peak) │
        │   │      labels → stats → edges                      │
        │   │                                                  │
        │   │  8d: Local ID remapping (bulk Polars join)       │
        │   │      global node_id → 0-based per-window index   │
        │   │                                                  │
        │   │  8e: Polars → torch bulk handoff                 │
        │   │      .to_torch(dtype=Float32) for all columns    │
        │   │      Flat tensors: [total_nodes, 35], etc.       │
        │   │                                                  │
        │   │  8f: _assemble_graphs() — ProcessPoolExecutor    │
        │   │      up to 8 fork workers (CPU only, no CUDA)    │
        │   │      Per window:                                 │
        │   │        - slice flat tensors by (start, count)    │
        │   │        - NetworkX clustering_coeff (~0.65ms/win) │
        │   │        - np.bincount → in_degree, out_degree     │
        │   │        - Build Data(x, edge_index, edge_attr,    │
        │   │                     node_id, y, attack_type)     │
        │   │      set_02: ~50K windows × 0.65ms ≈ 30s w/8 CPUs│
        │   └──────────────────────────────────────────────────┘
        │
        ▼  Step 9: PyG InMemoryDataset.collate()
        │           Stacks all Data objects into one giant Data + slices dict
        │           This is the PREPROCESSING collation (done once)
        │
        ▼  Step 10: atomic_save() → torch.save + fsync + rename
        │
        ├──→ {lake}/cache/v7.0.0/{dataset}/processed/data_train.pt
        │    (single file: collated Data + slices, set_02 ≈ 5.9 GB)
        │
        ├──→ {lake}/cache/v7.0.0/{dataset}/processed/num_arb_ids.txt
        │
        ├──→ {lake}/cache/v7.0.0/{dataset}/cache_metadata.json
        │    (graph stats: node counts p95/mean/max, edge counts, etc.)
        │
        └──→ {lake}/cache/v7.0.0/{dataset}/processed/.complete
```

### Phase 2: Training Data Loading (every epoch, every batch)

```
           Storage hierarchy
           ─────────────────
           NFS (permanent)     →    Scratch (GPFS)     →    TMPDIR (local SSD)
           /fs/ess/PAS1266/         /fs/scratch/PAS1266/     /tmp/kd-gat-data/
           ~50ms/read               ~5ms/read                ~0.1ms/read
                                                             (staged by _preamble.sh)


    ┌─────────────────────────────────────────────────────────────────────┐
    │                     TRAINING LOOP (per epoch)                      │
    │                                                                    │
    │  1. torch.load(data_train.pt, mmap=True)                          │
    │     ┌──────────────────────────────────────────────────┐           │
    │     │  Returns (data, slices) — memory-mapped tensors  │           │
    │     │  No full copy into RAM. Pages fault on access.   │           │
    │     │  Done ONCE at setup(), not per epoch.            │           │
    │     └──────────────────────────────────────────────────┘           │
    │                                                                    │
    │  2. Train/val split — torch.randperm(seed=42)                     │
    │     80% train indices, 20% val indices                             │
    │     Both share same mmap'd data, different _indices                │
    │     Done ONCE at setup().                                          │
    │                                                                    │
    │  3. DynamicBatchSampler (runs in MAIN process)                    │
    │     ┌──────────────────────────────────────────────────┐           │
    │     │  Walks graph indices, accumulates node counts    │           │
    │     │  until budget (batch_size × p95_nodes) is hit.   │           │
    │     │  Yields list[int] of graph indices per batch.    │           │
    │     │  Cheap: just integer arithmetic on cached stats. │           │
    │     └──────────────────────────────────────────────────┘           │
    │         │                                                          │
    │         │ sends index lists to worker queues                       │
    │         ▼                                                          │
    │  4. WORKER PROCESSES (spawn, num_workers=2)                       │
    │     ┌──────────────────────────────────────────────────┐           │
    │     │  Each worker:                                    │           │
    │     │                                                  │           │
    │     │  a) dataset[i] for each index in batch           │           │
    │     │     → InMemoryDataset.__getitem__                │           │
    │     │     → slice (data, slices) tensors               │           │
    │     │     → reconstruct individual Data objects         │           │
    │     │     Cost: cheap (tensor slicing, no copy)        │           │
    │     │                                                  │           │
    │     │  b) Batch.from_data_list(data_list)   ← THE BOTTLENECK     │
    │     │     For each Data object in batch:               │           │
    │     │       - cat x tensors → [total_nodes, 35]        │           │
    │     │       - reindex edge_index (add node offsets)     │           │
    │     │       - cat edge_attr → [total_edges, 11]        │           │
    │     │       - build batch vector [total_nodes]          │           │
    │     │       - stack y → [batch_size]                    │           │
    │     │     Cost: ~70ms per batch (CPU-bound)            │           │
    │     │     This is pure Python + torch.cat loops.        │           │
    │     │                                                  │           │
    │     │  c) Return collated Batch to main via IPC queue  │           │
    │     │     (file_system sharing strategy → /dev/shm)    │           │
    │     └──────────────────────────────────────────────────┘           │
    │         │                                                          │
    │         ▼  pin_memory=True: DataLoader pins batch in page-locked RAM
    │         │                                                          │
    │  5. MAIN PROCESS pulls batch from queue                           │
    │     ┌──────────────────────────────────────────────────┐           │
    │     │  batch.to(device, non_blocking=True)             │           │
    │     │  Async DMA: pinned CPU → GPU via PCIe            │           │
    │     │  Cost: ~1-2ms for typical batch                  │           │
    │     └──────────────────────────────────────────────────┘           │
    │         │                                                          │
    │         ▼                                                          │
    │  6. GPU FORWARD + BACKWARD                                        │
    │     ┌──────────────────────────────────────────────────┐           │
    │     │  VGAE: encoder(x, edge_index, edge_attr)         │           │
    │     │        → mu, logvar → z (reparameterize)         │           │
    │     │        → decoder(z, edge_index)                  │           │
    │     │        → recon_loss + kl_loss                    │           │
    │     │  Cost: ~10ms (VGAE) / ~25ms (GAT)               │           │
    │     │  This is the ONLY GPU work per step.             │           │
    │     └──────────────────────────────────────────────────┘           │
    │         │                                                          │
    │         ▼                                                          │
    │  7. GPU sits idle waiting for step 4b to finish next batch        │
    │     (this is the 70% idle time you see in gpu_stats.csv)          │
    │                                                                    │
    │  WRITES (async, non-blocking):                                    │
    │  - CSVLogger → metrics.csv (every log_every_n_steps=50)           │
    │  - DeviceStatsMonitor → same CSV (CUDA allocator stats + cpu)     │
    │  - ModelCheckpoint → best_model.ckpt (on val improvement)         │
    │                                                                    │
    └─────────────────────────────────────────────────────────────────────┘
```

### The bottleneck, visualized

```
    Time (ms) →  0    10    20    30    40    50    60    70    80
                 │     │     │     │     │     │     │     │     │
    Worker 0:    │▓▓▓▓▓▓▓▓▓▓▓▓▓ Batch.from_data_list() ▓▓▓▓▓▓▓│▓▓▓▓▓▓▓
    Worker 1:    │  ▓▓▓▓▓▓▓▓▓▓▓▓▓ Batch.from_data_list() ▓▓▓▓▓▓│▓▓▓▓▓▓▓
                 │     │     │     │     │     │     │     │     │
    GPU:         │█fwd█│█bwd█│                              │█fwd█│█bwd█│
                 │ 5ms │ 5ms │◁──── 60ms idle ────────────▷│ 5ms │ 5ms │
                 │     │     │     │     │     │     │     │     │
                          ▲
                          └── GPU done. Waiting. Workers still collating.
```

Step 4b — `Batch.from_data_list()` — is the single operation that dominates wall time. Everything else (sampling, slicing, H2D transfer, GPU compute, checkpoint writes) is negligible by comparison. Pre-collating would eliminate this 70ms per batch.
