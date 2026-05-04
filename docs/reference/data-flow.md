# Pipeline Data Flow

## Phase 1: Preprocessing (runs once per dataset, CPU only)

```
Raw CSVs (NFS)                        Cache artifacts (NFS -> scratch -> TMPDIR)

data/automotive/<dataset>/
+-- normal_traffic.csv
+-- dos_attack.csv
+-- ...

        |
        v  1. pl.scan_csv() -- lazy, no data loaded yet
        v  2. Column normalization (arbitration_id -> arb_id, data_field -> payload)
        v  3. Hex payload parsing -> 8 Float32 byte columns + Shannon entropy
        v  4. Attack type tagging from filename stem + parent dir
        v  5. Sort by timestamp, .collect()  <-- FIRST MATERIALIZATION
        |
        v  6. Vocabulary -- unique arb_ids -> dense int IDs
        |
        v  7. sliding_window_graphs()
        |
        |   7a: Window assignment (window_size=100, stride=100)
        |
        |   7b: Three lazy frames built from one scan:
        |     +- stats_lf: group_by(_wid, node_id) -> 35 node features
        |     +- edges_base: shift(-1).over(_wid) -> 11 edge features
        |     +- labels_lf: group_by(_wid) -> y, attack_type
        |
        |   7c: Sequential .collect() (saves ~20-30 GB peak)
        |   7d: Bidirectional edge flag via self-join
        |   7e: Clustering coeff + degree entirely in Polars (no NetworkX)
        |   7f: Local ID remapping (bulk Polars join)
        |   7g: Polars -> torch bulk handoff (.to_torch Float32)
        |   7h: Pre-collation -- (Data, slices) built directly from bulk tensors.
        |       RLE boundaries become slice offsets; no list[Data], no collate().
        |       Peak memory ~1x final tensor size.
        |   7i: Graphs presorted by node count before save. Adjacent graphs on
        |       disk have similar size -> NodeBudgetBatchSampler + bucket shuffle
        |       produces sequential mmap page faults; reduces VRAM fragmentation.
        |
        v  8. Returns (Data, slices, num_graphs) from bulk tensors
        v  9. atomic_save() -> torch.save + fsync + rename
        |
        +---> {lake_root}/cache/v9.0.0/{dataset}/processed/data_train.pt
        +---> {lake_root}/cache/v9.0.0/{dataset}/processed/data_test.pt
        +---> {lake_root}/cache/v9.0.0/{dataset}/cache_metadata.json
        +---> {lake_root}/cache/v9.0.0/{dataset}/processed/.complete
```

`num_arb_ids` is read at load time from `cache_metadata.json`. The
authoritative value is written from the *shared* arb-id vocab built
in `CANBusSource.build()` (scans every split's source_dirs before any
tensor is constructed) and persisted as `{root}/vocab.json`. Index 0
is reserved for UNK; real ids start at 1. The earlier per-split
`node_id.max() + 1` derivation was removed because test subdirs can
contain arb_ids absent from train, which under-sized the embedding
table relative to the real deployment vocabulary and crashed at
inference. See `graphids/core/data/preprocessing/vocab.py`,
`graphids/core/data/preprocessing/metadata.py` (schema v3; `vocab_digest` is an
invariant cache key), and `~/plans/oov-embedding-handling.md`.

## Phase 2: Training Data Loading

```
Storage hierarchy:
  NFS (~50ms/read) -> Scratch/GPFS (~5ms/read) -> TMPDIR/local SSD (~0.1ms/read)
  (staged by _preamble.sh before training starts)

+---------------------------------------------------------------------+
|                     TRAINING LOOP (per epoch)                       |
|                                                                     |
|  1. torch.load(data_train.pt, mmap=True)                           |
|     Memory-mapped tensors, pages fault on access.                   |
|     Done ONCE at setup(), not per epoch.                            |
|                                                                     |
|  2. Train/val split -- torch.randperm(seed=seed)                   |
|     val_fraction=0.2 default (configurable). Done ONCE.            |
|                                                                     |
|  3. PRE-BATCHED STANDARD PATH (dynamic_batching=True)              |
|     First train_dataloader() call only:                             |
|       a) node_budget() probes VRAM -> max nodes per batch           |
|       b) NodeBudgetBatchSampler plans all batches deterministically |
|       c) Batch.from_data_list() collates all batches upfront        |
|     Subsequent epochs: shuffle batch ORDER only, no re-collation    |
|     num_workers=0 -- each __getitem__ is O(1) Batch.clone()         |
|                                                                     |
|  4. Val/test loaders: _build_loader() with NodeBudgetBatchSampler  |
|     + PyG DataLoader, num_workers auto-sized via autosize_workers() |
|     Wrapped in PrefetchLoader for async H2D when GPU available.     |
|                                                                     |
|  5. batch.to(device, non_blocking=True) -- async DMA (~1-2ms)      |
|     (PrefetchLoader issues this via CUDA stream)                    |
|                                                                     |
|  6. GPU forward + backward                                         |
|     VGAE: ~10ms  |  GAT: ~25ms                                     |
|                                                                     |
|  WRITES: MLflowTrainingCallback (per-epoch metrics, peak VRAM),     |
|          ModelCheckpoint (best/last ckpts + SHA256 sidecar)         |
+---------------------------------------------------------------------+

CURRICULUM PATH (sampler="curriculum"):
  setup(): score normal-class graphs via VGAE, bucket into K difficulty tiers.
  first train_dataloader(): pre-batch each tier + attack tier.
  CurriculumEpochCallback: selects active tiers each epoch (O(1) tier swap).
  Also num_workers=0 -- same pre-batched O(1) clone pattern.
```

### Sampler: NodeBudgetBatchSampler

Replaces PyG's `DynamicBatchSampler`. Reads `num_nodes_per_graph` derived from
cache slice offsets at zero I/O cost -- avoids 50K mmap'd `Data` reconstructions
per epoch on large datasets. With `shuffle=True`, uses bucket shuffle (sort by
size -> chunk into buckets -> shuffle bucket order + within-bucket), keeping
batch-to-batch size variance low for VRAM allocator stability.

`make_graph_loader` (`sampler.py:28`) is the single factory for all loaders:
sets `spawn` multiprocessing context, `persistent_workers`, and `file_system`
sharing strategy in worker init. Wraps with `PrefetchLoader` when a GPU device
is available.
