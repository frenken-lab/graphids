Image pipeline (ImageFolder):
  Dataset object = list of file paths        → ~5 MB
  __getitem__(idx) = open JPEG, decode, transform → ~150 KB tensor
  Pickle to workers = list of file paths     → ~5 MB
  Worker memory = Python + torchvision + one batch in flight → ~1-2 GB
  Workers read images lazily from disk. The dataset object is tiny. Pickle is tiny.

  Your PyG pipeline (InMemoryDataset):
  Dataset object = ALL 503K graphs pre-collated as tensors → ~6.3 GB
  __getitem__(idx) = slice into pre-collated tensors → ~1 KB view
  Pickle to workers = ALL tensors, materialized from mmap → ~6.3 GB PER WORKER
  Worker memory = Python + PyG + full dataset copy → ~8-10 GB

  The difference: ImageFolder stores file paths, InMemoryDataset stores tensors. When spawn pickles
  the dataset to workers, ImageFolder sends kilobytes of paths, InMemoryDataset sends gigabytes of
  tensor data.

  It gets worse. Your main process does torch.load(data_train.pt, mmap=True) — the tensors are
  memory-mapped, pages fault in on demand. But pickle doesn't know about mmap. When it serializes a
  tensor to send to a worker, it reads the full tensor data and writes it to a NEW shared memory
  segment. The mmap was supposed to save memory, but pickle defeats it by materializing everything.

  And each worker gets its own pickle. So with 2 workers:
  Main process:     ~15 GB  (Python + CUDA + mmap'd pages faulted in by DynamicBatchSampler)
  Worker 0 pickle:  ~6.3 GB (tensors copied to /dev/shm)
  Worker 1 pickle:  ~6.3 GB (SEPARATE copy to /dev/shm)
  _data_list cache: ~3-5 GB (DynamicBatchSampler calls dataset[i] for all 503K graphs at construction)
  CUDA/overhead:    ~5-8 GB
  Total:            ~36-40 GB → matches observed 37.7 GB

  Why other GNN pipelines don't hit this: most PyG tutorials use small datasets (Cora: 2.7K nodes, TU
  datasets: <10K graphs with <100 nodes each). The entire dataset fits in <100 MB. The 6× bloat is
  invisible. Your dataset is 503K graphs with 19M total nodes — it's orders of magnitude larger than
  what InMemoryDataset was designed for.

  The fix is conceptually simple: workers should mmap the .pt file independently instead of receiving
  pickled tensor copies. Each worker opens the same file, the OS page cache deduplicates, physical
  memory = one copy shared by all processes. This is exactly how ImageFolder works — each worker
  independently opens the same JPEG files.

  Your _FastCollate is already 90% of the way there. It only needs _data tensors + slices. If instead
  of receiving these via pickle, each worker did its own torch.load(path, mmap=True) in
  worker_init_fn, you'd eliminate the pickle copy entirely.



  Right. Let me trace the chain of decisions that led here:

  1. CSV → graphs at preprocessing time — made sense to avoid recomputing 50K windows of features
  every job. But it means the training-time data format is PyG Data objects, not raw tensors. You're
  locked into PyG's storage and loading patterns.
  2. InMemoryDataset — needed because DynamicBatchSampler must know num_nodes per graph upfront to
  pack batches to a node budget. It iterates all 503K graphs at construction to read their sizes. If
  graphs lived on disk as individual files, that's 503K file opens just to read graph metadata.
  3. DynamicBatchSampler — needed because graphs are variable-sized (30-50 nodes). Fixed batch counts
  give wildly inconsistent VRAM usage. You spent time computing p95 node counts and mean nodes to set
  the budget. The sampler needs the full graph size distribution in memory.
  4. The node budget itself — exists because GAT attention is O(N²) per head, so a batch with one
  500-node graph behaves completely differently from a batch with ten 50-node graphs, even though both
   have 500 total nodes. You can't just count graphs.

  So the cascade is: variable-size graphs → need node budget → need all graph sizes in memory →
  InMemoryDataset → pickle bloat.

  The question is where to break this chain. The graph sizes are just integers — you don't need the
  full 6.3GB of tensor data to know that graph 42 has 38 nodes. That metadata could live in a separate
   tiny file (~4MB for 503K int32s). Then:

  - DynamicBatchSampler reads the metadata file (4MB), not the full dataset
  - Workers mmap the .pt file independently for the actual tensor data
  - No pickle of 6.3GB tensors

  The premature CSV→graph conversion isn't the problem — that preprocessing step is genuinely useful.
  The problem is that graph metadata (sizes) is entangled with graph data (tensors) in a single 5.9GB
  blob, and everything that needs sizes pulls in the full blob.
                                                                   
