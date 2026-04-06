"""Pre-batching pipeline for graph datasets.

# ==========================================================================
# HIGH-LEVEL DESIGN: Pre-batch Pipeline
# ==========================================================================
#
# Goal: Build pre-collated Batch objects once, reuse across epochs.
# Eliminates per-step separate() + copy.copy() + Batch.from_data_list()
# overhead that causes GPU idle time waiting on CPU workers.
#
# --------------------------------------------------------------------------
# Phase 1: Probe (offline, runs once per model×dataset combo)
# --------------------------------------------------------------------------
#
# Happens during data staging or as a separate SLURM job (probe-budget).
# Writes results to a lookup table (CSV/JSON in the data lake).
#
# Inputs:
#   - Dataset identity (name, preprocessing version, split)
#   - Model spec (family, scale, conv_type, heads, KD teacher if any)
#
# Steps:
#   1. Load model onto GPU. If KD: load teacher too → measure teacher VRAM.
#   2. Read dataset metadata from cache_metadata.json:
#      - Per-graph node counts (already stored as num_nodes_per_graph tensor)
#      - Per-graph edge counts (need to add to cache metadata if missing)
#      - Graph index → (node_start, node_end, edge_start, edge_end) offsets
#        from InMemoryDataset._data + slices dict
#   3. Measure free VRAM = total - model - teacher - overhead.
#   4. VRAM probe (_probe_vram): forward + backward on small batch →
#      bytes_per_node, backward_multiplier. O(nodes+edges) cost model
#      is more accurate than O(nodes) alone — edge_attr and message
#      passing scale with edges.
#   5. Compute node_budget = free_vram * safety / effective_bytes_per_node.
#   6. Write to lookup table:
#      key = (dataset, model_family, scale, conv_type, has_kd)
#      value = (node_budget, bytes_per_node, backward_mult, mean_nodes)
#
# The lookup table means real training never re-probes. budget.py already
# does steps 1-5; step 6 (persist to lookup) is the missing piece.
#
# --------------------------------------------------------------------------
# Phase 2: Plan batches (once per epoch for standard, per-epoch for curriculum)
# --------------------------------------------------------------------------
#
# Inputs:
#   - node_budget (from lookup table or live probe)
#   - Per-graph node counts (precomputed tensor, zero I/O)
#   - Sampler (NodeBudgetBatchSampler for standard, CurriculumSampler for curriculum)
#
# Steps:
#   1. Run sampler → list of batch index lists (batch_plans).
#      NodeBudgetBatchSampler already does this — it yields lists of graph
#      indices packed to the node budget.
#   2. For standard path: batch_plans are STABLE across epochs (same dataset,
#      same budget, shuffle order changes but packing is equivalent).
#      → Compute once, shuffle batch order per epoch.
#   3. For curriculum path: active indices change per epoch via set_epoch().
#      → Must recompute batch_plans each epoch. But the collation cost is
#      the same as current — no regression, just no free win.
#
# --------------------------------------------------------------------------
# Phase 3: Pre-collate (once for standard, per-epoch for curriculum)
# --------------------------------------------------------------------------
#
# Inputs:
#   - batch_plans from Phase 2
#   - Dataset (InMemoryDataset with _data blob + slices)
#
# Steps:
#   1. For each batch_plan, build a Batch:
#      - Current: Batch.from_data_list([dataset[i] for i in plan])
#        This calls separate() per graph + collate. Works, costs ~5ms/batch.
#      - Better (future): direct blob slicing using _slice_dict offsets.
#        Skip separate() entirely — slice x[node_start:node_end],
#        edge_index[:, edge_start:edge_end] for contiguous index ranges.
#        Requires pre-sorting by node budget groups (Phase 2 output).
#   2. Store as list[Batch] — the "pre-batched dataset".
#   3. Wrap in a trivial map-style Dataset:
#        __getitem__(i) → pre_batched[i]  # O(1), already a Batch
#        __len__() → len(pre_batched)
#
# --------------------------------------------------------------------------
# Phase 4: Training loop
# --------------------------------------------------------------------------
#
# DataLoader(PreBatchedDataset, batch_size=None, shuffle=True)
#
# Each worker fetches one pre-built Batch. The DataLoader's collate_fn
# is identity (batch_size=None means no auto-batching — each __getitem__
# returns a complete batch). Workers do O(1) work.
#
# For standard path:
#   - Pre-batched list built once in first train_dataloader() call
#   - Batch ORDER shuffled per epoch (torch.randperm over batch indices)
#   - Graphs within each batch are fixed (acceptable — packing is
#     deterministic for a given budget anyway)
#
# For curriculum path:
#   - set_epoch() → new active indices → new batch_plans → re-collate
#   - Same cost as current per-epoch, but future optimization path is:
#     pre-sort graphs by difficulty bucket, pre-collate bucket-aligned
#     batches, then curriculum just selects which buckets are active
#     (batch-level selection, not graph-level)
#
# ==========================================================================
# OPEN QUESTIONS
# ==========================================================================
#
# 1. Curriculum pre-batching strategy:
#    - Current: re-collate every epoch (no win over status quo).
#    - Idea: pad batches to a fixed node count with zero-filled dummy nodes.
#      All batches become the same "shape" → fixed memory, no fragmentation,
#      curriculum just masks which graphs are real vs padding.
#      This is how image batching works (pad to max size). Trade-off:
#      wasted compute on padding nodes, but consistent batch shape means
#      torch.compile can optimize, CUDA kernels don't recompile, and
#      memory allocation is predictable.
#    - Idea: pre-sort by difficulty, build batches per difficulty tier.
#      Curriculum selects tiers (batch-granular), not individual graphs.
#      Coarser but O(1) epoch transition — just change which pre-built
#      batches are in the active set.
#
# 2. Preprocessing metadata for faster lookups:
#    - cache_metadata.json already has aggregate stats (mean, p95, etc.)
#    - Missing: per-graph (node_count, edge_count, byte_size) index file.
#      A simple CSV or tensor: graph_idx → (num_nodes, num_edges).
#      NodeBudgetBatchSampler already reads num_nodes_per_graph from the
#      dataset — could persist this as a sidecar tensor during preprocessing.
#      Eliminates the need to touch the dataset at all for batch planning.
#    - Also missing: edge counts in budget calculation. Current O(nodes)
#      cost model under-estimates dense graphs. Adding O(nodes+edges)
#      would tighten the budget and reduce OOM risk.
#
# 3. Pre-sorting for blob slicing:
#    - If graphs in the cache are sorted by node count (or by node-budget
#      bin), then NodeBudgetBatchSampler's output tends to be contiguous
#      ranges in the blob. Contiguous = single tensor slice instead of
#      N separate() calls.
#    - v8.0.0 cache is already sorted by node count (for mmap locality).
#      This means batch plans from the bucket-shuffle sampler produce
#      near-contiguous ranges. A "defrag" pass that reorders the cache
#      to match a reference batch plan would make blob slicing exact.
#
# 4. TensorDict / nested tensors:
#    - torch.nested.nested_tensor (PyTorch 2.8+) can represent variable-
#      length sequences as a single tensor object. Could replace the
#      separate()/collate() dance for node features (x) — store all graphs
#      as a NestedTensor, slice by graph index, batch by concatenation.
#    - torch.utils._pytree + TensorDict (from torchrl) could decompose
#      the Batch into a structured dict of tensors with explicit size
#      tracking, avoiding the PyG-specific _slice_dict/inc_dict machinery.
#    - Risk: PyG's message passing assumes Batch format (batch vector,
#      edge_index with global node IDs). Switching to NestedTensor would
#      require changes in the model forward pass. Evaluate compatibility
#      before adopting.
#
# 5. Zero-padding for fixed batch shapes:
#    - Pad each batch to max_node_budget nodes with x=0, no edges.
#    - Pros: fixed tensor shapes → torch.compile friendly, no recompilation,
#      predictable VRAM, curriculum can reuse pre-padded batches.
#    - Cons: wasted FLOPs on padding nodes (attention over padding is
#      still O(N²) for GPS, O(E) for GAT where E=0 for padding).
#      Need a padding mask to exclude from loss and pooling.
#    - This is essentially what the NLP world does with sequence padding.
#      GNNs have an advantage: padding nodes with no edges contribute
#      zero messages, so GATv2/GCN skip them naturally. Only global
#      pooling (mean/sum) needs the mask.
#
# ==========================================================================
# IMPLEMENTATION ORDER
# ==========================================================================
#
# Step 1: Persist node_budget lookup table from probe-budget runs.
#         (budget.py writes CSV already — wire it into train_dataloader
#         as a fallback when live probe isn't available)
#
# Step 2: PreBatchedDataset for standard (non-curriculum) path.
#         Build pre-batched list in first train_dataloader() call.
#         Measure wall-clock improvement on SLURM.
#
# Step 3: Per-graph metadata sidecar (num_nodes, num_edges tensor)
#         written during preprocessing. Eliminates dataset access for
#         batch planning entirely.
#
# Step 4: Profile curriculum path. If re-collation per epoch is the
#         bottleneck, implement difficulty-tier batching (coarse
#         curriculum at batch granularity).
#
# Step 5: Evaluate zero-padding. Prototype fixed-shape batches,
#         measure wasted FLOPs vs compilation + memory benefits.
#         This may be the curriculum fix — pre-pad once, mask per epoch.
#
# Step 6: Evaluate blob slicing / NestedTensor. Only if Steps 2-3
#         leave significant collation overhead.
"""
