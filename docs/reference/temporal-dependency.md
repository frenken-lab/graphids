Yes. The scan decomposes cleanly into a dependency graph. The key is that **not all 7 operations are equally “sequential.”** Some are causal across time, but many are only sequential because the current Python/PyTorch loop launches them one by one.

For one timestep, the actual order is:

```text
1. memory_context
2. rhythm_context
3. motif_context
4. backbone
5. memory_update
6. rhythm_update
7. motif_update
```

**Dependency Graph**

```text
inputs at t:
  encoded_t, src_t, dst_t, iat_t, time_t
  memory_state[t], last_seen[t], rhythm_state[t], motif_state[t], backbone_state[t]

memory_context ┐
rhythm_context ├── x_t = encoded_t + contexts ──> backbone ──> z_t
motif_context  ┘

z_t ──> memory_update ──> memory_state[t+1], last_seen[t+1]
iat_t ─> rhythm_update ─> rhythm_state[t+1]
dst_t, iat_t ─> motif_update ─> motif_state[t+1]
z_t ──> backbone_state[t+1]
```

So:

| Operation        |      Can Vectorize Across Lanes? |            Can Run In Parallel With Others? | Depends On                   |
| ---------------- | -------------------------------: | ------------------------------------------: | ---------------------------- |
| `memory_context` |                              yes |              yes, with rhythm/motif context | memory rows, last_seen, time |
| `rhythm_context` |                              yes |              yes, with memory/motif context | rhythm rows                  |
| `motif_context`  |                              yes |             yes, with memory/rhythm context | lane motif state             |
| `backbone`       | yes across lanes, no across time |                          after all contexts | summed `x_t`, backbone state |
| `memory_update`  |                 yes across lanes |                              after backbone | `z_t`, memory rows           |
| `rhythm_update`  |                 yes across lanes |            after or alongside memory update | `iat_t`, rhythm rows         |
| `motif_update`   |                 yes across lanes | can happen before/alongside backbone/update | `dst_t`, `iat_t`             |

The shared context is substantial. Right now the loop computes common indexing once:

```python
lane_idx
src_idx
dst_idx
flat_src = lane_idx * num_ids + src_idx
flat_dst = lane_idx * num_ids + dst_idx
same_id = src_idx == dst_idx
```

But then memory, rhythm, motif each do their own separate work and launch their own ops. A combined scan step would use that shared indexing once and gather all needed state together:

```text
flat_src / flat_dst
  -> memory src/dst rows
  -> last_seen src/dst values
  -> rhythm src/dst rows
lane_idx
  -> motif ids/iats
```

Then it would produce one combined context:

```text
context_t =
    memory_context(src_mem, dst_mem, src_last, dst_last, time)
  + rhythm_context(src_rhythm, dst_rhythm)
  + motif_context(motif_ids, motif_iats)

x_t = encoded_t + context_t
```

That is the first obvious fusion target: **fused context read/construction**.

The second is **fused post-backbone state update**:

```text
given z_t, iat_t, dst_t, same_id:
  update memory src/dst
  update last_seen src/dst
  update rhythm src/dst
  shift/update motif
  write backbone state
```

These updates write different state tensors, so they are mostly independent once `z_t` exists.

**What Is Truly Custom**
The custom part is not the linear layers themselves. PyTorch/cuBLAS is good at those.

The custom part is the causal indexed state machine:

```text
for each timestep:
  gather per-lane src/dst state
  compute context
  run small recurrent cell
  scatter updated src/dst state back
  preserve read-before-write semantics
  handle src == dst correctly
```

That is awkward for regular PyTorch because it is many small gather/scatter/elementwise ops inside a short recurrent loop. This is exactly the kind of thing where a custom Triton/CUDA kernel can help, but only if we are careful.

A plausible custom boundary is one function:

```python
scan_chunk_fused(
    encoded, src_idx, dst_idx, t, iat, valid,
    memory_state, last_seen, rhythm_state, motif_ids, motif_iats, backbone_state,
    weights,
) -> features, next_states
```

But full training makes this hard because it needs backward/autograd. A more realistic first custom kernel is narrower:

```python
fused_context_and_state_gather(...)
```

or

```python
fused_rhythm_motif_update(...)
```

The best candidate for Triton/CUDA is the **elementwise/indexed state work**, not the dense projections. For example:

- gather memory/rhythm/last_seen rows using shared `flat_src/flat_dst`;
- compute rhythm features;
- update rhythm stats;
- shift motif state;
- update last_seen;
- maybe perform memory GRU elementwise math if hidden size is small/fixed.

The less attractive custom-kernel pieces are:

- `Linear`;
- `LayerNorm`;
- `GELU`;
- embedding lookup;
- general GRU/SSM projection matmuls.

Those are better left to PyTorch unless we commit to a fully fused small-hidden custom op with custom backward.

**Making State Block-Parallel**
Current state is sequential because event `t+1` may read an ID that event `t` just updated. To make it block-parallel, you need one of these compromises:

1. **ID-conflict-free blocks**
   Partition a chunk into sub-blocks where no `(src,dst)` ID conflicts occur within the block. Events inside a block can run in parallel; blocks remain sequential.
   Tradeoff: complex scheduler, variable block sizes, may not help if IDs repeat often.

2. **Lagged state reads**
   Read all contexts from state at block start, compute several timesteps in parallel, apply updates after the block.
   Tradeoff: changes model semantics. Faster but no longer strictly causal within the block.

3. **Associative state formulation**
   Replace updates with operations that compose associatively, so a parallel prefix scan can compute states.
   Tradeoff: requires redesigning memory/rhythm/backbone updates. Current GRU-style memory update is not naturally associative.

4. **Per-stream independent backbone only**
   Run backbone sequence per lane with `forward_sequence`, but keep ID memory/rhythm/motif outside or simplified.
   Tradeoff: works only if cross-ID state modules are disabled or delayed.

5. **Two-level model**
   Use cheap block-parallel features inside a chunk, then apply a smaller causal correction at block boundaries.
   Tradeoff: bigger modeling change, needs accuracy validation.

My recommendation: do not start with full block-parallel redesign. First, build and benchmark two implementation-level fusions:

```text
A. fused context builder:
   one shared gather path for memory/rhythm/motif context

B. fused update writer:
   one shared update path for memory/last_seen/rhythm/motif writes
```

Then benchmark module ablations. If rich-state modules are still necessary and scan remains dominant, only then consider a Triton kernel for the indexed state machine.

## Implementation Check: September 30, 2026

The current lane scan has already implemented the first layer of this plan:

- `_lane_step_context` computes shared `flat_src`/`flat_dst` row indices once per
  timestep;
- the `lane_fastpath` backend uses flattened lane-ID addressing for memory,
  last-seen, and rhythm state;
- gathered memory/rhythm rows are retained on `_LaneStepGather` and reused by
  `_write_lane_step_updates`;
- lane memory/rhythm/motif/backbone state is cloned once per chunk, not once per
  timestep;
- batched memory and rhythm helpers handle source/destination row updates.

That means the remaining target is not generic "add lane batching." It is the
smaller hot path left inside the 32-step recurrent loop:

```text
per step:
  gather memory/last_seen/rhythm/motif rows
  run three small context builders
  run backbone.step
  run memory/rhythm/motif updates
  write state rows back
```

The strict rich phase probe measured `scan = 108.25 ms` inside
`forward = 110.10 ms`; subphase timing was:

| Subphase | Time |
| --- | ---: |
| memory context | `18.21 ms` |
| memory update | `10.39 ms` |
| rhythm context | `19.01 ms` |
| rhythm update | `14.62 ms` |
| motif context | `12.66 ms` |
| motif update | `6.05 ms` |
| backbone | `12.78 ms` |

Clear next targets:

1. Fuse the rich context path more aggressively. The code shares row indices,
   but memory, rhythm, and motif still execute as separate gather/projection
   paths each timestep.
2. Fuse update/writeback after `z_t`. Memory, last-seen, rhythm, motif, and
   backbone updates write different state tensors, so the dependency barrier is
   `backbone.step`, not each individual writer.
3. Remove motif-update cloning in the lane path or replace it with a fixed-size
   rolling-buffer representation.
4. Treat rhythm as the first custom-kernel candidate if PyTorch-level fusion
   stalls. Its math is simple, but context plus update still costs about
   `33.6 ms`.
5. Avoid full block-parallel redesign until these implementation fusions are
   exhausted; strict causality across repeated IDs is still the main semantic
   constraint.

### Working-Tree Follow-up

The first implementation pass has been applied locally:

- precomputed `flat_src_all` / `flat_dst_all` for lane-fastpath backends;
- one per-chunk `full_chunk_fastpath` check instead of per-step `active.all()`;
- no scan subphase timing helper calls when subphase profiling is disabled;
- in-place motif history shifts rather than per-step active-row clones.

Verification:

```text
uv run pytest tests/core/models/test_temporal_hybrid.py -q
31 passed, 4 warnings

uv run ruff check graphids/core/models/temporal/_hybrid/scan.py tests/core/models/test_temporal_hybrid.py
All checks passed
```

This preserves current strict causal semantics. It still needs a strict
`16 lanes x 32 chunk` benchmark before it should be treated as a new speed
baseline.



  ## Summary
  Reduce temporal hybrid scan time by targeting the real hot path: repeated per-timestep context reads, small tensor ops, and indexed state updates inside `graphids/core/models/temporal/_hybrid/
  scan.py`. Do not pursue lane widening or input pinning as primary fixes. The current `set_01` train split only supports 18 streams, so `16 lanes x 32 chunk` remains the valid strict diagnostic
  geometry.

  The goal is to preserve current rich-model semantics while reducing scan overhead through two staged implementation passes:

  1. Fuse context gathering/construction across memory, rhythm, and motif.
  2. Fuse state update/writeback across memory, last_seen, rhythm, motif, and backbone state where dependencies allow.

  Only consider Triton/CUDA after measuring that Python/PyTorch gather/scatter overhead remains dominant after these fusions.

  ## Key Changes

  ### 1. Establish Scan Dependency Boundaries
  Use the current 7 scan operations as the optimization contract:

  | Phase | Depends On | Can Run Across Lanes | Can Run With |
  | --- | --- | ---: | --- |
  | `memory_context` | memory rows, last_seen, time | yes | rhythm/motif context |
  | `rhythm_context` | rhythm rows | yes | memory/motif context |
  | `motif_context` | lane motif state | yes | memory/rhythm context |
  | `backbone` | summed context input, backbone state | yes across lanes, sequential over time | after contexts |
  | `memory_update` | `z_t`, memory rows | yes | rhythm/motif update |
  | `rhythm_update` | `iat_t`, rhythm rows | yes | memory/motif update |
  | `motif_update` | `dst_t`, `iat_t` | yes | can be prepared before/after backbone |

  Keep strict causal semantics: contexts at timestep `t` must read pre-update state; updates from timestep `t` become visible at `t+1`.

  ### 2. Add a Fused Context Builder
  Create a helper used inside the lane scan loop that performs shared indexing once per timestep:

  ```python
  flat_src = lane_idx * num_ids + src_idx
  flat_dst = lane_idx * num_ids + dst_idx
  same_id = src_idx == dst_idx
  ```

  The helper should gather all context inputs in one place:

  - memory src/dst rows from `memory_flat` or `memory_state`;
  - last_seen src/dst values;
  - rhythm src/dst rows from `rhythm_flat` or `rhythm_state`;
  - motif ids/iats from `motif_ids[lane_idx]` and `motif_iats[lane_idx]`.

  It should return:

  ```python
  context_t: Tensor  # summed memory + rhythm + motif context, shape [active_lanes, hidden]
  gathered: lightweight object/dict containing rows/indices needed later by update
  ```

  Do not change module math in this pass. Continue using existing `read_context_rows_batch`, rhythm projection, motif embedding/time/proj, and memory mix. The immediate win is fewer duplicated
  gathers, fewer temporary branches, and one clear context pathway.

  ### 3. Add a Fused Update Writer
  After `z_t = backbone(x_t, state)`, use the gathered context object from step 2 to update all mutable states through one helper.

  The helper should:

  - update memory src/dst rows using the selected backend behavior:
    - `lane_fastpath`: current two-GRUCell logic;
    - `fused_memory`: `update_src_dst_batch`;
    - `compiled_inner`: `compiled_update_src_dst_batch`;
  - update `last_seen` for src/dst;
  - update rhythm src/dst rows with `_updated_rows_indexed_batch`;
  - shift/write motif ids/iats for each active lane;
  - write updated rows back through `memory_flat`, `last_seen_flat`, and `rhythm_flat` when available.

  Keep the current `src == dst` semantics exactly:
  - memory destination update reads `src_new` when `same_id`;
  - rhythm destination update reads `src_new` when `same_id`.

  ### 4. Add Focused Ablation Diagnostics
  Add configs, not defaults, for strict `16 lanes x 32 chunk` diagnostics:

  - `rich`: current best `lane_fastpath` reference.
  - `minus_motif`: disable motif only.
  - `minus_rhythm`: disable rhythm only.
  - `minus_memory`: disable TGN memory only.
  - `memory_only`: memory enabled, rhythm/motif disabled.
  - `backbone_only`: memory/rhythm/motif disabled, same backbone.
  - `backbone_none`: no memory/rhythm/motif/backbone, heads only sanity probe.

  Use:

  ```yaml
  batch_mode: stream_lanes
  stream_lanes: 16
  chunk_size: 32
  batch_size: 512
  drop_partial_lane_batches: true
  model.scan.backend: lane_fastpath
  num_workers: 0
  prebuild_batches: 0
  pin_memory: false
  expected_train_events: 512
  fail_on_event_mismatch: true
  ```

  Run two kinds of measurements:

  - Throughput configs without subphase timing for events/sec.
  - Phase/subphase configs only for the best or most informative variants.

  ### 5. Triton/CUDA Gate
  Do not start with Triton/CUDA. Promote it only if fused PyTorch still shows scan dominated by indexed state work.

  Candidate custom-kernel boundary:

  ```python
  fused_indexed_state_step(
      encoded_t,
      src_idx,
      dst_idx,
      iat_t,
      t_t,
      lane_idx,
      memory_state,
      last_seen,
      rhythm_state,
      motif_ids,
      motif_iats,
      backbone_state,
      weights_or_preprojected_inputs,
  ) -> context_or_z_t, updated_state_parts
  ```

  The first realistic custom target should be narrower than full scan:

  - shared gather of memory/rhythm/last_seen rows;
  - rhythm feature/update math;
  - motif shift/update;
  - last_seen writeback;
  - maybe memory GRU elementwise update if hidden size is fixed and backward strategy is clear.

  Avoid custom kernels for dense projections initially:
  - Linear;
  - LayerNorm;
  - GELU;
  - embedding lookup;
  - general GRU/SSM projection matmuls.

  Those should stay in PyTorch/cuBLAS unless a later benchmark proves launch overhead dominates them.

  ### 6. Block-Parallel State Research Track
  Treat block-parallel state as a model-design experiment, not an implementation optimization.

  Evaluate only after fused scan work:

  - **ID-conflict-free blocks:** parallelize events in a block only when their src/dst IDs do not conflict. Preserves semantics, but scheduling may be complex and block sizes may be small.
  - **Lagged block reads:** read state once at block start, compute several timesteps in parallel, apply updates after block. Fast, but changes causal semantics.
  - **Associative state redesign:** replace GRU/motif/rhythm updates with composable operations for prefix-scan style parallelism. Large modeling change.
  - **Backbone-only block parallelism:** use `forward_sequence` when ID memory/rhythm/motif are disabled or delayed. Useful as an ablation, not a rich-model replacement.

  Do not mix block-parallel changes with fusion benchmarking; they answer different questions.
