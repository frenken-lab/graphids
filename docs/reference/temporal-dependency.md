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

## Current Lane Scan Path

Lane-batched `TemporalStreamLaneBatch` inputs now use one invariant rich scan
implementation whenever `valid_mask` is present. The path uses flattened
lane/ID row indices for memory, last-seen, and rhythm state, keeps partial
active-lane handling, and performs state writes through `_write_lane_step_state`
after `backbone.step`. Plain non-lane `TemporalData` scans remain supported for
compatibility and reference tests, but they are not a separate optimized backend.

The scan still follows strict causal ordering:

```text
per step:
  gather memory/last_seen/rhythm/motif rows
  build memory, rhythm, and motif context from pre-update state
  run backbone.step
  write backbone, memory, last_seen, rhythm, and motif state updates
```

Subphase profiling is controlled only by `model.scan.profile_subphases` and is
useful when phase timing is also enabled. The current stable subphase names are:

- `memory_context`
- `rhythm_context`
- `motif_context`
- `backbone`
- `backbone_write`
- `memory_update`
- `rhythm_update`
- `motif_update`

The promoted strict-rich implementation comes from the writer-fusion diagnostic
that preserved `16 lanes x 32 chunk`, `512` train events, and zero padding. Do
not reintroduce a public scan-backend matrix for throughput experiments; add
focused diagnostic configs around profiling or feature ablations instead.

## Remaining Optimization Targets

The lane scan has already implemented the first layer of fusion:

- `_lane_step_context` computes shared `flat_src`/`flat_dst` row indices once per
  timestep.
- Flattened lane-ID addressing is used for memory, last-seen, and rhythm state.
- Gathered memory/rhythm rows are retained on `_LaneStepGather` and reused by
  `_write_lane_step_state`.
- Lane memory/rhythm/motif/backbone state is cloned once per chunk, not once per
  timestep.
- Motif history rolls in place for active lanes.

The remaining hot path is the small sequential per-timestep loop. Keep future
experiments focused on preserving strict causal semantics while reducing context
construction and indexed state-write overhead. Consider Triton/CUDA only after
PyTorch-level fusion is exhausted and a fresh strict benchmark still shows scan
time dominated by indexed state work.
