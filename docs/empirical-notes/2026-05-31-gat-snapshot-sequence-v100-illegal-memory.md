# 2026-05-31 — GAT snapshot_sequence V100 illegal memory access

## Trigger

`gat_snapshot_sequence_real` on `set_01`, Pitzer V100, after switching the GAT
input representation from `snapshot` to `snapshot_sequence`.

Failed jobs:

- `47913502`: failed after `00:03:27`.
- `47919702`: failed after PyG CUDA wheel alignment, after `00:06:02`.

Both surfaced as:

```text
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
graphids/core/models/supervised/gat.py:260
edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
```

## What changed with snapshot_sequence

`snapshot` emits one graph per window. `snapshot_sequence` emits one training
sample from several ordered snapshots by offsetting each step's node IDs and
joining the steps into one disconnected PyG graph. The materializer also adds
node/edge sequence metadata (`node_sequence_step`, `edge_sequence_step`, etc.).

This means the GAT sees larger packed tensors before sequence pooling. On the
failed set_01 run, dynamic batching packed roughly `58k` nodes and `183k` edges
per training batch. Memory was not close to exhaustion: peak GPU memory was
about `5.5 GB` on a 16 GB V100.

The model did not fail on the first forward pass. It logged many successful
train points and one validation epoch before the later training crash:

- `train_loss=0.040147`, `train_acc=0.988636`
- `val_loss=0.124480`, `val_acc=0.949677`, `val_auroc=0.982472`

## Ruled out

- **PyG wheel mismatch:** real issue found and fixed separately
  (`torch 2.8.0+cu128`, PyG native wheels now `+pt28cu128`, `pyg-lib` installed),
  but the crash persisted.
- **Bad cached graph data:** CPU checks over suspect packed batches found edge
  indices in bounds, matching `edge_attr` lengths, finite features, valid
  sequence steps, and successful CPU self-loop rewrites.
- **Activation checkpointing:** disabling `gradient_checkpointing` did not fix
  the crash. `diag_no_checkpoint_manual_loops` failed the same way.
- **Oversized dynamic batches:** same packed-batch scale completed two epochs
  once the manual self-loop rewrite was removed.

## Root cause

The unsafe path was GraphIDS' manual GATv2 self-loop rewrite:

```python
edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
edge_index, edge_attr = add_self_loops(edge_index, edge_attr, fill_value="mean", ...)
```

That rewrite was originally added to precompute self-loops once outside the
checkpointed GATv2 layers. On V100 with large `snapshot_sequence` batches it
intermittently corrupted CUDA state or triggered a PyG/PyTorch indexed-mask
kernel race. The traceback consistently surfaced at `remove_self_loops`, but
CUDA asynchrony made the exact kernel ambiguous.

Diagnostic controls:

- Current manual rewrite + no checkpointing: failed.
- Current manual rewrite + `CUDA_LAUNCH_BLOCKING=1`: completed two epochs.
- PyG native `GATv2Conv(add_self_loops=True)` + normal async CUDA: completed two
  epochs at the same batch scale.

Conclusion: do not manually pre-add GATv2 self-loops for packed
`snapshot_sequence` training on V100. Let PyG handle self-loops inside each
`GATv2Conv`.

## Fix

`graphids/core/models/supervised/gat.py` now keeps PyG's native GATv2 self-loop
handling and no longer runs project-level `remove_self_loops/add_self_loops` in
`GAT.forward`.

`graphids/primitives_models.py` also exposes `gradient_checkpointing` on
`GATCfg` because the diagnostic showed the model class supported the knob but
experiment YAML could not pass it through.

## If this recurs

1. Check PyTorch/PyG wheel suffixes first: all PyG native extensions must match
   `torch.version.cuda`.
2. Reproduce with `CUDA_LAUNCH_BLOCKING=1` only as a diagnostic. It serializes
   CUDA and can hide races, so passing under launch blocking is not proof the
   async training path is safe.
3. Compare manual tensor rewrites against native PyG layer behavior at the same
   packed-batch scale before lowering the batch budget.
