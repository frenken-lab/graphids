# GraphIDS scientific and runtime constraints

- Preserve every observed CAN frame and its physical order, including tied clocks.
  Never merge, drop, pad, or replay scored rows to fit a batch.
- Validate reviewed source provenance and cache identity at the data boundary.
  Never open reserved target recordings through the source-training path.
- Preserve causal history, distinct recording state, label masks, and separately
  normalized classification/payload objectives. Keep bounded kernel workspaces.
- Resume only complete epoch-boundary checkpoints with matching source and model
  identity. Preserve checkpoint hashes and atomic artifact publication on NFS.
- Use spawn multiprocessing with CUDA; do not fork CUDA processes.
- Restore model training mode after evaluation utilities.
- Use one device for the current event data path; chronological streams cannot
  be padded or split across ranks.

See `docs/reference/research-code-style.md` for validation ownership and tests.
