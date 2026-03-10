# KD-GAT Critical Constraints

These fix real crashes — DO NOT VIOLATE:

- **PyG `Data.to()` is in-place.** Always `.clone().to(device)`, never `.to(device)` on shared data.
- **Use spawn multiprocessing.** Never `fork` with CUDA. Set `mp_start_method='spawn'` and `multiprocessing_context='spawn'` on all DataLoaders.
- **NFS filesystem.** `.nfs*` ghost files appear on delete. Already in `.gitignore`.
- **No GUI on HPC.** Git auth via SSH key, not HTTPS.
- **Never run pytest on login nodes.** Always submit via `bash scripts/slurm/run_tests_slurm.sh`.
- **Dynamic batching for variable-size graphs.** DynamicBatchSampler packs graphs to a node budget instead of a fixed count. Budget = batch_size × p95_nodes. Disable with `-O training.dynamic_batching false` for reproducibility.
- **Save/restore model.training state.** Always save and restore `model.training` around `model.eval()` calls in utility functions — leaking eval mode silently disables dropout/batchnorm.
- **Clamp statistical moments to ±10.** Skewness/kurtosis features can reach 1e17, causing MSE loss overflow in fp16 (max ~65504).
