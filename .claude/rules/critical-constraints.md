# GraphIDS Critical Constraints

These fix real crashes — DO NOT VIOLATE:

- **PyG `Data.to()` is in-place.** Always `.clone().to(device)`, never `.to(device)` on shared data.
- **Use spawn multiprocessing.** Never `fork` with CUDA. Set `mp_start_method='spawn'` and `multiprocessing_context='spawn'` on all DataLoaders.
- **NFS filesystem.** `.nfs*` ghost files appear on delete. Already in `.gitignore`.
- **No GUI on HPC.** Git auth via SSH key, not HTTPS.
- **Never run pytest on login nodes.** Always submit via `scripts/slurm/submit.sh tests`.
- **Dual-budget node+edge sampling.** The sampler closes a batch when adding a graph would exceed EITHER `max_nodes` OR `max_edges` (probed jointly by `BudgetProfiler.probe`). Single-axis node-only budgets allowed edge-heavy batches to OOM. Live sampler walks indices; `pack_offline` (FFD) is used at prebatch time for ~10-20% tighter packing.
- **Save/restore model.training state.** Always save and restore `model.training` around `model.eval()` calls in utility functions — leaking eval mode silently disables dropout/batchnorm.
- **Clamp statistical moments to ±10.** Skewness/kurtosis features can reach 1e17, causing MSE loss overflow in fp16 (max ~65504).
- **VRAMDriftCallback warns, doesn't act.** Logs a one-shot warning when free VRAM shrinks past `GRAPHIDS_VRAM_DRIFT_THRESHOLD` (default 0.20) between epochs. The probe is a baseline at fit-start; mid-run re-probing would race optimizer state. Researcher decides whether to abort.
