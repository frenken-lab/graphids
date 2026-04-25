# GraphIDS Critical Constraints

These fix real crashes — DO NOT VIOLATE:

- **PyG `Data.to()` is in-place.** Always `.clone().to(device)`, never `.to(device)` on shared data.
- **Use spawn multiprocessing.** Never `fork` with CUDA. Set `mp_start_method='spawn'` and `multiprocessing_context='spawn'` on all DataLoaders.
- **NFS filesystem.** `.nfs*` ghost files appear on delete. Already in `.gitignore`.
- **No GUI on HPC.** Git auth via SSH key, not HTTPS.
- **Never run pytest on login nodes.** Always submit via `python -m graphids submit --mode cpu --command "python -m pytest"`.
- **Dual-budget node+edge sampling.** The sampler closes a batch when adding a graph would exceed EITHER `max_nodes` OR `max_edges` (probed jointly by `budget.probe`). Single-axis node-only budgets allowed edge-heavy batches to OOM. Live sampler walks indices; `pack_offline` (FFD) is used at prebatch time for ~10-20% tighter packing.
- **Two-point probe for `bpn_node` / `bpn_edge`.** `budget.probe` runs fwd+bwd at two batch sizes (2k + 20k nodes) and takes the slope of peak-vs-nodes; the single-point probe it replaced charged small batches with fixed overhead (cuDNN workspaces, optimizer state, KD teacher), inflating bpn by ~3-4× and capping packs at ~20% of real VRAM on H100. Keep both probe points. `GRAPHIDS_BUDGET_SAFETY_MARGIN=0.95` leans on the slope-fit's implicit intercept to cover fixed costs — do NOT drop it below 0.90 without restoring a resident-subtract pathway.
- **Save/restore model.training state.** Always save and restore `model.training` around `model.eval()` calls in utility functions — leaking eval mode silently disables dropout/batchnorm.
- **Clamp statistical moments to ±10.** Skewness/kurtosis features can reach 1e17, causing MSE loss overflow in fp16 (max ~65504).
- **Clamp VGAE `logvar` to ±10.** `exp(logvar)` in the KL term must stay inside fp16's max (~65504) under `precision: 16-mixed`; `exp(10) ≈ 22026` is safe, the previous ±20 bound (`exp(20) ≈ 4.85e8`) overflowed fp16 and produced NaN in the first validation epoch (jobs 46959925, 46962244).
- **VRAMDriftCallback warns, doesn't act.** Logs a one-shot warning when free VRAM shrinks past `GRAPHIDS_VRAM_DRIFT_THRESHOLD` (default 0.20) between epochs. The probe is a baseline at fit-start; mid-run re-probing would race optimizer state. Researcher decides whether to abort.
