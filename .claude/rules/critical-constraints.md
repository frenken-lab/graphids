# GraphIDS Critical Constraints

These fix real crashes — DO NOT VIOLATE:

- **PyG `Data.to()` is in-place.** Always `.clone().to(device)`, never `.to(device)` on shared data.
- **Use spawn multiprocessing.** Never `fork` with CUDA. `mp_start_method='spawn'` and `multiprocessing_context='spawn'` on all DataLoaders.
- **NFS filesystem.** `.nfs*` ghost files appear on delete. Already in `.gitignore`. No GUI on HPC — git auth via SSH key, not HTTPS.
- **Dual-budget node+edge sampling.** Sampler closes a batch when adding a graph would exceed EITHER `max_nodes` OR `max_edges` (probed jointly by `budget.probe`). Single-axis budgets allowed edge-heavy batches to OOM. `pack_offline` (FFD) used at prebatch time for ~10-20% tighter packing.
- **Two-point probe for `bpn_node` / `bpn_edge`.** `budget.probe` runs fwd+bwd at 2k and 20k nodes and takes the slope of peak-vs-nodes. The single-point probe it replaced inflated bpn ~3-4× by charging small batches with fixed overhead, capping packs at ~20% of real H100 VRAM. Keep both probe points. `GRAPHIDS_BUDGET_SAFETY_MARGIN=0.95` leans on the slope-fit's intercept to cover fixed costs — do NOT drop below 0.90 without restoring a resident-subtract pathway.
- **Save/restore `model.training` state** around `model.eval()` calls in utility functions — leaking eval mode silently disables dropout/batchnorm.
- **Clamp statistical moments to ±10.** Skewness/kurtosis can reach 1e17, overflowing fp16 (max ~65504).
- **Clamp VGAE `logvar` to ±10.** `exp(logvar)` in the KL term must stay inside fp16 max under `precision: 16-mixed`. The previous ±20 bound (`exp(20) ≈ 4.85e8`) overflowed and produced first-epoch val NaN.
- **VRAMDriftCallback warns, doesn't act.** One-shot warning when free VRAM shrinks past `GRAPHIDS_VRAM_DRIFT_THRESHOLD` (default 0.20) between epochs. Probe is a fit-start baseline; mid-run re-probing would race optimizer state.
