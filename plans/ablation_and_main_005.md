# Run 005 — Ablation + Main Results (2026-03-31)

Cluster: Ascend (A100-PCIE-40GB), partition: nextgen
Recipes: `main_results.yaml` (6 datasets x 2 configs) + `ablation.yaml` (set_01 x 18 configs)
Totals: 22 COMPLETED, 6 OUT_OF_MEMORY, 7 FAILED, 1 RUNNING

## Main Results

| Dataset | large autoencoder | large curriculum | small KD autoencoder |
|---------|-------------------|------------------|----------------------|
| hcrl_ch | COMPLETED (17m) | COMPLETED (35m) | COMPLETED (14m) |
| hcrl_sa | COMPLETED (13m) | COMPLETED (7m) | COMPLETED (10m) |
| set_01 | COMPLETED (prior) | COMPLETED (3h08m) | COMPLETED (1h39m) |
| set_02 | COMPLETED (3h05m) | COMPLETED (3h26m) | FAILED (wall timeout 1h55m) |
| set_03 | OOM x3 (exhausted retries) | blocked | blocked |
| set_04 | OOM x3 (exhausted retries) | blocked | blocked |

Curriculum, fusion, and KD fusion columns not yet reached for most datasets.

## Ablation (set_01 only)

| Stage | Status |
|-------|--------|
| Autoencoders (all variants) | COMPLETED |
| Normals (3 conv variants) | COMPLETED |
| Curricula (6 loss x curriculum variants) | COMPLETED |
| Fusions (5 jobs) | ALL FAILED — IsADirectoryError |

## Failures & Fixes (all applied)

| # | Failure | Root cause | Fix |
|---|---------|-----------|-----|
| 1 | Fusion `IsADirectoryError` | `pipeline.yaml` only declared `curriculum` as GAT dep. Normal-only configs got empty `gat_ckpt_path` → `Path("")` → CWD | Added `normal` as alternative GAT source in `pipeline.yaml`, dedup in `component.py`, guard in `datamodule.py` |
| 2 | set_03/set_04 OOM during preprocessing | 3x memory copies in `InMemoryDataset.process()` (bulk tensors → per-window Data list → collate back) | Zero-copy collation: bulk tensors are the collated format directly. Peak 1x (was 3x) |
| 3 | KD autoencoder wall timeout set_02 | 2h allocation insufficient | Wall times increased +2h in `resources.yaml` |

## GPU Metrics — Working

Both original "broken" diagnoses were wrong:
- **CSVLogger**: GPU stats ARE written — sparse rows (separate from model metrics, same step). Merge: `pd.read_csv(...).groupby('step').first()`
- **wandb on Ascend**: Full GPU metrics present on all non-trivial runs. Short-lived fusion failures (<1min) had no samples.

## What Worked

- Multiprocess executor: independent assets launched in parallel correctly
- Skip logic: prior `.complete` markers honored, no redundant training
- Small datasets (hcrl_ch, hcrl_sa): full pipeline through curriculum completed for both configs

## Relaunch Notes

All fixes applied. Skip logic will skip 22 completed jobs. set_03/set_04 will run `process()` for first time with zero-copy path. Existing set_01/set_02 caches are format-compatible.
