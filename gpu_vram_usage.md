# GPU VRAM Usage — Ablation Study (2026-03-24)

Source: Lightning `DeviceStatsMonitor` metrics from `metrics.csv`.
V100 total VRAM: 16,384 MB.

## Per-Model Summary (best version per model)

| Dataset | Model / Stage | Epoch | Reserved (MB) | Allocated (MB) | Active (MB) | V100 VRAM % |
|---------|---------------|------:|-------------:|---------------:|------------:|------------:|
| set_01 | gat_small_normal | 242 | 8566 | 5508 | 5508 | 52.3% |
| set_01 | vgae_small_autoencoder | 289 | 6408 | 5464 | 5464 | 39.1% |
| set_02 | vgae_small_autoencoder | 0 | 1648 | 1556 | 1556 | 10.1% |

## All Versions (raw)

| Dataset | Model / Stage | Version | Epoch | Reserved (MB) | Allocated (MB) | Active (MB) | V100 VRAM % |
|---------|---------------|---------|------:|-------------:|---------------:|------------:|------------:|
| set_01 | gat_small_normal | version_0 | 239 | 8566 | 5508 | 5508 | 52.3% |
| set_01 | gat_small_normal | version_1 | 219 | 8566 | 5508 | 5508 | 52.3% |
| set_01 | gat_small_normal | version_2 | 242 | 8566 | 5508 | 5508 | 52.3% |
| set_01 | vgae_small_autoencoder | version_0 | 249 | 6408 | 5464 | 5464 | 39.1% |
| set_01 | vgae_small_autoencoder | version_1 | 251 | 6408 | 5464 | 5464 | 39.1% |
| set_01 | vgae_small_autoencoder | version_2 | 289 | 6408 | 5464 | 5464 | 39.1% |
| set_01 | vgae_small_autoencoder | version_3 | 188 | 6268 | 5369 | 5369 | 38.3% |
| set_01 | vgae_small_autoencoder | version_4 | 0 | 816 | 762 | 762 | 5.0% |
| set_01 | vgae_small_autoencoder | version_5 | 0 | 1076 | 1010 | 1010 | 6.6% |
| set_02 | vgae_small_autoencoder | version_0 | 0 | 1648 | 1556 | 1556 | 10.1% |

## Analysis

- **VGAE autoencoder** (set_01): peaks at 6,408 MB reserved (39% of V100). ~940 MB fragmentation gap (reserved - allocated).
- **GAT normal** (set_01): peaks at 8,566 MB reserved (52% of V100). ~3,058 MB fragmentation gap.
- Both models leave 8-10 GB of V100 VRAM unused — dynamic batch sizing could push batch_size higher.
- VGAE versions 4/5 and set_02/version_0 show very low VRAM (<2 GB) — these were OOM-killed runs that
  crashed during graph assembly (host RAM), before GPU training started or shortly after.

## Implications for Dynamic Batch Sampler

Current node budget: `batch_size(6144) × p95_nodes(35) = 215,040` nodes per batch.

With ~8-10 GB free VRAM headroom:
- VGAE could potentially double the node budget (~430K) before hitting VRAM limits.
- GAT has less headroom (~8 GB free) but could still increase ~80%.
- The dynamic batch sampler's budget should be tuned per model type, not uniform.
