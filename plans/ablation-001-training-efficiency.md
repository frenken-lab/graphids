# Ablation Run 001 — Training Efficiency Issues

> Researched: 2026-03-24 | Audited: 2026-03-30
>
> Research findings from 69-job ablation run. Implementation references are against the
> current codebase (`graphids/core/preprocessing/datamodule.py`, `graphids/config/`).

## Issue 1: VRAM Underutilization — batch_size, not percentile

`vram_node_budget()` (`datamodule.py:80`) computes `budget = batch_size * p95_nodes`.
Non-GPS small models peak 4-6 GB on 16 GB V100 (33-42%).

**Root cause:** `batch_size=4096` is too small for ~100K-param models, not the p95 multiplier.
With low CV (p95/mean ~1.2 for CAN bus graphs), p95 adds only ~19% headroom over mean.

**Recommendation:** Increase `batch_size` to 8192 for small VGAE/GAT/DGI. Keep p95 as budget
percentile — switching to p99 risks near-OOM for negligible throughput gain.

## Issue 2: GPS OOM — O(N^2) global attention

`GPSConv` with `attn_type="multihead"` computes full N x N attention across ALL nodes in the
PyG mega-graph batch. `_make_conv()` in `_conv.py:90` wires this.

Evidence: batch of 155K nodes → 48.5 GB attention matrix alone. Immediate first-batch OOM,
not a leak.

| batch_size | Budget (nodes) | Attempted alloc | Dataset |
|-----------|----------------|-----------------|---------|
| 4096 | 155,648 | 105 GB | set_01 |
| 4096 | 184,320 | 169 GB | set_02 |
| 512 | 19,456 | 3.9 GB | set_01 |

**Recommendation:** GPS-specific batch_size cap (~256-384) for V100. Safe N_max ~15-20K nodes.
Follow-up: test `attn_type="performer"` for O(N) memory (GPSConv supports it natively).

## Issue 3: Data Staging Bottleneck

Current: `cp -r` copies entire 86 GB cache/ to TMPDIR per-job. Ablation CPU eval jobs timed
out at 30 min.

**Recommendation:**
1. **Dataset-scoped staging** — copy only the needed dataset (4-6 GB, not 86 GB). ~15-30 sec.
2. **Skip TMPDIR for CPU eval** — read directly from Scratch GPFS. Zero copy time.
3. **Parallel ESS→Scratch** — `xargs -P` per-dataset rsync for initial staging.

## Implementation status

These findings fed into ablation run 004 (`plans/ablation-run-004-failures.md`):
- RAM bumped to 36G (Issue 1 related)
- GPS batch_size right-sizing still needed
- Dataset-scoped staging not yet implemented
- `vram_node_budget()` shown to be model-blind (Issue 3 in run 004)

## Open questions

1. Per-dataset batch_size for GPS: single conservative 256, or dataset-conditional?
2. Performer attention quality on small CAN bus graphs (~30-50 nodes)?
3. Scratch cache cleanup: 64 GB of stale versioned dirs (v3-v7) could be deleted.
