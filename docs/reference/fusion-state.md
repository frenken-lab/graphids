# Fusion State

> Last reviewed: 2026-05-09. This page describes the current cached
> fusion representation, not the history of how it was designed.

`graphids` keeps fusion offline. The extract job runs GAT and VGAE once,
writes a frozen feature cache, and the fusion models train on that cache
instead of live encoder outputs.

## Current cache

The default fusion state is a flat 18-dimensional vector assembled by
`flatten_features(td)`.

| Block | Shape | Meaning |
|---|---|---|
| `gat/conf` | 1 | GAT confidence from normalized entropy |
| `gat/emb_stats` | 4 | mean, std, max, min over node embeddings |
| `gat/probs` | 2 | benign / attack probabilities |
| `vgae/affinity` | 1 | mean latent affinity |
| `vgae/conf` | 1 | inverse reconstruction-error confidence |
| `vgae/errors` | 3 | recon, mahal, kl |
| `vgae/rq` | 1 | Rayleigh quotient |
| `vgae/spike` | 1 | max masked-node reconstruction error |
| `vgae/z_stats` | 4 | mean, std, max, min over latent z |

## What this state is good at

- compact supervised fusion
- score baselines such as `weighted_avg`
- RL baselines that need a fixed-size state vector
- fast cache loading and reproducible comparisons

## What it discards

- per-node embedding structure
- layerwise GAT evolution
- per-node VGAE reconstruction structure
- edge-level error distributions
- attention or disagreement patterns between encoders

Those loss modes are the reason the repo still keeps the richer-feature
exploration in `docs/drafts/fusion-improvement-plan.md`.

## Operational takeaway

Treat the 18-dim cache as the stable published interface. If a future
experiment needs more signal, extend the cache deliberately rather than
letting the fusion layer infer structure from a scalar summary.
