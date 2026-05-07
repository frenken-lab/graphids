# 2026-05-06 — Drop `neighborhood_decoder`, swap in TAM affinity

**Shipped:** commit `1eafb7f`. Current code: `graphids/core/models/autoencoder/vgae.py:165`.

## TAM (Qiao & Pang, *Truncated Affinity Maximization*, NeurIPS 2023)

Closed-form, training-free per-node anomaly score:

```python
def neighbor_affinity(z, edge_index):
    """Per-node 1 - mean cosine similarity to neighbors. High = anomalous."""
    src, dst = edge_index
    sim = F.cosine_similarity(z[src], z[dst], dim=-1)              # [E]
    per_node = scatter(sim, src, reduce='mean', dim_size=z.size(0))  # [N]
    return 1 - per_node                                             # [N]
```

Per-node, vocab-free, computable on the `z` the encoder already produces. >10 AUROC over reconstruction baselines across 10 datasets.

## Why `neighborhood_decoder` was wrong

1. Workaround for classical Kipf adjacency reconstruction being random under chain-windowed topology (98 deterministic edges, nothing to predict).
2. Vocab dependency — UNK cliff on OOD; learns arbitrary `embedding(ID_X) → bag-of-neighbor-IDs` lookup with no semantic prior.
3. One of two 1791-dim matmuls that saturated V100 fp32 (see prior log).
4. Uniform output when attacks are in-vocab — set_04, the case we most need signal on.

## Final score (post-1eafb7f)

```
score(G) = max-σ(recon, recon_max, affinity, rq)
```

4 calibrated z-norm axes. `mahal`/`KL` retained in `extract_features['errors'][N, 3]` for fusion's frozen reward weight but no longer scored. Forward returns 6-tuple `(cont_out, canid_logits, nbr_pred, z, kl_per_node, edge_logits)` — position 2 is now GAD-NR continuous neighbor-mean prediction `[N, latent_dim]` (not vocab-bag); see `kl_neighbor_loss` in `graphids/core/losses/autoencoder.py` for the standard-convention KL (upstream pygod / GAD-NR notebook have the sign inverted).
