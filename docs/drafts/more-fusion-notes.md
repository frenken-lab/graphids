Is the 18-dim scalar summary a lossy bottleneck the fusion model could
exploit? Three options below (A/B/C) referenced from
`docs/drafts/fusion-improvement-plan.md`. Yes — question is *which*
lossiness matters.

### What the 18-dim state discards

| Discarded | From | What's lost |
|---|---|---|
| Per-node embeddings `h_v ∈ ℝ^d` | GAT last layer | Spatial distribution of anomaly — *which nodes* are anomalous |
| JK-pool candidates `{h^(1), ..., h^(K)}` | GAT intermediate layers | Layer-wise feature evolution (local vs global) |
| Latent `z_v ∈ ℝ^d` per node | VGAE encoder | Per-node reconstruction difficulty |
| `z_stats = [mean, std, max, min]` | VGAE | Higher-order moments, multimodality |
| Edge reconstruction scores | VGAE decoder | Which *edges* are anomalous |
| Attention coefficients `e_ij` | GAT | Which node pairs the classifier attended to |

`emb_stats`/`z_stats` are **bag-of-nodes statistics** — `N` vectors to 4
scalars. DeepSets (Zaheer et al., 2017) warns: sufficient for mean, lossy
for higher-order moments, multimodality, spatial structure.

---

## Three Architectures, Increasing Complexity

### Option A: Richer Statistics (Minimal Change)

Test whether the 4-statistic bottleneck is the actual problem before going
to embeddings. Replace `[mean, std, max, min]` with a larger fixed summary:

```python
def richer_stats(h):                                  # h: [N, d]
    return torch.cat([
        h.mean(0), h.std(0), h.max(0).values, h.min(0).values,
        h.kurtosis(0),                                # heavy-tailed
        h.quantile(0.1, dim=0), h.quantile(0.9, dim=0),  # tails
        (h > h.mean(0) + 2*h.std(0)).float().mean(0), # outlier fraction
    ], dim=-1)
```

Pipeline unchanged (frozen extraction → fixed-dim → fusion). State grows
from 18 to ~8d/block; d=64 → ~500-dim, trivial for MLP/DQN.

**Enough when:** anomaly is *global* (all nodes shift together; suppress
may qualify — graph sparser, mean/quantile of `z_v` shift). **Not when:**
anomaly is *spatially localized* (statistics average out the signal).

---

### Option B: Jumping Knowledge Pooling for GAT

**JK-Net** (Xu et al., 2018, ICML) chooses the right aggregation depth
per node. Standard GAT layer K aggregates K-hop neighborhoods, but
optimal radius is attack-dependent: 1-hop for injection on a single ECU,
K-hop for fuzzy attacks. JK keeps intermediates `{h^(1), ..., h^(K)}`:

```
h_v^final = AGGREGATE({h_v^(1), ..., h_v^(K)})    # concat | max-pool | LSTM
```

Each node selects its own depth — graph pooling captures heterogeneous
locality a fixed K-layer GAT cannot. CAN bus: injection on ECU `v`
detectable at 1-hop before propagating; timing attacks need K-hop.

```python
# Extraction: collect intermediates and pool
layer_embs = [h := layer(h, edge_index) for layer in gat.layers]  # [N,d] each
# Concat → [N, K*d]; max-pool stack(...).max(0) → [N,d] (mem-efficient);
# LSTM lstm(stack(...))[0][-1] → [N,d]
jk_graph_emb = h_final.mean(0)             # or attention-weighted
```

Fusion sees `jk_graph_emb ∈ ℝ^d`, not `gat/emb_stats ∈ ℝ^4`. Vector input
lets MLP/attention attend to discriminative dimensions.

**Tradeoff:** stores `K × N × d` per graph (vs `N × d`). Max-pool has
strong empirical performance (Xu et al., 2018).

**Sources:** Xu et al. (2018), *Jumping knowledge networks*, ICML; Xu
et al. (2019), *How powerful are GNNs?*, ICLR (WL expressivity).

---

### Option C: Full Per-Node Embedding Fusion (Graph-Level Attention)

Pass full `H ∈ ℝ^{N×d}` to fusion. Requires permutation-invariance and
variable `N` handling.

```python
class GraphAttentionPool(nn.Module):
    def __init__(self, node_dim, n_heads=4, out_dim=64):
        super().__init__()
        self.attn = nn.MultiheadAttention(node_dim, n_heads, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, node_dim))   # learnable
        self.proj = nn.Linear(node_dim, out_dim)
    def forward(self, H):                                        # [B,N,d]
        q = self.query.expand(H.shape[0], -1, -1)
        pooled, attn_w = self.attn(q, H, H)                      # [B,1,d]
        return self.proj(pooled.squeeze(1)), attn_w              # [B,1,N]
```

`attn_w` is interpretable: high weight on `v` → `v` drove the decision.
Cross-modal `cross_attn(gat_q, vgae_k, vgae_k)` asks *do GAT-anomalous
nodes coincide with VGAE-hard-to-reconstruct nodes?* Spatial agreement is
stronger than the scalar concordance bonus in the current reward, and is
the most novel contribution here.

---

## Extraction Pipeline and Storage

Currently `graphs → GAT+VGAE → 18-dim → fusion_states.pt`; with embeddings
`→ [N×d_gat, N×d_vgae, 18-dim] → fusion_states/`. For `N≈50`, `d=64`, 10k
graphs: current 720 KB; per-node 128 MB/model; JK (K=3) 384 MB — all fine.
Variable-N via padding+mask or list-of-tensors (PyG `Batch` handles).
**RL state changes:** fixed-18-dim DQN/SAC becomes variable-length — use
attention pooling (fixed-dim before Q-net) or a graph Q-network (larger).

---

## Staged Recommendation

Don't skip to C without knowing whether 18-dim is the binding constraint:

1. **Fix reward, run clean baselines** — MLP-on-18-dim with correct
   reward is the lower bound (Bug 2 confounds current ceiling).
2. **A (richer stats)** — zero pipeline change, ~2h experiment. Closing
   the gap implicates the 4-statistic summary, not perm-invariant aggregation.
3. **B (JK-pool)** — extraction change, fixed-dim pipeline. Most likely
   to help on set_01/04 where heterogeneous locality bites.
4. **C (full node embeddings)** — only if A+B don't close the gap;
   specifically for suppress (t05) where per-node spatial structure is
   essential. Cross-attention spatial alignment justifies a paper section.

See `docs/drafts/fusion-improvement-plan.md` for how A/B/C plug in.
