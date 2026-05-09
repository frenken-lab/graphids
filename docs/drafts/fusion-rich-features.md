# Rich Per-Graph Features for Fusion: Beyond Aggregate Scalars

> **Scope.** Richer per-graph feature classes for Stage-3 fusion beyond the
> 18-dim aggregate-scalar vector. VGAE/GAT extraction is offline + frozen —
> cost paid once per graph. Targets four failure modes (2026-05-06):
> (a) suppress-attack collapse to 0.000, (b) RL policy collapse to constant
> arm, (c) VGAE→GAT subsumption (α→1.0), (d) threshold miscalibration.
> Companion: `fusion-research-notes.md`, `2026-05-06-fusion-analysis-prep.md`.

---

## Executive Summary

Current state captures **first-order moments** of node-wise quantities the
encoders already compute internally, discarding information in encoder
activations + topology.

| # | Feature class | Failure mode | Cost | Architecture |
|---|---|---|---|---|
| 1 | Raw GAT/VGAE node-embedding **sets** + Set Transformer / DeepSets pooling | (a), (c) | cache `[N, d]` | new pooling head |
| 2 | Per-node anomaly **quantiles** `{q05, q25, q50, q75, q95}` | (a), (d) | O(N log N), bounded | drops into 18-dim MLP |
| 3 | **Spectral / Laplacian** (top-k λ, VN entropy, gap, accumulated energy) | (a), (c) | O(N²) full or O(k·E) Lanczos | bounded, MLP-compat |
| 4 | **Graphlet / motif** count vectors | (a), interpret | O(N·d²) sampled | bounded, MLP-compat |
| 5 | **Cross-encoder interaction** (cos, attention, per-node disagreement) | (b), (c) | O(N·d) | bounded, MLP-compat |
| 6 | Per-edge VGAE recon **histogram** (deciles + edge AUC) | (a), (d) | O(E) | bounded |
| 7 | Attention-weight distribution (entropy, KL, top-k mass, head agreement) | (b), (c) | O(E·H) | bounded |
| 8 | Persistent homology features (Betti, persistence images) | (a) | O(E·α(N)) for β_0 | bounded |
| 9 | Latent-density / OOD (k-NN, full-Σ Mahalanobis, NF likelihood) | (a), (c) | O(d²) or O(NF) | bounded |
| 10 | Multi-scale / coarsened-graph features | multi-scale | O(N·d²) | learned pooling |
| 11 | SSL pretrained embeddings (InfoGraph / GraphMAE) | (c) | Stage-1.5 pretrain | bounded |

**Top three (do these first):**

1. **Cross-encoder interaction (#5).** Cheapest by an order of magnitude
   (≈10 lines, no architecture change), directly attacks α→1.0. Produces a
   feature score-fusion is architecturally incapable of constructing from
   `gat_attack` and `vgae_anom`. Grounded by Blum & Mitchell 1998
   (co-training, conditional independence) and Hazarika et al. 2020 (MISA).
2. **Per-node quantiles (#2).** `{mean,std,max,min}` → `{q05,q25,q50,q75,q95}`
   drops into existing MLP, captures distribution shape. Suppress visible in
   lower tail of edge-recon; `min` (single sample) throws this away.
   Quantiles robust to spike noise (Gan et al. 2018).
3. **Spectral (#3) with literature-attested basis.** Current `vgae/rq` is one
   scalar Rayleigh quotient. RQGNN (Dong et al. 2023, ICLR 2024) shows
   graph-level anomaly is best captured by **accumulated spectral energy**
   (per-band, not single ratio). Top-k λ + spectral gap + VN entropy = 6–8 dim.

These three: ≈30–40 extra dims, ≈one extraction-time epoch, preserve existing
MLP. Set-aggregation (#1) is highest-ceiling but commits new architectural
component — defer until #2/#3/#5 validated.

**Dataset anchors** (`extract.py:73`, `_base.py:101`): `window_size=100`,
`stride=100` → ≤100 events/graph collapsed to ≤vocab unique IDs. Realistic
**N ≈ 20–80, E ≤ few hundred**. Cache `[N, d]` at d=64 fp16 = ≤10 KB/graph;
~7.5K hcrl_sa graphs → ≤75 MB/dataset. **Budget allows raw per-node tensors.**

---

## §3.1 Raw Node-Embedding Sets (GAT and VGAE)

Keep `gat_emb ∈ R^{N × d_gat}` and `vgae_z ∈ R^{N × d_z}` as sets, consume
via permutation-invariant aggregator: DeepSets `f(X) = ρ(Σ_i φ(x_i))` (Zaheer
et al. 2017) or Set Transformer SAB + PMA with k learnable seeds
`S ∈ R^{k × d}` (Lee et al. 2019). Current 4-stat collapse is DeepSets with
fixed `ρ`; learned `ρ` recovers DeepSets universal approximation (Wagstaff
et al. 2019).

**Why:** (a) attention routes a single outlier embedding into the pool;
mean/std/max/min wash it out. (c) score-fusion blends scalars and discards
16 dims; set-pooling lets both encoders contribute full sets and cross-attention
compute disagreement at node level. **Cost:** cache `[N, d]` ≤10 KB/graph
at d=64 fp16, ≤75 MB/dataset. Set Transformer L=2 SAB + k=4 PMA at N=80,
d=64 → ~410K mults/graph; DeepSets O(N·d²) cheaper. **Architecture change
required** (variable N).

```
gat_pooled  = SetTransformer(gat_emb)    # [B, k·d]
vgae_pooled = SetTransformer(vgae_z)
state       = cat[scalars_18, gat_pooled, vgae_pooled]
```

**Issues.** Set Transformer data-hungry (Lee 2019 §4); at hcrl_sa scale
(~7.5K graphs) DeepSets safer first cut. Permutation-invariance audit
needed. Inherits 1-WL upper bound (Xu et al. 2019, GIN); matters less for
CAN since attacks alter node features (byte payloads), not just topology.

**Cite.** Zaheer et al. (2017), *Deep Sets*, NeurIPS, arXiv:1703.06114.
Lee et al. (2019), *Set Transformer*, ICML, arXiv:1810.00825. Wagstaff et al.
(2019), *On the Limitations of Representing Functions on Sets*, ICML. Sun
et al. (2020), *InfoGraph*, ICLR, arXiv:1908.01000.

---

## §3.2 Per-Node Anomaly Quantile Features

For each per-node series (`recon_per_node`, `mahal_per_node`, `kl_per_node`,
GAT confidence, GAT attention entropy, edge-recon-prob), replace 4-stat
`{mean,std,max,min}` with 5-stat `{q05,q25,q50,q75,q95}` via `torch.quantile`
(`q_p(x) = inf{t : P(X ≤ t) ≥ p}`). 5 dims × ~4 series = 20-dim block.

**Why:** (a) suppress removes CAN IDs → lower tail of edge-existence-prob
anomalously low; `min` single-sample (high variance), `q05` robust order
statistic. Distribution shape is load-bearing (Gan et al. 2018: quantiles
outperform 4-moment stats for anomaly detection, specifically for tail
shape). (d) quantiles monotonically transformed by score function →
threshold decisions inherit calibration (Calikus et al. 2023). **fp16-safe:**
bounded by input range; current code clamps moments to ±10 to avoid skew/kurt
→ 1e17 per `.claude/rules/critical-constraints.md`. **Cost:** O(N log N),
N ≤ 100. Cache 20 dims, replaces 16. **Net +4 dims. Drops into existing MLP.**
Modify `extract_features` in `graphids/core/models/{supervised/gat.py:296,
autoencoder/vgae.py:482}`. **Issues:** N < 20 → `q05`/`q95` collapse to
min/max; tied values → `method='linear'` or `'lower'`. Expect +0.05–0.1
AUROC on shape-sensitive subtypes (Gan 2018 §6).

**Cite.** Gan et al. (2018), *Moment-Based Quantile Sketches*, PVLDB 11(11),
1647–1660. Calikus, Nowaczyk, Pinheiro Sant'Anna (2023), *Explainable
contextual anomaly detection using quantile regression forests*, DMKD 37,
2517–2563. Karnin, Lang, Liberty (2016), *Optimal Quantile Approximation in
Streams* (KLL), FOCS.

---

## §3.3 Spectral / Laplacian Signatures Beyond Rayleigh Quotient

`L = D - A` or `L_norm = I - D^{-1/2} A D^{-1/2}`, eigendecomp `L = U Λ U^T`.
Per-graph features:

- **Top-k / bottom-k eigenvalues.**
- **Spectral gap** `λ_2 - λ_1` (Fiedler / algebraic connectivity).
- **Spectral entropy** `H(Λ) = -Σ p_i log p_i`, `p_i = λ_i / Σλ_j`.
- **VN graph entropy** `S(ρ) = -tr(ρ log ρ)`, `ρ = L / tr(L)` (Chen et al.
  2019, FINGER).
- **Accumulated spectral energy in band [a,b]:**
  `E(a,b) = Σ_{i: a ≤ λ_i ≤ b} ⟨x, u_i⟩²`. Load-bearing in **RQGNN** (Dong
  et al. 2023): anomalous graphs have systematically different energy
  distributions; current scalar `vgae/rq` collapses this multi-band quantity
  into one number.

**Why:** (a) topology attacks → λ_2 drops sharply on near-disconnection;
current `rq` averages over frequency, washing this out. (c) spectral computed
from topology + features, not from classifier-trained encoder → **structurally
orthogonal** to GAT. Per RQGNN, accumulated spectral energy is the
discriminator unsupervised methods exploit best on graph anomaly benchmarks.
**Cost:** Top-k Lanczos (`scipy.sparse.linalg.eigsh`) O(k · nnz(L)) ~5K
flops/graph. Full eig at N≈100: O(N³) ≈ 1M flops; extraction one-time, just
do full eig. Cache 8 top + 8 bottom + 3 scalars = 19 dims. **Drops into MLP.**
**Issues:** use eigvalues only (eigenvectors need SignNet/BasisNet — Huang et
al. 2023 — marginal gain). Variable N → fixed top-k/bottom-k + global scalars,
don't pad. VN entropy ≈ structural (degree) entropy (Liu et al. 2021); pick
one. Spectral fails on near-isomorphic families (Dell et al. 2018) — rarely
limiting for CAN.

**Cite.** **Dong, Zhang, Wang (2023), *Rayleigh Quotient GNN for Graph-level
AD*, ICLR 2024, arXiv:2310.02861. Most directly relevant.** Chen et al.
(2019), *FINGER*, ICML. Liu et al. (2021), *Bridging the Gap between VN
Graph Entropy and Structural Information*, TheWebConf. Akoglu, Tong, Koutra
(2015), *Graph based anomaly detection: a survey*, DMKD 29(3) §3. Huang et
al. (2023), *SPE*, NeurIPS, arXiv:2310.02579.

---

## §3.4 Graphlet / Motif Count Features

Count occurrences of size-k connected subgraphs. k=3: 4 graphlets; k=4: 11;
k=5: 34. Directed extension (Milo et al. 2002). CAN-relevant: triangles
(legitimate inter-ID), 2/3-stars (DoS/flood), reciprocal edges
(request-response).

**Why:** (a) suppressed ID → motifs containing it disappear; triangle/star/
reciprocal counts drop. Count-domain features visible *because* suppressed
nodes literally aren't in the graph. Interpretability: each graphlet
domain-meaningful in CAN. **Cost:** Triangle O(N · d_max²) ≤ 10K ops;
4-graphlets via Pržulj orbit counting O(E · d_max²) ≤ 100K ops. Cache 4+11=15
dims undirected; ~30 directed. **Drops into MLP.** **Issues:** heavy-tailed
counts → log(1+count); k=3 ≈ 1-WL (Lanzinger & Barceló 2023), use k=4/5 for
finer discrimination; discriminative signal is distribution vs. benign, not
raw count → store ratios or let MLP learn contrast.

**Cite.** Shervashidze et al. (2009), *Efficient graphlet kernels*, AISTATS.
Pržulj (2007), *Bioinformatics* 23, e177–e183. Milo et al. (2002), *Network
motifs*, Science 298, 824–827. Lanzinger & Barceló (2023), arXiv:2309.17053.

---

## §3.5 Cross-Encoder Interaction Features (HIGH PRIORITY)

Explicit comparisons between GAT and VGAE for the same graph:

- **Graph-level cosine** `cos(g_GAT, g_VGAE)` and **L2** `||g_GAT - g_VGAE||_2`.
- **Per-node cosine quantiles** `s_i = cos(emb_GAT[i], proj(z_VGAE[i]))` then
  `{q05, q50, q95}` of `{s_i}`. (`proj`: linear, fusion-trained, ≤2K params.)
- **Cross-attention** `gat_emb` Q × `vgae_z` KV (or reverse), pooled. d-dim.
- **Per-node prediction disagreement.** GAT per-node anomaly marginal vs.
  VGAE per-node recon z-score; count nodes disagreeing by ≥ threshold.

**Why:** (c) α→1.0 **central diagnosis.** Score-fusion `α·gat_attack +
(1−α)·vgae_anom` goes to α=1 because `vgae_anom` is monotone in `gat_attack`
on labeled data — adds no information **as scalar**. But cosine-between-encoders
is a feature neither produces individually. High = both see graph similarly;
low = disagree (novel). Orthogonal signal (Hazarika et al. 2020 MISA:
modality-specific reps carry complementary information). (b) RL collapse:
LinUCB couldn't extract per-sample variation from 18-dim state. Cross-encoder
features have natural per-graph variance: cos ∈ [-1, 1] depending on graph,
not encoder bias — gives bandit per-sample signal even when `gat_attack`/
`vgae_anom` at typical benign values. **Theory:** two-view co-training (Blum
& Mitchell 1998): conditionally-independent error modes outperform either
view; cosine disagreement is the proxy. **Cost:** graph cos+L2 O(d), per-node
cos O(N·d), cross-attention O(N²·d) ≤ 640K flops/graph at N=100, d=64. Cache
5–10 scalars (+optional cross-attn d-dim). **Scalars drop into MLP.**
**Issues:** dim mismatch (d_gat=64 vs. d_z=32) → learned `proj`,
fusion-trained. Cos on small-norm vectors high-variance → normalize at
extraction. Score-fusion (WeightedAvg) can't consume these; needs MLP gate
(AttentionFusion per `fusion-research-notes.md` §2.2).

**Cite.** **Hazarika, Zimmermann, Poria (2020), *MISA*, ACM MM, arXiv:2005.03545.
Most directly applicable.** Blum & Mitchell (1998), *Co-training*, COLT.
Tian, Krishnan, Isola (2020), *Contrastive Multiview Coding*, ECCV.
Xu, Tao, Tao (2013), *A survey of multi-view learning*, AI Review 42(2).

---

## §3.6 Per-Edge VGAE Reconstruction Histogram Features

VGAE decoder outputs `σ(z_u^T z_v)` per edge candidate. Currently only mean
recon-error is cached. Replace with histogram:

- `p_pos = σ(z_u^T z_v)` for (u,v) ∈ E; `q_neg` for (u,v) ∉ E (sampled).
- `{q05,q25,q50,q75,q95}` of `p_pos` (low = expected-missing edges that
  survived) and `q_neg` (high = expected-present edges that didn't appear).
- **Edge AUC** `AUC(p_pos, q_neg)`. 1 scalar.
- **Banded recon** by (src_id_band, dst_id_band) 3×3 = 9 dims;
  attack-localizing for ID-range floods.

**Why:** (a) suppress removes edges; standard recon loss only sums observed
edges, but `q_neg` distribution **directly captures** "edges decoder
predicted should exist but don't" — the attack signal. (d) Edge AUC naturally
[0,1] calibrated. **Cost:** O(E + |neg|) ≤ 400 dot products. Cache 10–15
dims (+optional banded). **Drops into MLP.** **Issues:** uniform negatives
too easy → hard-negative mining (small `||z_u - z_v||` but no edge) or
degree-stratified. VGAE saturation: if benign edge AUC > 0.99, quantiles
collapse — audit at extraction. Stride=100 → ≤200 events/window → sparse
edges, per-edge features may not stabilize.

**Cite.** Kipf & Welling (2016), *VGAE*, NIPS Workshop, arXiv:1611.07308.
Fan et al. (2020), *AnomalyDAE*, ICASSP, arXiv:2002.03665. Wang et al.
(2020), *OCGNN*, arXiv:2002.09594.

---

## §3.7 Attention-Weight Distribution Features (GAT)

Per-edge `α_{ij}^h`. Per-graph: entropy `H = -Σ α log α` (per-node,
graph-mean); KL from uniform; top-k attention mass; inter-head agreement
(variance across H heads); attention rollout (Abnar & Zuidema 2020:
recursive matmul across layers, then quantiles).

**Why:** (b) "looking everywhere" (high entropy) vs. "looking at one edge"
(low) is observationally distinct, *not* in current features. (c) metadata
about how GAT processed graph; doesn't duplicate `gat_conf`/`gat_probs`.
**Cost:** O(E·H). Cache 4–6 dims. Drops into MLP. **Issues:** vanilla GAT
"static" attention (Brody et al. 2021 GATv2) — ranking independent of query;
audit within-query vs. across-query variance. Raw attention ≠ explanation
(Wiegreffe & Pinter 2019); use rollout.

**Cite.** Veličković et al. (2018), *GAT*, ICLR. Brody, Alon, Yahav (2021),
*GATv2*, arXiv:2105.14491. Abnar & Zuidema (2020), *Attention Flow*, ACL,
arXiv:2005.00928. Miao, Liu, Li (2022), *GSAT*, ICML, arXiv:2201.12987.

---

## §3.8 Topological / Persistent Homology Features

PH tracks birth/death of topological features (CC, cycles, voids) under
filtration. Vectorize via persistence images (Adams et al. 2017,
Gaussian-convolved + discretized, ~20×20=400 dims), persistence landscapes
(Bubenik 2015), Betti numbers β_0/β_1 at thresholds, total persistence
Σ(death−birth). Standard graph filtration is vertex-based (Hofer et al.
2020, GFL): score per vertex (GAT conf or VGAE recon), filter by sublevel
sets, track CC/cycle birth+death.

**Why:** (a) PH directly captures *connectedness* changes; suppress
disconnects → distinctive shift in β_0 pattern. Most theoretically
principled "topology changed" detector. Orthogonal: captures how filter is
*distributed across graph*, not just mean. **Cost:** β_0 union-find
O(E·α(N)); β_1 O(E²); persistence image ~1K ops. Cache 64–256 dims. Drops
into MLP. **Issues:** most expensive to **implement** (Gudhi/Ripser/giotto-tda;
library overhead ≫ compute). Filter function dominates result. Multi-parameter
PH (Carrière et al. 2020) addresses single-filter limits but doubles cost.

**Cite.** Adams et al. (2017), *Persistence Images*, JMLR 18, 1–35. Bubenik
(2015), JMLR 16, 77–102. Hofer, Kwitt, Niethammer (2020), *Graph Filtration
Learning*, ICML. Hofer et al. (2017), *Deep Learning with Topological
Signatures*, NeurIPS, arXiv:1707.04041. Rieck (2023), arXiv:2302.09826.
Moor, Horn, Rieck, Borgwardt (2020), *Topological Autoencoders*, ICML,
arXiv:1906.00722.

---

## §3.9 Latent-Density / OOD Scores

OOD scores in VGAE `z`: **k-NN distance** to frozen benign ref (≤5K×64 =
1.3 MB); **full-Σ Mahalanobis** (Lee et al. 2018) vs. current diagonal;
**NF likelihood** on z (RealNVP/TARFlow; ≤50K params, <1ms); **energy**
`E(x) = -logsumexp(logits)`.

**Why:** (c) benign-only density captures distributional anomaly label-trained
classifiers don't: rare-but-well-formed benign registers low under density,
high under classifier — opposite of attacks. (a) sparse topology → z's
near origin; full-Σ Mahalanobis catches, diagonal doesn't. **Cost:** k-NN
O(|ref|·d) ~300K flops; Mahalanobis O(d²); NF O(params). Cache 1–4 scalars
+ ref tensor. Drops into MLP. **Issues:** multi-modal benign → mixture
(PALM, Lu et al. 2024); ref set frozen (drift = stale, OK for paper).

**Cite.** Lee et al. (2018), *Mahalanobis-OOD*, NeurIPS, arXiv:1807.03888.
Denouden et al. (2018), arXiv:1812.02765. Ruff et al. (2018), *Deep SVDD*,
ICML. Lu et al. (2024), *PALM*, arXiv:2402.02653. Aathreya & Canavan
(2024), *FlowCon*, arXiv:2407.03489.

---

## §3.10 Multi-Scale / Coarsened-Graph Features

Hierarchy via DiffPool (Ying 2018), Top-k (Gao&Ji 2019), SAGPool (Lee 2019),
or spectrum-preserving GW (Chen 2023). **Verdict: defer for this dataset.**
N≈80 → 1–2 levels max; compelling on N≫1000.
**Cite.** Ying et al. (2018), *DiffPool*, NeurIPS. Lee, Lee, Kang (2019),
*SAGPool*, ICML. Loukas (2019), JMLR. Chen et al. (2023), arXiv:2306.08854.

---

## §3.11 Self-Supervised Pretrained Embeddings (InfoGraph / GraphMAE)

Pretrain third encoder via label-agnostic SSL (InfoGraph MI; GraphCL
contrastive; GraphMAE masked-feature recon). Use graph-level `g_SSL` as
third feature view.

**Why:** (c) GAT supervised on labels; VGAE recon on benign; SSL on third
objective, by design orthogonal. Two-view co-training (Blum & Mitchell
1998) extends to multi-view. **Cleanest theoretical answer to "where is the
orthogonal signal?"** **Cost:** Stage-1.5 pretrain ~30 epochs × 7.5K graphs
≈ current VGAE pretrain. Extraction +1 ms/graph. Cache d_SSL ~128. Drops
into MLP. **Issues: GraphCL augmentations conflict with suppress** —
node/edge-drop positive pairs teach encoder that suppress-like graphs are
benign (You et al. 2020). Use **GraphMAE** (masked features, no edge aug)
or **InfoGraph** (no aug). Operational: another KD-stage ckpt.

**Cite.** **Sun et al. (2020), *InfoGraph*, ICLR, arXiv:1908.01000.** You
et al. (2020), *GraphCL*, NeurIPS, arXiv:2010.13902. Hou et al. (2022),
*GraphMAE*, KDD, arXiv:2205.10803. Veličković et al. (2019), *DGI*, ICLR,
arXiv:1809.10341.

---

## Cross-Cutting: Failure Modes vs. Features

| Failure | First-line | Reasoning |
|---|---|---|
| (a) suppress → 0.000 | #3 spectral (λ_2), #4 motifs, #6 per-edge hist, #8 PH β_0 | Topology attack → topology-domain features structurally sensitive |
| (b) RL collapse | #5 cross-encoder, #7 attention dist., #2 quantiles | Per-sample variance is the missing ingredient |
| (c) α→1.0 subsumption | #11 SSL, #5 cross-encoder, #3 spectral | "What does unsupervised proxy compute that label-trained doesn't?" |
| (d) miscalibration | #2 quantiles, #6 per-edge AUC, #9 density | Order-statistic + density naturally calibrated |

---

## §4 Recommended Ablation Plan

**§4.1 Phase 0 — no architecture change (this week).** Combine #2 quantiles +
#5 cross-encoder scalars + #3 spectral (no RQGNN extension yet). Net new dims
≈30. Modify `extract.py::_extract_states` on existing GAT/VGAE ckpts — no
Stage-1/2 retraining. Re-train all four fusion methods. **Hypothesis tests:**
does WeightedAvg α move off 1.0? Does MLP gain on suppress? Does DQN per-sample
variance appear?

**§4.2 Phase 1 — next sprint, conditional on Phase 0.** If partial: add #4
motifs + #6 per-edge histogram (next-cheapest topology-domain). If fails on
suppress: add #8 PH via giotto-tda.

**§4.3 Phase 2 — committed architecture change.** Add #1 Set Transformer over
raw embedding sets — highest-ceiling, highest commitment. Only after
#2/#5/#3 confirm bottleneck.

**Phase 3 — third encoder.** GraphMAE or InfoGraph as Stage-1.5. End-state if subsumption persists.

**Always defer (this dataset/scale):** #10 multi-scale (N too small).

---

## Appendix: Numeric Anchors

- N (per-graph nodes) ≈20–80; E ≤ few hundred. Train: hcrl_sa ~7.5K, set_01–04 larger.
- `d_gat` ~64 (`gat.py`); `d_z` ~32 (`vgae.py`); existing fusion state 18 dims.
- Phase-0 add ≈30 (quantiles + cross-encoder + spectral) → final ≈48 dims.
- Phase-2 Set Transformer cache 18 + N·(d_gat + d_z) ≈ 7700 floats/graph; 7.5K × 4 B = 232 MB/dataset.

**Load-bearing files:** `graphids/core/data/extract.py` (`_extract_states`
calls each model's `extract_features`; `CACHE_VERSION = 5` at line 31, bump
on schema change). `graphids/core/models/supervised/gat.py:296` — GAT
`extract_features`. `graphids/core/models/autoencoder/vgae.py:482` — VGAE
`extract_features`, per-edge features. `graphids/core/models/fusion/base.py`
— flatten layout; update on state-dim growth.
