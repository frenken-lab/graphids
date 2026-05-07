# Fusion improvement plan — implementation staging

> Synthesizes `docs/research-notes/fusion-research-notes.md` (theory),
> `docs/empirical-notes/2026-05-06-fusion-analysis-prep.md` (what shipped + what failed),
> and `docs/research-notes/more-fusion-notes.md` (state-bottleneck options A/B/C).
> Action-oriented; staged so each phase has a defined off-ramp.

## Executive summary

Two failure axes interact in the 2026-05-06 results: a **broken reward** (PBRS-violating
`balance` + agreement bonuses, exploited by the 86%-benign equilibrium) and a **lossy state**
(18-dim aggregate scalars; per-node structure is computed inside `gat.extract_features` and
`vgae.extract_features` then thrown away). Don't mix the two fixes in one row — they have
different blast radii and different acceptance criteria. Diagnose first (Phase 0), fix the
reward without touching the cache (Phase 1–2), then decide whether the residual gap justifies
extraction-pipeline changes (Phase 3–4). Phase 5 is the off-ramp if RL stays brittle.

## Choice space

| Phase | What changes | Cache regen? | Model changes? | Justifies if… |
|---|---|---|---|---|
| 0. Diagnostics | reward logging, MLP rerun | no | no | always — baseline gate |
| 1. Reward strip | `reward.py::compute` | no | no | bandit/DQN MCC≈0 confirmed reward, not state |
| 2. Algorithm swap | new fusion variants (IQL/TD3+BC, threshold-as-action, BC warm-start) | no | new fusion classes | Phase 1 reward fix improves but doesn't close MCC gap |
| 3. Bundled re-extract | quantiles + cross-encoder cosines + spectral, in one pass | **yes** (CACHE_VERSION 5→6) | extract only | Phase 1+2 plateau; suspect aggregate scalars are bottleneck |
| 4. Per-node + 3rd encoder | JK-pool / full `H` in cache + GraphMAE/InfoGraph | **yes** | extract + fusion + new pretrain | Phase 3 closes some gap on set_01/04 but not t05 (suppress) |
| 5. Drop RL, MoE-BCE | new fusion variant w/ supervised gating | no (Phase 1 cache) | new fusion class | RL collapse persists across reward fixes |

---

## Phase 0 — Diagnostics (do this first, blocks nothing else)

### 0.1 Per-component reward logging  ✅ done — commit `0949e8e` (2026-05-07)
`reward.py::compute` returns a sum; no caller can tell which term dominates. Five-line diff:
return a `dict[str, Tensor]` of named components, accumulate per-epoch in
`MLflowTrainingCallback`, log as `r_classification`, `r_confidence`, `r_agreement`,
`r_balance`. This would have caught Bug 4 (inverted vgae_conf) at epoch 1 instead of post-test.
**Acceptance:** for the 2026-05-06 DQN runs, see `r_agreement` dominate `r_classification` —
confirms the all-benign-equilibrium diagnosis in `fusion-analysis-prep.md` §Findings.

> **Implemented as:** `compute()` returns `(total, components)`; `train_episode` adds
> components to its return dict (auto-aggregated by Lightning); `validation_step` logs them
> with `val_` prefix. `sum(components.values()) == total` by construction.

### 0.2 MLP rerun (clean supervised baseline)  🟡 in flight — pitzer `47361091/2/3` (2026-05-07)
The hcrl_sa MLP row from plan `019e0028` is Bug-2-era: 1 val epoch only. Resubmit on the
post-Bug-2 chassis. This is the **lower bound** for any fusion-vs-supervised comparison; nothing
downstream is publishable without it. Single submission, ~1h, no code change.
**Acceptance:** clean MLP curve with full-epoch val_acc trajectory.

> **Submitted as** plan_id `019e0338-f7f7-78de-8fe0-0f1527763a13` on commit `0949e8edb3e2`.
> Chain: `47361091` (extract, gpu) → `47361092` (mlp fit, cpu) → `47361093` (mlp-test, cpu).
> Re-extract is forced by CACHE_VERSION 5→6 from §0.3 below.

### 0.3 Stratified subtype metrics  ✅ done — commit `0949e8e` (2026-05-07)
`fusion-research-notes.md` §2.3: aggregate AUROC on set_01/04 conflates injection / suppress /
fuzzy / timing. Add `auroc_per_attack/{name}` to test-phase metrics if not already there. Costs
nothing at train time; required to evaluate any later VGAE-leveraging change.

> **Implemented as:** `extract.py` propagates `batch.attack_type` into the cache + stashes the
> schema's name map. `FusionDataModule.attack_type_names` exposes it; `prepare_from_datamodule`
> picks it up via fallback. Existing `_log_per_attack_auroc` (base.py:219) fires automatically
> on the fusion test path. CACHE_VERSION 5→6 forces re-extract on first use.

---

## Phase 1 — Reward strip (no extraction change, no algorithm change)

### 1.1 PBRS-compliant reward  🟡 code ready (uncommitted) — awaits Phase 0 diagnostics
Replace the current sum (`fusion-analysis-prep.md` §RL reward structure) with the minimum
viable from `fusion-analysis-prep.md` §Reward redesign and `fusion-research-notes.md` §4.1:

- Drop: `balance`, `agreement`, `disagreement_penalty`, `combined_conf_weight`.
- Keep: asymmetric FN/FP costs (FN=−6, FP=−1.5), TP=+3, TN=+1.5, attack-gated confidence bonus.
- Add (deferred to 1.2): pairwise ranking bonus.

> **Implemented as** `MinimalFusionRewardCalculator` (reward.py) + `REWARD_MINIMAL` primitive
> (plan/primitives.py). Factory `FusionRewardCalculator.from_kwargs(**reward_kwargs)`
> dispatches on optional `mode` field — legacy plans (no `mode`) keep getting the old
> calculator. Components dict keys uniform across both calculators (zero-fills for inactive
> shaping terms) so MLflow comparisons are clean. Smoke-tested. **Hold submission** until
> Phase 0 chain returns and `r_agreement` dominance confirmed under legacy reward.

The `balance` term is the immediate cause of α≈0.5–0.7 keeping the blended score in the
miscalibrated regime. Drop it, the policy is free to find the data-optimal direction. The
agreement bonus is the immediate cause of the all-benign equilibrium on hcrl_sa
(86% benign × +0.3 ≈ exceeds attack-detection reward; computed in
`fusion-analysis-prep.md` §Findings).

**Acceptance:** bandit/DQN MCC > 0.5 on hcrl_sa with no other change. If MCC stays near 0,
state is the bottleneck, not reward — skip to Phase 3.

### 1.2 Pairwise ranking reward (additive)
`fusion-research-notes.md` §4.3 — Wilcoxon-Mann-Whitney surrogate, 256 attack×benign pairs per
batch, O(N) cost. Majority-class neutral. Add as a component alongside §1.1, weighted s.t. it
contributes at the same magnitude as the classification term at random-policy baseline.
**Acceptance:** AUROC on set_01/04 improves; confirms the ranking-vs-threshold separation
is the right frame.

### 1.3 Post-hoc Platt scaling on existing DQN Q-values
`fusion-research-notes.md` §3.1 — `Q(s,1)−Q(s,0)` already ranks correctly (AUROC≈1.0 on
hcrl_sa). Logistic regression on val set against binary labels. Standalone fix that converts
AUROC≈1, MCC≈0 into AUROC≈1, MCC≈something. Useful as a sanity check that the ranking
really is correct — if Platt scaling doesn't fix MCC, ranking is also broken and §1.1 isn't
sufficient.

**Reclassified 2026-05-07: NOT post-hoc / NOT zero-retrain.** Verified that
`base.py:_finalize_test_predictions` builds `model._test_predictions` in-memory only — no
hook persists it to disk. The 2026-05-06 DQN runs' Q-values are gone. Sequence is now:
(a) wire test-prediction persistence (add `torch.save(model._test_predictions, run_dir /
"test_predictions.pt")` in the evaluate path); (b) resubmit DQN once with the new
persistence; (c) THEN Platt-scale post-hoc. Step (a) is small; step (b) costs one fit job
per dataset/seed. Treat 1.3 as Phase 1 cost, not free.

> **(a) done 2026-05-07 (uncommitted).** `orchestrate.py:evaluate` now writes
> `{run_dir}/test_predictions.pt` after `trainer.test`. The in-flight `mlp-test` job
> (47361093) will produce the first set of persisted predictions; future Platt fits can
> read directly from disk. (b) and (c) still pending.

---

## Phase 2 — Algorithm swaps (no extraction change)

These are new fusion variants registered alongside `weighted_avg`/`bandit`/`dqn`/`mlp`,
trained on the same `fusion_states.pt` cache.

### 2.1 BC warm-start for any RL variant
GAT achieves AUROC≈1 on set_02/03 — it is the de-facto expert. Pre-train the policy
(Q-net or actor) by supervised regression onto `gat/probs[:,1]`, then fine-tune with the
Phase-1 reward. Single largest fix for the constant-arm-20 collapse: policy starts on the
correct ranking surface, reward only refines the threshold. Cost: a few hundred BC steps
before RL training begins.
**Acceptance:** `avg_alpha` per-batch std > 0 throughout training (i.e., per-sample
adaptivity preserved instead of collapsing to a constant arm).

### 2.2 Offline-RL methods (IQL or TD3+BC)
`fusion-research-notes.md` recommends SAC because it's off-policy. Off-policy ≠ offline.
A frozen cache with no rollouts is strictly offline; SAC suffers from extrapolation error
on out-of-distribution Q-values, which is the textbook signature of "ranks correctly but
threshold is wrong" — exactly what hcrl_sa DQN showed. Drop in either:
- **IQL** (Kostrikov et al. 2022, ICLR) — expectile regression, never queries Q on OOD
  actions during training. Generally most stable on small datasets (~7.5k train graphs).
- **TD3+BC** (Fujimoto & Gu 2021, NeurIPS) — TD3 + behavior-cloning regularizer. The BC
  term is "stay close to GAT's prediction," compatible with §2.1.
**Acceptance:** match or exceed the Phase-1 DQN MCC at the same AUROC; smaller AUROC-vs-MCC
gap than DQN.

### 2.3 Threshold-as-action policy (decouple ranking from calibration)
`fusion-research-notes.md` §3.4 Option C. Action = decision threshold τ ∈ [0,1] applied to
`gat/probs[:,1]`. State can include batch attack-rate estimate, recent FP/FN counts. The
ranking problem is solved by GAT; this isolates the calibration problem on a 1-D continuous
action space. Trivially learnable. Use as a **second RL row** alongside α-as-action — gives
a working baseline if α-policy keeps misbehaving.
**Acceptance:** MCC > 0.5 on hcrl_sa from threshold policy alone, with AUROC unchanged
(by construction equal to GAT's).

### 2.4 Distributional RL (C51 / QR-DQN) — optional
`fusion-research-notes.md` doesn't cover this. Mean-Q collapses the bimodal value
distribution under class imbalance (most steps benign-correct, rare attack-correct steps
high-reward). Drop-in replacement for the existing DQN code path. Only justifies the
implementation cost if §2.1+§2.2 don't close the calibration gap.

---

## Phase 3 — Bundled re-extraction (cache regen required, no model retrain)

`gat.extract_features` (gat.py:296) and `vgae.extract_features` (vgae.py:482) currently
aggregate per-node embeddings to `[mean, std, max, min]` *before* the cache write. Per-node
tensors are computed and dropped. The feature-research note
(`docs/research-notes/fusion-rich-features.md`) recommends bundling **three** feature classes
into one re-extraction pass — they share the cache-regen cost, and isolating them costs
3× the SLURM time without giving 3× the information (the failure modes overlap). Bump
`CACHE_VERSION = 5 → 6` once. State dim grows from 18 to ~48; trivially handled by existing
MLP/Q-net.

### 3.1 Cross-encoder interaction features (top recommendation, ~5 scalars)
`fusion-rich-features.md` §5. Compute `cos(g_GAT, g_VGAE)`, `||g_GAT − g_VGAE||₂`, and
per-node-cosine quantiles `{q05, q50, q95}` of `cos(emb_GAT[i], proj(z_VGAE[i]))`. The
projection `proj` is a fusion-train-time layer (no re-extract dependency); the cosine itself
is computed at extraction. **Directly attacks α→1.0**: this is a feature score-fusion is
architecturally incapable of constructing from `gat_attack` and `vgae_anom` alone. Theoretical
basis: Hazarika et al. 2020 (MISA), Blum & Mitchell 1998 (co-training conditional independence
proxy). Cheapest by an order of magnitude — ~10 lines, no architecture change.

### 3.2 Per-node quantile features (~20 dims, replaces 16 dims of stats)
`fusion-rich-features.md` §2. Replace `{mean, std, max, min}` blocks on `gat_emb`, `vgae_z`,
and per-node recon error with `{q05, q25, q50, q75, q95}`. Net +4 dims. Quantiles are robust
order statistics — survive the spike-noise that contaminates `mean`/`std`, capture the
*lower tail* that suppress attacks live in (`min` is one sample; `q05` is the empirical
percentile). Bonus: bounded by input range, never overflow fp16 (current code clamps moments
to ±10 for this reason — see `.claude/rules/critical-constraints.md`).

**Caveat:** for graphs with N < 20 nodes, q05/q95 collapse to min/max. Verify per-dataset
node-count distribution before committing — see Open Questions below.

### 3.3 Spectral signatures beyond Rayleigh quotient (~19 dims)
`fusion-rich-features.md` §3. Current `vgae/rq` is one scalar from energy density at all
frequencies. Replace/extend with: top-8 Laplacian eigenvalues, bottom-8 (algebraic
connectivity / Fiedler value), spectral entropy, von Neumann entropy, spectral gap. Compute
via Lanczos (`scipy.sparse.linalg.eigsh`, O(k·nnz(L))) or full eig at N≈80 (O(N³) ≈ 1M flops,
trivial at extraction). **Directly attacks suppress (t05)**: λ₂ drops sharply when the graph
near-disconnects; the current scalar `rq` averages this out. Theoretical basis: RQGNN
(Dong et al. 2023, ICLR 2024) — accumulated spectral energy beats single Rayleigh quotient
on graph-anomaly benchmarks. Use eigvalues only, not eigenvectors (sign/permutation
ambiguity, would require SignNet stabilization).

### Phase 3 acceptance criteria

- WeightedAvg α moves off 1.000 — the cross-encoder feature gives the score-fusion blend
  something the two scalars don't produce. (Strong signal that §3.1 is doing its job.)
- DQN/bandit `avg_alpha` per-batch std > 0 — per-sample variance from §3.1 + §3.3 breaks
  the constant-arm-20 collapse.
- MLP MCC improves on set_01/04 — distribution-shape signal from §3.2 captures GAT's
  systematic-failure regime.
- t05 (suppress) AUROC > 0.5 — §3.3 spectral features carry the topology signal.

If §3.3 doesn't move t05, fall back to **Phase 3.4** below.

### 3.4 Suppress-fallback features (only if Phase 3 doesn't fix t05)

Two cheaper, two more expensive. Defer all four unless §3.1–§3.3 together fail t05.

- **Motif counts** (`fusion-rich-features.md` §4) — directed-motif count vector (k=3,4):
  triangles, 2-stars, reciprocal edges. Suppress = motifs containing the suppressed ID
  disappear, observable in count domain. ≤50 dims, O(E·d_max²) trivial.
- **Per-edge VGAE recon histograms** (§6) — quantiles of `σ(z_u^T z_v)` over positive and
  negatively-sampled edges, plus edge-AUC. ~10–15 dims. Catches "edges the decoder predicted
  should exist but don't" — the suppress signature. Caveat: if benign edge-AUC > 0.99, the
  histogram is saturated; audit before committing.
- **Persistent homology** (§8, expensive) — Betti numbers + persistence images via
  giotto-tda or Ripser. Most theoretically grounded suppress detector (Rieck 2023 — captures
  topology features outside WL hierarchy) but heaviest dep + ~400-dim cache.

---

## Phase 4 — Per-node embedding fusion (extraction + fusion model changes)

Triggers only if Phase 3 doesn't close the set_01/04 gap *or* if t05 (suppress) is the
publication-blocking dataset. Two sub-options from `more-fusion-notes.md`.

### 4.1 JK-pool from GAT (Option B)
`more-fusion-notes.md` §Option B. Modify GAT inference to return all `K` layer embeddings;
JK-aggregate (max-pool variant — most memory-efficient, strong empirical performance).
Storage: `K × N × d` per graph during extraction. With K=3, N≈50, d=64, 10k graphs ≈
**384 MB** per dataset per model — manageable on `LAKE_ROOT`.

**Files touched:**
- `graphids/core/models/supervised/gat.py` — add `return_layer_embeddings` flag to forward;
  modify `extract_features` to compute JK-pool and emit a `[d]` graph-level vector.
- Cache version bump 6 → 7.
- Fusion variants stay 18-dim-input compatible (the new vector is just appended).

**Acceptance:** improvement on set_01/04 specifically — heterogeneous locality (1-hop
injection vs K-hop fuzzy/timing) is the JK-Net theoretical motivation.

### 4.2 Full per-node embedding sets + cross-modal attention (Option C)
`more-fusion-notes.md` §Option C. Cache stores `H_gat ∈ R^{N×d_gat}` and `H_vgae ∈
R^{N×d_vgae}` per graph; fusion model is a `GraphAttentionPool` + cross-modal attention head.

**This is a different fusion class, not a flag on the existing ones.** Variable-N inputs
require either padding-with-mask or PyG `Batch` handling at fusion-train time. The DQN/SAC
state space changes from 18-D to graph-shaped, which means the Q-network/actor must itself
become permutation-invariant — effectively a small graph network.

**Why it's worth doing despite the cost:** the cross-modal attention `(GAT-emb-as-query, VGAE-z-as-key)`
quantifies *spatial agreement* between the two encoders — does GAT find the same nodes
anomalous that VGAE finds hard to reconstruct? This is the principled replacement for the
broken agreement bonus in the current reward (`fusion-analysis-prep.md` §RL reward structure):
spatial agreement is a much stronger signal than scalar concordance.

**Acceptance:** improvement on t05 (suppress). t05 is the only dataset where Phase 3
quantiles/outlier-mass cannot help by construction (sparse-topology attacks have lower
node-level variance than benign).

**Storage:** ~128 MB per dataset per model with N≈50, d=64 — feasible but worth measuring
before committing.

### 4.3 Third encoder via self-supervised pretraining (GraphMAE / InfoGraph)
`fusion-rich-features.md` §11. The cleanest theoretical answer to "where is the orthogonal
signal?" — train a third encoder on a label-agnostic objective genuinely independent of
both GAT (cross-entropy) and VGAE (reconstruction). Multi-view co-training (Blum & Mitchell
1998) extends past two views; a third view trained on a third objective is the principled
extension.

**Avoid GraphCL.** Its augmentation set (node dropping, edge perturbation, subgraph
sampling) directly conflicts with the suppress-attack signal: training the encoder to map
edge-perturbed graphs to similar embeddings teaches it that suppress-like graphs are
benign (You et al. 2020 flag this explicitly). Use **GraphMAE** (masked feature
reconstruction, no edge augmentation; Hou et al. 2022, KDD) or **InfoGraph** (mutual-information
maximization between subgraph patches and graph-level summary; Sun et al. 2020, ICLR) —
neither augments edges.

**Cost:** new Stage-1.5 pretrain job (~30 epochs × 7.5K graphs ≈ similar to VGAE pretrain),
plus a third extraction pass adding ~128 dims to the cache. Operational cost: another KD
ckpt to track in the catalog and lineage in `LoggedModel`.

**Acceptance:** if Phase 3 cross-encoder cosines move WeightedAvg α off 1.0 but MCC gap
persists, a third encoder closes the residual subsumption. If Phase 3 already closes it,
defer 4.3 indefinitely.

---

## Phase 5 — Off-ramp: drop RL, run MoE+BCE

If Phase 1+2 don't close the MCC gap and Phase 3+4 don't either, the RL framing itself is
the problem. The fusion setup has no temporal credit assignment — state is observed once
per graph, action affects only that graph's prediction, reward is delivered immediately.
TD(0) over this MDP is BCE with extra steps and a worse optimization surface.

### 5.1 Mixture-of-experts with learned router, BCE loss
`fusion-research-notes.md` §5.2 — three experts (injection/fuzzy GAT-dominant; suppress
VGAE-topology-dominant on rq+z_stats; timing VGAE-temporal). Router is a softmax over
inputs. Train end-to-end with class-weighted BCE. **Per-sample gating without RL** —
identical capability to "continuous-α SAC," no reward-shaping bug surface, no policy
collapse, calibrated-by-construction (BCE is proper-scoring).

This is the architecturally honest answer to "why does RL keep collapsing." The gating-network
literature (Jacobs et al. 1991, Shazeer et al. 2017) is the right frame; RL is the wrong
frame imposed onto a problem that doesn't need it.

**Paper narrative if 5.1 wins:** "We tried RL fusion (bandit/DQN/SAC) and discovered the
fundamental issue — fusion is a contextual gating problem, not a sequential decision problem.
Supervised gating (MoE) matches RL ranking and beats it on calibration, while being
PBRS-immune by construction." Stronger than "we did RL fusion."

---

## Decision tree

```
Phase 0 done → r_agreement dominates? ─yes→ Phase 1 sufficient (reward was the bug)
                                       └no→ state is the bottleneck, jump to Phase 3

Phase 1 done → bandit/DQN MCC > 0.5? ─yes→ Phase 2 optional polish
                                      └no→ Phase 2.1 BC warm-start + Phase 2.2 IQL/TD3+BC

Phase 2 done → set_01/04 gap closed? ─yes→ Phase 3 only if t05 is required (3.3 spectral may help)
                                      └no→ Phase 3 bundled re-extract (3.1+3.2+3.3)

Phase 3 done → α moved off 1.0?    ─yes→ subsumption broken; if MCC gap persists → Phase 4.3 (3rd encoder)
                                    └no→ Phase 4.2 per-node + cross-modal attention
              t05 still 0.000?     ─yes→ Phase 3.4 fallback (motifs / per-edge / persistence)
                                    └no→ paper-ready

Any phase: RL still collapsing? ───→ Phase 5 MoE+BCE; reframe paper around supervised gating
```

## Open questions / verification tasks before each phase

- **~~(Blocks Phase 3.2 commitment) Per-dataset node-count distribution.~~ Resolved
  2026-05-07** from `cache/v10.0.0/{dataset}/voc_all/cache_metadata.json` — train-split
  node-count min / mean / max:

  | Dataset  | n_graphs | N_nodes | Action for §3.2                |
  | -------- | -------- | ------- | ------------------------------ |
  | hcrl_sa  | 7,496    | 21 / 27 / 64 | use full {q05, q25, q50, q75, q95} |
  | set_01   | 120,904  | **12** / 25 / 40 | drop q05; use {q25, q50, q75, q95} |
  | set_02   | 162,840  | **10** / 33 / 48 | drop q05; use {q25, q50, q75, q95} |
  | set_03   | 132,912  | 21 / 38 / 57 | use full {q05, q25, q50, q75, q95} |
  | set_04   | 97,955   | 22 / 32 / 46 | use full {q05, q25, q50, q75, q95} |

  The catalog only stores min/mean/max, not the actual q05 — but min ≥ 12 across all
  datasets means the bulk of every distribution is comfortably ≥ 20 and only the
  edge-tail is at risk. Conservative: drop q05 on set_01/set_02 in the §3.2 implementation;
  keep q95 on all datasets (max ≥ 40 everywhere).
- **(Blocks Phase 3.3 commitment) Eigvalue extraction shape.** Top-k / bottom-k requires
  fixed k across graphs of varying N. Decision: pad with zeros only if k > N (rare at our
  size), else truncate. Document the convention in the schema bump.
- **(Blocks Phase 3.4.b commitment) Benign edge-AUC saturation.** Per-edge histogram features
  collapse if VGAE achieves edge-AUC > 0.99 on benign — quantiles concentrate at the bounds.
  Audit edge-AUC at extraction time on existing VGAE ckpts before re-extracting with §3.4.b
  features.
- **Storage budget for per-node caches.** Phase 4.1 ≈ 384 MB/dataset, 4.2 ≈ 128 MB/dataset
  (per the research note's appendix: 18 + 80 · 96 ≈ 7700 floats × 7.5K graphs × 4 bytes
  ≈ 232 MB at fp32, ~half at fp16). Total across 4 datasets and 2 models ≈ 1–2 GB at fp16.
  Confirm against `LAKE_ROOT` quota before committing — `gx disk` on the lake-root partition.
- **Variable-N handling at fusion-train time.** Phase 4.2 requires padding-with-mask or PyG
  `Batch`. Decision: stay in PyG ecosystem, fusion variant becomes a `MessagePassing`
  subclass. This is a substantially larger code change than 4.1 — flagged as gating
  decision before committing to Phase 4.2.
- **Subtype label availability for MoE routing.** Phase 5.1 trains end-to-end on binary
  labels; the experts specialize via gradient routing without explicit subtype labels at
  training time. But evaluating whether each expert *did* specialize correctly requires
  subtype labels at test time. Confirm `attack_type` is in the test-phase batch payload —
  `gat.py:288` references `getattr(batch, "attack_type", None)`, so it's optional;
  verify it's populated for can-train-and-test datasets before relying on it.
- **GraphCL is contraindicated for Phase 4.3.** GraphCL's edge-perturbation augmentations
  conflict with the suppress signal (You et al. 2020). Use GraphMAE (Hou et al. 2022) or
  InfoGraph (Sun et al. 2020). Recorded here so the choice doesn't get re-litigated when
  4.3 is picked up.

## File-touch inventory

| Phase | Files |
|---|---|
| 0.1 | `graphids/core/models/fusion/reward.py`, `graphids/_mlflow.py` |
| 0.2 | none — resubmit existing plan |
| 0.3 | `graphids/core/models/supervised/gat.py::test_step` (already wires `attack_type`); fusion test path |
| 1.1 | `graphids/core/models/fusion/reward.py::compute` |
| 1.2 | `graphids/core/models/fusion/reward.py` (additive component) |
| 1.3 | new `analyze` action — Platt fit on val, apply to test |
| 2.1 | new fusion variant or flag — BC pretrain in `fusion/{dqn,bandit}.py::on_fit_start` |
| 2.2 | new `iql.py` / `td3bc.py` under `graphids/core/models/fusion/` |
| 2.3 | new `threshold_policy.py` |
| 3.1 | `gat.py::extract_features` (cosine + L2 + per-node cosine quantiles), `vgae.py::extract_features`, `extract.py::CACHE_VERSION` |
| 3.2 | `gat.py`, `vgae.py` extract_features (replace 4-stat blocks with 5-quantile blocks), `plan/plans/ablations/fusion.py::_state_dim` |
| 3.3 | `vgae.py::extract_features` (top-k eigvalues, gap, VN entropy), or new `graphids/core/data/spectral.py` helper |
| 3.4 | (a) new `motifs.py` helper; (b) `vgae.py::extract_features` (per-edge histogram); (c) new `tda.py` helper + giotto-tda dep |
| 4.1 | `gat.py` forward + extract_features; cache version bump |
| 4.2 | `gat.py`, `vgae.py` extract_features (full `H` return); new fusion variant w/ `GraphAttentionPool` |
| 4.3 | new Stage-1.5 model under `graphids/core/models/ssl/` (GraphMAE or InfoGraph); new pretrain plan; extend `extract.py` to load 3rd model |
| 5.1 | new `moe.py` under `graphids/core/models/fusion/`; supervised, BCE loss |
