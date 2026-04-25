# Curriculum vs. Static Undersampling — Investigation

> **WORKING DRAFT — 2026-04-25.** Not set-in-stone documentation. Captures
> in-flight reasoning, conflicting evidence, and open questions from a single
> session diagnosing why current seed-42 ablation results diverge from the
> Frenken et al. 2025 paper baseline. Update or delete as understanding
> sharpens. Do NOT cite as authoritative.

## TL;DR

- **MLP fusion is fine** — exceeds the paper's fixed-weight fusion (attack
  F1 0.939 vs paper 0.895, +4.4 pts).
- **GAT-only regressed materially** vs the paper baseline (~7.6 pts accuracy,
  ~21 pts attack-F1 on test_01_known_vehicle_known_attack).
- **Three confounders make it impossible to currently say WHY:**
  1. Current VGAE has test AUC = 0.397 — its reconstruction-error signal
     is essentially noise, so any VGAE-driven curriculum or undersampling
     is operating on garbage.
  2. CE / weighted-CE / focal loss all land within 0.001 f1_macro on this
     data — loss reweighting is doing nothing.
  3. The paper's "GAT-Only" was trained on VGAE-undersampled data; the
     current code has no equivalent ablation variant.
- **Curriculum-as-implemented underperforms** by 0.20 f1_macro vs no
  curriculum, but the implementation has 4+ confounded failure modes so
  this isn't clean evidence against curriculum-in-general.
- **The literature on schedule direction (balanced→imbalanced vs imbalanced→balanced)
  is more contradictory than supportive of either side at the extreme
  imbalance ratios (36:1–927:1) seen in CAN IDS.** No published study
  runs the 4-way comparison that would settle this.

## 1. Context

Researcher's prior work — Frenken, Bhatti, Zhang, Ahmed (2025), "Multi-Stage
Knowledge-Distilled VGAE and GAT for Robust Controller-Area-Network Intrusion
Detection," [arXiv:2508.04845](https://arxiv.org/abs/2508.04845) — reports
the following on `set_01` (= S01 in paper Table 2):

| Method | Acc | F1 |
|---|---:|---:|
| KD-GAT (prior baseline) | 99.29 | 88.08 |
| **Ours (paper)** | **99.38** | **89.86** |

Paper Table 3 ablation on S01:

| | F1 |
|---|---:|
| GAT-Only | 0.899 |
| Fusion (fixed-weight 0.85·GAT + 0.15·VGAE) | 0.895 |

Paper § "Score Fusion" — fusion is **fixed-weight**: P_fused = ω_anomaly·P_VGAE +
ω_GAT·P_GAT, ω_anomaly=0.15, ω_GAT=0.85, chosen empirically on validation.

Paper § Datasets — "This work will limit evaluation to the known vehicle and
attack testing set" — i.e. test_01_known_vehicle_known_attack only.

Paper § 4.2 Stage 1 — VGAE-selective hardest-K undersampling: "we implement
selective undersampling based solely on reconstruction errors:
R_error(i) = ||A_i − Â_i||₂ … This identifies normal samples with highest
reconstruction errors—those most difficult to reconstruct and likely on
decision boundaries—maintaining a 4:1 normal-to-attack ratio for Stage 2."

## 2. Current seed-42 results vs paper

`compare leaderboard` outputs (N=1, no CIs) — **all on `set_01`**.

### Overall test (averaged across 4–5 attack subdirs)

| Group | Variant | f1_macro | accuracy | f1_attack |
|---|---|---:|---:|---:|
| conv_type | gps | 0.7264 | — | — |
| conv_type | gat | 0.6897 | 0.7595 | 0.5425 |
| conv_type | gatv2 | 0.6872 | — | — |
| gat_loss | focal | 0.6875 | — | — |
| gat_loss | ce | 0.6870 | — | — |
| gat_loss | weighted_ce | 0.6869 | — | — |
| gat_sampling | none (=focal default) | 0.6868 | — | — |
| gat_sampling | curriculum_random | 0.4911 | 0.778 | — |
| gat_sampling | curriculum_vgae | 0.4877 | 0.614 | — |
| id_encoding | hash | 0.7446 | — | — |
| id_encoding | lookup | 0.6871 | — | — |
| id_encoding | learned_unk | 0.6702 | — | — |
| fusion | mlp | **0.9659** | **0.9868** | **0.9392** |
| fusion | bandit | 0.9524 | 0.9821 | — |
| fusion | dqn | 0.9524 | — | — |
| fusion | weighted_avg | **0.10** ⚠ | 0.111 | — |
| unsupervised | dgi | 0.4270 (auc) | 0.7175 | — |
| unsupervised | gae | 0.4012 (auc) | 0.2395 | — |
| unsupervised | vgae | 0.3971 (auc) | 0.2393 | — |

### test_01_known_vehicle_known_attack only (paper-comparable subset)

| Variant | Accuracy | F1 attack | F1 macro |
|---|---:|---:|---:|
| **Paper Ours (S01)** | **0.9938** | **0.8986** | — |
| **Paper KD-GAT (S01)** | **0.9929** | **0.8808** | — |
| Current `gat` | 0.918 | 0.686 | 0.820 |
| Current `mlp fusion` | 0.987 (overall avg) | 0.939 (overall) | 0.966 |

## 3. Two methodological critiques the researcher raised

**Critique 1: source-selection bias.** A first-pass research agent dispatched
with the prompt shape "research static undersampling vs curriculum learning
for imbalanced graph classification — does static beat curriculum?" returned
a tidy 8-citation answer. **The prompt was biased.** A symmetric prompt that
explicitly asked the agent to find evidence FOR balanced-warm-up curriculum
returned a different — and more conflicted — synthesis. The first pass was
not a fair literature review; it was confirmation. The corrected synthesis
is below.

**Critique 2: VGAE quality assumption.** The "static VGAE-undersample beats
current curriculum" framing assumed the current VGAE produces a meaningful
difficulty signal like the paper's did. Verified empirically:

- VGAE seed-42 fit: 1200 epochs, train_loss 2.09→0.64, val_loss 2.07→0.63
  (training did happen).
- VGAE seed-42 test: AUC=0.397, F1=0.342, accuracy=0.239.
- Test AUC < 0.5 means the VGAE reconstruction error has near-zero (or
  weakly anti-correlated) discrimination between benign and attack on
  this dataset. The "difficulty score" returned by `score_difficulty()`
  is information-free.

Therefore: `curriculum_vgae` and `curriculum_random` are interchangeable in
information content here, and the empirical numbers confirm it
(0.488 vs 0.491 f1_macro). Any claim about "VGAE-driven undersampling"
on the current codebase requires the VGAE to actually work — and it
doesn't.

## 4. New empirical finding — loss reweighting does nothing on this data

| variant | loss | f1_macro |
|---|---|---:|
| gat_loss/ce | vanilla CE | 0.6870 |
| gat_loss/weighted_ce | inverse-frequency CE | 0.6869 |
| gat_loss/focal | focal (γ=2) | 0.6875 |
| gat_sampling/none | focal (default) | 0.6868 |

All four loss/sampling variants land within 0.001 of each other.

This is informative. It means **loss-side imbalance handling is not the
bottleneck**. Three plausible explanations:

1. The GAT representation has hit a quality ceiling that loss reweighting
   can't break (consistent with Kang et al. 2020 ICLR finding that
   reweighting helps the classifier but hurts representations).
2. The decision threshold (0.5) is far enough from Bayes-optimal that all
   three losses are equally suboptimal.
3. The test metric (f1_macro at threshold 0.5) is insensitive to
   rank-quality differences.

What this kills: the prior framing "static-undersample wins because OHEM /
focal is the strong baseline" — there is **no evidence of OHEM-style
hard-mining helping on this data at all.**

## 5. First-principles walkthrough — each approach

For each: **mechanism** (what actually happens during training),
**predicted failure mode**, **predicted success mode**, **citations**.
Marked **[V]** for verifiable from cited paper, **[R]** for reasoning from
premise, **[S]** for speculation.

### 5.1 Naive imbalanced training (true ratio + CE/BCE)

**Mechanism.** For each minibatch B: gradient = (1/|B|) Σ ∇L(xᵢ, yᵢ). With
36:1 imbalance, expected batch = 36 benigns + 1 attack. As benigns become
easy (loss → 0), per-sample gradient shrinks but their count stays high.

**Predicted bias [V].** Buda, Maki, Mazurowski (2018) "A systematic study of
the class imbalance problem in convolutional neural networks," Neural
Networks 106, [arXiv:1710.05381](https://arxiv.org/abs/1710.05381) — at
imbalance ratios from 1:1 to 1:5000, CNN decision boundaries shift toward
majority; minority recall collapses past ~1:10 without intervention.

**Empirical check on this dataset [V].** `gat_loss/ce` (vanilla CE):
f1_macro=0.687, recall(attack)=0.63. **The model is biased but not
catastrophically collapsed.** The researcher's premise that "naive training
collapses to all-benign" doesn't strictly hold here.

### 5.2 Loss reweighting (focal / class-balanced / weighted-CE)

**Mechanisms:**
- Weighted CE: Lᵢ = wᵧᵢ · CE, wᵧᵢ ∝ 1/Nᵧᵢ. Each minority sample gradient
  scaled by imbalance ratio.
- Focal **[V]**: L = -α(1-p_t)^γ log(p_t). Modulator (1-p_t)^γ → 0 for easy
  examples → implicit on-the-fly OHEM. Lin, Goyal, Girshick, He, Dollár
  ICCV 2017, [arXiv:1708.02002](https://arxiv.org/abs/1708.02002).
- Class-Balanced **[V]**: w_y = (1-β)/(1-β^Nᵧ). Cui, Jia, Lin, Song,
  Belongie CVPR 2019, [arXiv:1901.05555](https://arxiv.org/abs/1901.05555).

**Predicted limitation [V].** Kang, Xie, Rohrbach et al. ICLR 2020
"Decoupling Representation and Classifier for Long-Tailed Recognition,"
[arXiv:1910.09217](https://arxiv.org/abs/1910.09217) — loss reweighting
helps classifier but hurts representations on long-tail benchmarks. They
recommend two-stage training (instance-balanced for representations +
classifier-only rebalancing).

**Empirical check on this dataset.** ce/weighted_ce/focal all tied at 0.687.
Loss reweighting is doing nothing here.

### 5.3 Static informed undersampling (paper's VGAE-4:1)

**Mechanism.** Offline: train VGAE on benigns. Score every benign by
reconstruction error. Pick top-K benigns where K = 4 × N_attack. Train GAT
on the fixed (K benigns + N attacks) set every epoch.

**Equivalence [V].** This is offline OHEM. Shrivastava, Gupta, Girshick CVPR
2016, [arXiv:1604.03540](https://arxiv.org/abs/1604.03540) — keep highest-
loss negatives, discard easy. Static-VGAE-4:1 does the same offline with
a fixed selector.

**Critical dependency [V on this data].** Selector must produce meaningful
difficulty scores. Current VGAE test AUC=0.397 → selector is broken →
"hardest" ≈ random. Confirmed by curriculum_vgae≡curriculum_random
empirical result.

**Status.** No current ablation variant in the codebase implements static
undersampling. Was deleted/replaced during the curriculum-learning research
pivot. Re-adding it is recommendation #1.

### 5.4 Static random undersampling (4:1 random)

**Mechanism.** Same 4:1 ratio, random benigns each epoch.

**[R].** With current broken VGAE, this should match informed undersampling.
With paper's working VGAE, informed should beat random by exactly the
discriminative power of the VGAE score.

**[V].** Drummond & Holte 2003 "C4.5, Class Imbalance, and Cost Sensitivity:
Why Under-Sampling beats Over-Sampling,"
[cs.toronto.edu/~holte/Publications/papers/icml-kdd03.pdf](https://www.cs.toronto.edu/~holte/Publications/papers/icml-kdd03.pdf)
— random undersampling beats SMOTE-style oversampling on tree classifiers.

### 5.5 Curriculum: balanced → imbalanced (researcher's proposal)

**Mechanism (proper implementation).** Sample at 1:1 attack:benign for first
phase → linearly shift toward true ratio over middle phase → train at true
ratio for tail phase.

**Researcher's stated argument:**
> "Naive training has GAT bias to classify everything as benign. Starting
> balanced makes sure it could understand attacks before approaching true
> ratios. Starting unbalanced would be like naive training, and as it
> progresses slowly upweights attack samples due to ratio."

**Direction-similar literature support [V — partial].** Bilateral-Branch
Network (BBN), Zhou, Cui, Yang, Liu, Yu CVPR 2020,
[arXiv:1912.02413](https://arxiv.org/abs/1912.02413) — two branches
(uniform + reversed sampling) combined with α(t) shifting attention from
uniform→reversed over training. **But: BBN ends with MORE minority emphasis,
not less, as the proposal does.** Mechanism is also two-branch, not
batch-composition shift on a single backbone.

**Direct literature CONTRADICTION [V — substantial].**
- DRW (Cao et al. NeurIPS 2019, [arXiv:1906.07413](https://arxiv.org/abs/1906.07413)):
  train at TRUE RATIO with UNIFORM weights first, then defer reweighting to
  late phase. **Direction inverse to proposal.** Authors: applying
  reweighting too early corrupts representation learning.
- Kang et al. ICLR 2020 (cited above): instance-balanced sampling
  (=true ratio) gives BETTER representations than class-balanced. Direct
  quote: "data imbalance might not be an issue learning high-quality
  representations; with simple instance-balanced sampling the representations
  can produce a strong classification head with mere classifier rebalancing."
  **Direction inverse to proposal.**
- DCL (Wang et al. ICCV 2019, [arXiv:1901.06783](https://arxiv.org/abs/1901.06783)):
  schedules imbalanced→balanced, NOT balanced→imbalanced. Direct quote:
  "training too balanced too early can hurt majority-class representation"
  and "keeping targeting at a balanced distribution in the whole process
  would hurt the generalization ability, particularly for a largely
  imbalanced task." **Direction inverse to proposal.**
- Buda et al. 2018 (cited above): "two-phase learning with ROS or RUS is not
  as effective as their plain ROS and RUS counterparts." **Negative on
  two-phase strategies generally.**
- Shi & Wei NeurIPS 2023 "How Re-sampling Helps for Long-Tail Learning?"
  [proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/file/eeffa70bcbbd43f6bd067edebc6595e8-Paper-Conference.pdf):
  "uniform sampling mainly learns label-relevant features, while re-sampling
  overfits the label-irrelevant features." Class-balanced re-sampling can
  hurt by overfitting tail context.

**[S]** The Kang/DRW/DCL results are on Long-Tail CIFAR / ImageNet-LT (max
imbalance ~1:256). CAN IDS at 1:36 to 1:927 is at or beyond their tested
range. There is no published evidence that says these results extrapolate
to extreme imbalance — so the proposal's direction may genuinely be valid
in this regime, but there's no paper that says so.

**Theoretical premise IS validated [V — partial].**
- Francazi et al. ICML 2023, [arXiv:2207.00391](https://arxiv.org/abs/2207.00391)
  prove that under SGD with class imbalance, "the minority class suffers
  from a higher directional noise" — the early-training majority bias is
  real.
- Fang et al. PNAS 2021 "Minority Collapse,"
  [doi:10.1073/pnas.2103091118](https://www.pnas.org/doi/10.1073/pnas.2103091118)
  prove that beyond a finite imbalance ratio, minority classifier vectors
  collapse — predicting that 36:1–927:1 are well past the threshold.

These results justify *some* intervention against majority dominance, but
do NOT specifically endorse balanced→unbalanced; they're equally consistent
with constant oversampling, focal, LDAM, or DRW.

### 5.6 Curriculum: imbalanced → balanced (DCL direction)

**Mechanism [V].** Wang et al. ICCV 2019 (cited above). Schedule sampling
distribution from natural imbalance toward balanced over epochs.

**Researcher's critique:** "This is essentially naive imbalanced training
early, which already fails. The model collapses to majority before
rebalancing has any effect."

**Counter-argument from DCL paper:** "training too balanced too early can
hurt majority-class representation."

**[S]** For mild imbalance (ImageNet-LT-style 1:256) this might apply; for
1:36+ it's unclear because there's so much majority data that "preserving
majority structure" isn't the bottleneck — minority discrimination is.

### 5.7 Curriculum: hard→easy on benigns (current `curriculum_vgae`)

**Mechanism (verified from `core/data/curriculum.py`).** 10 tiers of normal
samples sorted by VGAE difficulty; tier 0 (hardest) active at start; one
new tier unlocked every ~30 epochs; all 10 active by epoch 270. Attacks
always included.

**Effective ratio [R]:**
- Tier 0 (1/10 benigns + all attacks): ratio ≈ (N_benign/10) : N_attack
  ≈ 3.6:1 (true is 36:1). MORE BALANCED early.
- Tier 9 (all benigns): ratio = 36:1. TRUE IMBALANCE late.

**This IS approximately balanced → imbalanced in effective class ratio.**
Directionally the same as the researcher's proposal. So when this curriculum
underperforms `none` by 0.20 f1_macro, it's evidence against this specific
implementation, but **NOT strong evidence against the balanced→imbalanced
direction in general**, because:

- 30-epoch tier-unlock is a *step function*, not smooth. Sharp distribution
  shifts break Adam momentum + LayerNorm running stats.
- No two-phase decoupling (Kang 2020-style classifier reset).
- No two-branch combination (BBN-style).
- VGAE scorer is broken so within-tier ordering is noise.
- Test data still reflects true ratio; mid-training balanced phase may
  cause boundary that doesn't transfer.

So `curriculum_vgae` failure is consistent with **multiple confounded
failure modes** and doesn't distinguish "balanced-first is wrong" from
"this implementation is wrong."

### 5.8 Decoupled two-phase (Kang 2020) and DRW (Cao 2019)

**Kang mechanism [V].** Phase 1: train backbone + classifier on
instance-balanced sampling. Phase 2: freeze backbone, retrain classifier
with class-balanced sampling OR τ-normalize classifier weights.

**DRW mechanism [V].** Phase 1: train normally with uniform weights at
true ratio. Phase 2: switch to class-balanced reweighted loss for last
~20% of epochs.

**Common finding [V, both papers].** Decoupling representation learning
from classifier rebalancing gives 5–10% accuracy gains over single-stage
rebalancing on Long-Tail CIFAR/ImageNet-LT.

**Implication for the current codebase [R].** Even if balanced-warmup
curriculum is adopted, the Kang/DRW evidence suggests **the right place to
inject class balancing is at the classifier head, not the data sampler**.
A single-backbone tier-unlock can't decouple these.

### 5.9 Bilateral-Branch Network (BBN) — partial direction-match

**[V].** Zhou et al. CVPR 2020 (cited above). Cumulative learning
α-schedule: "first learn the universal patterns and then pay attention to
the tail data gradually." Two branches: uniform-sampling "conventional"
branch + reversed-sampling "re-balancing" branch. Alpha shifts attention
from conventional (early) to re-balancing (late).

**Important caveat.** BBN goes natural → reversed, ending with MORE minority
emphasis. The researcher's proposal goes balanced → imbalanced, ending with
LESS minority emphasis. **Direction-similar in spirit; mechanism opposite at
the late phase.**

## 6. Direct schedule-direction comparisons (head-to-head studies)

| Study | Schedule tested | Winning direction | Implication for proposal |
|---|---|---|---|
| Kang 2020 (Decouple) | inst-bal vs progressively-balanced (toward class-bal) vs class-bal | instance-balanced (=natural) | Contradicts |
| BBN Zhou 2020 | constant uniform vs cumulative natural→reversed vs reversed→natural | natural→reversed | Direction-similar; mechanism opposite at late phase |
| DCL Wang 2019 | imbalanced→balanced; balanced-throughout | imbalanced→balanced | Contradicts |
| LDAM-DRW Cao 2019 | reweight-from-start vs DRW (defer to late) | DRW | Contradicts |
| Buda 2018 | various two-phase | constant ROS | Contradicts |
| CUDA Ahn 2023 ([arXiv:2302.05499](https://arxiv.org/abs/2302.05499)) | per-class augmentation curriculum | stronger aug on heads | Orthogonal but contradicts intuition |

**5 of 6 head-to-heads contradict the proposal's first phase.** BBN
contradicts the proposal's late phase.

## 7. CAN IDS / network intrusion detection — what the field actually does

**[V].** No 2022–2026 CAN-bus or NIDS paper found that uses a
balanced-warmup-then-shift schedule. Standard practice:
- Constant SMOTE/ADASYN/Borderline-SMOTE oversampling
  ([DOI:10.1145/3697467.3697595](https://dl.acm.org/doi/10.1145/3697467.3697595)).
- Constant focal/CB loss.
- Self-supervised pre-training on benign-only (CGTS,
  [Cybersecurity 2025](https://link.springer.com/article/10.1186/s42400-025-00365-6),
  uses one-class SVDD).
- Frenken et al. 2025 (the researcher's prior): static VGAE-selective 4:1.

**The minority position (balanced→imbalanced curriculum at extreme
imbalance) is genuinely untested in the IDS subfield.**

## 8. GAT / graph attention — schedule findings

**[V].** Surveyed: GAT-RWOS ([arXiv:2412.16394](https://arxiv.org/html/2412.16394v1))
uses GAT attention to guide oversampling but with constant balanced sampling.
GraphSMOTE ([arXiv:2103.08826](https://arxiv.org/abs/2103.08826)) and
GraphSHA ([arXiv:2306.09612](https://arxiv.org/pdf/2306.09612)) augment
minority nodes with constant ratios. The "Curriculum Graph ML Survey"
([arXiv:2302.02926](https://arxiv.org/html/2302.02926)) catalogs
sample-difficulty curricula on graphs but no class-ratio schedules.
**No published GAT-specific schedule-direction comparisons.**

## 9. Honest verdict

The literature actively disfavors the proposal's first phase
(balanced warm-up) by 5+ canonical results, mildly disfavors its second
phase (shift toward true ratio) by BBN's reverse direction, and is silent
on the specific binary-extreme-imbalance regime where it might still work.

The proposal's **theoretical premise** — that early SGD favors the majority
class and minority collapse is real at extreme imbalance — is independently
validated (Francazi 2023, Fang 2021). But this premise equally supports
constant oversampling, focal loss, LDAM, DRW, or BBN — all of which are
better-validated than balanced→imbalanced scheduling.

This is **not** "the proposal is wrong." It's: **the proposal has weaker
literature backing than the alternatives, and would need the specific
4-way ablation that nobody has run to be defensible in a paper.**

The empirical observation in the current ablation (`curriculum_vgae` =
0.488 vs `none` = 0.687, a 0.20 f1_macro hit) is consistent with multiple
failure modes and does not cleanly settle "balanced-first is wrong" —
but it's also not evidence that the direction works.

## 10. What's missing from the literature that would actually settle this

1. A **direct 4-way ablation** at extreme imbalance (>100:1) comparing:
   (a) balanced→imbalanced, (b) imbalanced→balanced (DCL),
   (c) constant LDAM/focal, (d) Kang-style decoupled.
2. **Theoretical analysis of the deployment-distribution-fine-tune phase**:
   does shifting back to imbalance late actually fine-tune the boundary, or
   does it cause catastrophic forgetting of minority features learned during
   the balanced warm-up? Continual-learning literature suggests the latter
   is a real risk.
3. **Binary extreme-imbalance graph classification specifically.** None of
   the canonical long-tail papers tackle 36:1–927:1 binary at the graph
   level. CAN IDS is precisely this regime; field has converged on constant
   rebalancing.

## 11. Concrete next-step options

**A. Restore the paper baseline.** Re-add static VGAE-undersampling at fixed
4:1 as a `gat_sampling/static_undersample_vgae` ablation. **Prerequisite:
fix the VGAE first** (current AUC=0.397 means the selector is broken; this
must be resolved before any VGAE-driven undersampling result is meaningful).

**B. Run the missing 4-way comparison.** Implement four variants:
- `static_undersample_random` (4:1 random — control)
- `static_undersample_vgae` (4:1 informed — paper's method)
- `curriculum_balanced_to_imbalanced` (researcher's proposal, smooth shift)
- `decoupled_two_phase` (Kang 2020-style: instance-balanced + classifier reset)

Plus the current `none` (focal full-data) as baseline. This is the
literature-missing comparison; running it on N=3 seeds at set_01 would
genuinely contribute beyond the existing ablation.

**C. Investigate the VGAE AUC=0.397 first.** It may be a sign convention
or scaler bug; could be a one-line fix. Until VGAE is recovered, options A
and B (the VGAE-dependent variants) cannot be fairly tested.

## 12. Open questions

- **Why are CE / weighted_CE / focal indistinguishable on this data?** This
  is the most suspicious finding. If the GAT representation has hit a
  ceiling, no sampling strategy will help; the bottleneck is architecture
  or unsupervised pretraining, not loss/sampling.
- **Is the test_01 metric (overall test) being computed correctly given
  multiple test subdirs?** The compare leaderboard f1_macro = 0.687 averaged
  across 4-5 attack subdirs; the per-subdir test_01 = 0.918 accuracy. The
  large drop on test_02–04 (unknown vehicles / unknown attacks) is the real
  story but the leaderboard hides it.
- **Why does VGAE test AUC = 0.397?** Bug, sign convention, or genuinely
  uninformative reconstruction error on this dataset?
- **Did the paper's evaluation use overall test or test_01 only?** Paper
  says "limit to known vehicle and attack" — so test_01. This is the right
  comparison.

## 13. Citations index (alphabetical by first author)

- Ahn et al., ICLR 2023 — CUDA per-class augmentation curriculum.
  [arXiv:2302.05499](https://arxiv.org/abs/2302.05499)
- Buda, Maki, Mazurowski 2018, Neural Networks 106 — systematic CNN
  imbalance study. [arXiv:1710.05381](https://arxiv.org/abs/1710.05381)
- Cao, Wei, Gaidon, Arechiga, Ma, NeurIPS 2019 — LDAM-DRW.
  [arXiv:1906.07413](https://arxiv.org/abs/1906.07413)
- Cui, Jia, Lin, Song, Belongie, CVPR 2019 — Class-Balanced Loss.
  [arXiv:1901.05555](https://arxiv.org/abs/1901.05555)
- Drummond & Holte, ICML Workshop 2003 — under-sampling beats
  over-sampling.
  [cs.toronto.edu/~holte/Publications/papers/icml-kdd03.pdf](https://www.cs.toronto.edu/~holte/Publications/papers/icml-kdd03.pdf)
- Fang et al., PNAS 2021 — Minority Collapse.
  [doi:10.1073/pnas.2103091118](https://www.pnas.org/doi/10.1073/pnas.2103091118)
- Francazi et al., ICML 2023 — directional noise in imbalanced SGD.
  [arXiv:2207.00391](https://arxiv.org/abs/2207.00391)
- Frenken, Bhatti, Zhang, Ahmed 2025 — KD-GAT paper.
  [arXiv:2508.04845](https://arxiv.org/abs/2508.04845)
- Hacohen & Weinshall, ICML 2019 — curriculum learning.
  [arXiv:1904.03626](https://arxiv.org/abs/1904.03626)
- Kang, Xie, Rohrbach et al., ICLR 2020 — Decoupling Representation and
  Classifier. [arXiv:1910.09217](https://arxiv.org/abs/1910.09217)
- Lin, Goyal, Girshick, He, Dollár, ICCV 2017 — Focal Loss.
  [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)
- Liu, Ao, Qu et al., 2023 — Class-Imbalanced Learning on Graphs survey.
  [arXiv:2304.04300](https://arxiv.org/abs/2304.04300)
- Shi & Wei, NeurIPS 2023 — How Re-sampling Helps for Long-Tail Learning.
  [proceedings link](https://proceedings.neurips.cc/paper_files/paper/2023/file/eeffa70bcbbd43f6bd067edebc6595e8-Paper-Conference.pdf)
- Shrivastava, Gupta, Girshick, CVPR 2016 — OHEM.
  [arXiv:1604.03540](https://arxiv.org/abs/1604.03540)
- Soviany, Ionescu, Rota, Sebe, IJCV 2022 — curriculum learning survey.
  [arXiv:2101.10382](https://arxiv.org/abs/2101.10382)
- Wang, Gan, Yang, Wu, Yan, ICCV 2019 — DCL.
  [arXiv:1901.06783](https://arxiv.org/abs/1901.06783)
- Wei et al., 2024 — loss-based sample selection failure under imbalance.
  [arXiv:2402.11242](https://arxiv.org/abs/2402.11242)
- Wu, Dyer, Neyshabur, ICLR 2021 — When Do Curricula Work?
  [arXiv:2012.03107](https://arxiv.org/abs/2012.03107)
- Zhao, Zhang, Wang, WSDM 2021 — GraphSMOTE.
  [arXiv:2103.08826](https://arxiv.org/abs/2103.08826)
- Zhou, Cui, Yang, Liu, Yu, CVPR 2020 — BBN.
  [arXiv:1912.02413](https://arxiv.org/abs/1912.02413)
