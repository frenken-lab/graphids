# Curriculum vs. Static Undersampling — Investigation

> **WORKING DRAFT — 2026-04-25.** In-flight reasoning. Do NOT cite as authoritative.

## TL;DR

- **MLP fusion is fine** — exceeds paper's fixed-weight fusion (attack F1 0.939 vs 0.895).
- **GAT-only regressed** materially vs paper baseline (~7.6 pts acc, ~21 pts attack-F1 on test_01).
- **Three confounders prevent diagnosis:**
  1. Current VGAE test AUC = 0.397 — reconstruction-error signal is noise; any VGAE-driven curriculum/undersampling operates on garbage.
  2. CE / weighted-CE / focal land within 0.001 f1_macro — loss reweighting is doing nothing.
  3. Paper's "GAT-Only" used VGAE-undersampled data; current code has no equivalent.
- **Curriculum-as-implemented underperforms `none`** by 0.20 f1_macro, but with 4+ confounded failure modes — not clean evidence against curriculum-in-general.
- **Decision: drop curriculum** (or run the missing 4-way ablation). Literature actively disfavors balanced→imbalanced at extreme imbalance (5 of 6 head-to-heads contradict); IDS field has converged on constant focal/oversampling.

## 1. Context

Frenken et al. 2025 ([arXiv:2508.04845](https://arxiv.org/abs/2508.04845)) on `set_01` reports Acc 99.38 / F1 89.86 (Ours), 99.29 / 88.08 (KD-GAT). Paper Table 3: GAT-Only F1=0.899, fixed-weight fusion (0.85·GAT + 0.15·VGAE) F1=0.895. Evaluation limited to `test_01_known_vehicle_known_attack`. Stage 1 used VGAE-selective hardest-K undersampling at 4:1 normal:attack ratio.

## 2. Headline current-vs-paper numbers (seed 42, set_01)

| Variant | f1_macro | Notes |
|---|---:|---|
| Paper (S01) | — | Acc 0.9938, F1_attack 0.8986 |
| Current `gat` | 0.6897 | Acc 0.918 / F1_attack 0.686 on test_01 |
| Current `mlp fusion` | **0.9659** | Acc 0.987 / F1_attack 0.939 (overall) |
| `gat_loss/{ce, weighted_ce, focal}` | 0.687 ± 0.001 | All tied |
| `curriculum_random` | 0.4911 | |
| `curriculum_vgae` | 0.4877 | ≡ random (VGAE broken) |
| `unsupervised/vgae` | AUC 0.397 | **Selector is broken** |

## 3. Methodological critiques

**Source-selection bias.** First-pass research agent prompted "does static beat curriculum?" returned 8 tidy citations. Symmetric prompt asking for evidence FOR balanced-warm-up returned a more conflicted synthesis. First pass was confirmation, not review.

**VGAE-quality assumption.** "Static VGAE-undersample beats curriculum" framing assumed current VGAE produces a meaningful difficulty signal. Verified empirically: 1200 epochs of training (loss 2.09→0.64), test AUC=0.397, F1=0.342. AUC < 0.5 means reconstruction error has ~zero discrimination between benign/attack. `score_difficulty()` returns information-free output → `curriculum_vgae` and `curriculum_random` are interchangeable in info content (confirmed: 0.488 vs 0.491). Any claim about VGAE-driven undersampling on this codebase requires the VGAE to actually work, and it doesn't.

## 4. Loss reweighting does nothing

CE / weighted-CE / focal / focal-default tied within 0.001 f1_macro. Three plausible causes: (1) GAT representation has hit a quality ceiling (consistent with Kang 2020 — reweighting helps classifier but hurts representations); (2) decision threshold far from Bayes-optimal so all losses equally suboptimal; (3) f1_macro at 0.5 insensitive to rank-quality. **What this kills:** the prior framing "static-undersample wins because focal/OHEM is the strong baseline" — there is no evidence of OHEM-style hard-mining helping at all here.

## 5. Schedule-direction literature (head-to-head studies)

| Study | Schedule tested | Winning direction | vs proposal |
|---|---|---|---|
| Kang 2020 (Decouple) | inst-bal vs progressively-bal vs class-bal | instance-balanced | Contradicts |
| BBN Zhou 2020 | uniform / natural→reversed / reversed→natural | natural→reversed | Direction-similar early, opposite late |
| DCL Wang 2019 | imbalanced→balanced vs balanced-throughout | imbalanced→balanced | Contradicts |
| LDAM-DRW Cao 2019 | reweight-from-start vs DRW (defer) | DRW | Contradicts |
| Buda 2018 | various two-phase | constant ROS | Contradicts |
| CUDA Ahn 2023 | per-class aug curriculum | stronger aug on heads | Orthogonal |

**5 of 6 contradict the proposal's first phase.** DCL: "training too balanced too early can hurt majority-class representation." Kang: "data imbalance might not be an issue for high-quality representations; instance-balanced sampling … with mere classifier rebalancing." Buda: "two-phase learning with ROS or RUS is not as effective as their plain counterparts."

**Theoretical premise IS validated.** Francazi ICML 2023 ([arXiv:2207.00391](https://arxiv.org/abs/2207.00391)) — minority class suffers higher directional noise under SGD. Fang PNAS 2021 ([Minority Collapse](https://www.pnas.org/doi/10.1073/pnas.2103091118)) — beyond a threshold ratio, minority classifier vectors collapse; 36:1–927:1 is past threshold. But these equally support constant oversampling, focal, LDAM, DRW, BBN.

**Caveat.** Kang/DRW/DCL test ≤1:256; CAN IDS at 1:36–1:927 is at/beyond their range. No published evidence extrapolates either way.

## 6. Current `curriculum_vgae` mechanism

10 tiers of normal samples sorted by VGAE difficulty; tier 0 (hardest) at start; one new tier every ~30 epochs; all 10 by epoch 270. Effective ratio: tier 0 ≈ 3.6:1, tier 9 = 36:1 — approximately balanced→imbalanced. Underperformance vs `none` (0.20 f1_macro hit) is consistent with multiple confounded failure modes: 30-epoch step-function shifts break Adam/LayerNorm; no two-phase decoupling (Kang); no two-branch (BBN); VGAE scorer broken; test data reflects true ratio. Doesn't cleanly distinguish "balanced-first wrong" from "this implementation wrong."

## 7. Verdict

Literature actively disfavors the proposal's first phase (5+ canonical results), mildly disfavors second phase (BBN reverse direction), silent on binary-extreme-imbalance regime. Theoretical premise valid but equally supports better-validated alternatives (focal, LDAM, DRW, BBN, constant oversampling). **No 2022–2026 CAN-bus or NIDS paper uses balanced-warmup-then-shift.** Standard practice: constant SMOTE, constant focal/CB loss, self-supervised pretraining, or static VGAE-selective 4:1 (Frenken 2025).

## 8. Decision and next steps

**Drop curriculum** unless someone runs the missing 4-way ablation: (a) static_undersample_random 4:1, (b) static_undersample_vgae 4:1 (paper's method, requires fixing VGAE first), (c) curriculum_balanced_to_imbalanced (smooth), (d) decoupled_two_phase (Kang). **Prerequisite for any VGAE-dependent variant: fix VGAE AUC=0.397 first** — may be a sign-convention/scaler bug.

Open: why CE/weighted/focal indistinguishable (representation ceiling?); leaderboard f1_macro averages across attack subdirs and hides the test_02–04 unknown-vehicle drop.

## Citations

Buda 2018 [arXiv:1710.05381](https://arxiv.org/abs/1710.05381) · Cao LDAM-DRW NeurIPS 2019 [arXiv:1906.07413](https://arxiv.org/abs/1906.07413) · Cui Class-Balanced CVPR 2019 [arXiv:1901.05555](https://arxiv.org/abs/1901.05555) · Fang Minority Collapse PNAS 2021 [doi](https://www.pnas.org/doi/10.1073/pnas.2103091118) · Francazi ICML 2023 [arXiv:2207.00391](https://arxiv.org/abs/2207.00391) · Frenken 2025 [arXiv:2508.04845](https://arxiv.org/abs/2508.04845) · Kang Decouple ICLR 2020 [arXiv:1910.09217](https://arxiv.org/abs/1910.09217) · Lin Focal ICCV 2017 [arXiv:1708.02002](https://arxiv.org/abs/1708.02002) · Shrivastava OHEM CVPR 2016 [arXiv:1604.03540](https://arxiv.org/abs/1604.03540) · Wang DCL ICCV 2019 [arXiv:1901.06783](https://arxiv.org/abs/1901.06783) · Zhou BBN CVPR 2020 [arXiv:1912.02413](https://arxiv.org/abs/1912.02413) · Ahn CUDA ICLR 2023 [arXiv:2302.05499](https://arxiv.org/abs/2302.05499)
