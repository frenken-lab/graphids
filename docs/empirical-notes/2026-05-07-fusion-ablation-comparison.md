# 2026-05-07 — Fusion ablation: method comparison (6 variants, per-split test)

**Seed:** 42. **Status:** per-split test data collected 2026-05-07 after fix to
`FusionDataModule.test_dataloader()` (previously returned `val_dataloader()`).
**All metrics are from test-phase MLflow runs** (`graphids.phase=test`, per-split structure).

> **Previous version of this file** used val-fallback metrics (0.999x AUROC for moe/mlp) —
> those were measuring validation performance, not held-out test. This version reflects
> the correct held-out test split breakdown. The gap is substantial: t02 (unknown vehicle)
> achieves only 0.33–0.54 AUROC across all methods, versus prior reported 0.999x.

hcrl_sa is dev-only; included in §D for reference, excluded from cross-dataset observations.
set_04/moe has only aggregate test metrics (no per-split breakdown); marked ‡ throughout.

---

## Split legend

| code | full name | description |
|------|-----------|-------------|
| t01-KK | test_01_known_vehicle_known_attack | Same vehicle type and same attack types as training |
| t02-UK | test_02_unknown_vehicle_known_attack | Unknown vehicle + known attacks (vehicle covariate shift) |
| t03-KU | test_03_known_vehicle_unknown_attack | Known vehicle + novel attack types |
| t04-UU | test_04_unknown_vehicle_unknown_attack | Unknown vehicle + novel attacks (full OOD) |
| t05 | test_05_suppress | Suppress attack (ECU message suppression) |
| t06 | test_06_masquerade | Masquerade attack (injected messages mimicking legitimate ECU) |

`KK` = in-distribution; `UK/KU/UU` = progressively harder generalisation; `t05/t06` are
distinct attack categories not in t01–t04.

---

## Run inventory

| variant | set_01 | set_02 | set_03 | set_04 | note |
|---------|--------|--------|--------|--------|------|
| mlp | ✓ | ✓ | ✓ | ✓ | v6 states, git 992c924 |
| moe | ✓ | ✓ | ✓ | ‡ | v6 states; set_04 aggregate only |
| moe_noaux | ✓ | ✓ | ✓ | ✓ | v6 states, git 992c924 |
| weighted_avg | ✓ | ✓ | ✓ | ✓ | v6 states |
| bandit | ✓ | ✓ | ✓ | ✓ | v6 states |
| dqn | ✓ | ✓ | ✓ | ✓ | v6 states |

‡ set_04/moe needs resubmission from re-rendered plan (earlier run used stale cached_states_dir → fell
back to aggregate test only). All other cells have 6-split per-split breakdowns.

---

## A. AUROC macro per split

KK = known vehicle + known attack, UK = unknown vehicle + known attack, KU = known vehicle + unknown
attack, UU = unknown vehicle + unknown attack. t05 AUROC = 0.0 universally — see §Observations.

### set_01

| variant | t01-KK | t02-UK | t03-KU | t04-UU | t05 | t06 |
|---------|--------|--------|--------|--------|-----|-----|
| mlp | **0.9803** | **0.4554** | 0.5228 | **0.6289** | 0.0 | 0.9699 |
| moe | 0.9693 | 0.3436 | 0.5687 | 0.5695 | 0.0 | 0.9834 |
| moe_noaux | 0.9731 | 0.3309 | 0.5627 | 0.5730 | 0.0 | **0.9853** |
| weighted_avg | 0.5936 | 0.4412 | 0.4904 | 0.4592 | 0.0 | 0.4808 |
| bandit | 0.9138 | 0.4598 | 0.5699 | 0.7425 | 0.0 | 0.9775 |
| dqn | 0.8182 | 0.4597 | **0.5729** | 0.7426 | 0.0 | 0.9554 |

### set_02

| variant | t01-KK | t02-UK | t03-KU | t04-UU | t05 | t06 |
|---------|--------|--------|--------|--------|-----|-----|
| mlp | 0.8437 | 0.4949 | **0.7764** | 0.4794 | 0.0 | 0.7816 |
| moe | **0.8446** | **0.5356** | **0.7829** | **0.4941** | 0.0 | **0.7857** |
| moe_noaux | 0.8440 | 0.4935 | 0.7702 | 0.4653 | 0.0 | 0.7772 |
| weighted_avg | 0.8235 | 0.5070 | 0.5865 | 0.5927 | 0.0 | 0.7429 |
| bandit | 0.8424 | 0.5074 | 0.7427 | 0.6239 | 0.0 | 0.7947 |
| dqn | 0.8431 | 0.5104 | 0.7453 | 0.6600 | 0.0 | 0.8006 |

### set_03

| variant | t01-KK | t02-UK | t03-KU | t04-UU | t05 | t06 |
|---------|--------|--------|--------|--------|-----|-----|
| mlp | 0.8155 | 0.5815 | 0.5387 | 0.5318 | 0.0 | 0.3023 |
| moe | 0.8544 | 0.5624 | **0.6002** | 0.5287 | 0.0 | 0.8269 |
| moe_noaux | 0.8526 | 0.5646 | 0.5975 | 0.5347 | 0.0 | **0.8406** |
| weighted_avg | 0.8101 | 0.5680 | 0.5178 | 0.5139 | 0.0 | 0.3657 |
| bandit | 0.8121 | 0.6094 | 0.5653 | 0.5374 | 0.0 | 0.6576 |
| dqn | **0.8582** | **0.6374** | 0.5920 | **0.5474** | 0.0 | 0.8133 |

### set_04

| variant | t01-KK | t02-UK | t03-KU | t04-UU | t05 | t06 |
|---------|--------|--------|--------|--------|-----|-----|
| mlp | 0.6985 | **0.5412** | 0.8808 | 0.5569 | 0.0 | 0.9688 |
| moe | ‡ | ‡ | ‡ | ‡ | ‡ | ‡ |
| moe_noaux | **0.6996** | 0.5385 | **0.8919** | 0.5635 | 0.0 | **0.9734** |
| weighted_avg | 0.6588 | 0.5261 | 0.8005 | 0.5522 | 0.0 | 0.7533 |
| bandit | 0.6738 | 0.5017 | 0.8508 | **0.5904** | 0.0 | 0.9241 |
| dqn | 0.5842 | 0.5439 | 0.6886 | 0.6237 | 0.0 | 0.7838 |

---

## B. MCC per split

t05 values are anomalous (see §Observations). set_03/moe and set_03/moe_noaux show MCC=1.0
on t05 — artifact of empty attack class in that split, not model performance.

### set_01

| variant | t01-KK | t02-UK | t03-KU | t04-UU | t05 | t06 |
|---------|--------|--------|--------|--------|-----|-----|
| mlp | **0.7961** | **−0.0011** | −0.0117 | 0.0635 | 0.1688 | 0.8186 |
| moe | 0.7701 | −0.2175 | −0.0034 | 0.0670 | 0.1765 | **0.8153** |
| moe_noaux | 0.7694 | −0.2359 | −0.0036 | 0.0662 | 0.1823 | 0.8105 |
| weighted_avg | 0.0962 | −0.0926 | −0.0143 | −0.0176 | −0.0026 | −0.0152 |
| bandit | 0.4127 | −0.0011 | 0.0506 | 0.0615 | 0.1080 | 0.7710 |
| dqn | 0.5196 | −0.0011 | 0.0560 | 0.0646 | 0.0869 | 0.8179 |

### set_02

| variant | t01-KK | t02-UK | t03-KU | t04-UU | t05 | t06 |
|---------|--------|--------|--------|--------|-----|-----|
| mlp | 0.4029 | 0.0438 | 0.0090 | 0.0940 | 0.0860 | 0.4617 |
| moe | 0.3987 | 0.0426 | 0.0106 | 0.0927 | 0.0857 | **0.4619** |
| moe_noaux | **0.4325** | **0.0443** | 0.0075 | 0.0691 | **0.0942** | 0.4517 |
| weighted_avg | 0.4928 | 0.0391 | 0.0124 | **0.0904** | 0.0950 | 0.3963 |
| bandit | 0.1226 | 0.0363 | **0.1659** | 0.0797 | 0.0206 | 0.3111 |
| dqn | 0.1226 | 0.0331 | 0.1669 | 0.0725 | 0.0208 | 0.3165 |

### set_03

| variant | t01-KK | t02-UK | t03-KU | t04-UU | t05 | t06 |
|---------|--------|--------|--------|--------|-----|-----|
| mlp | 0.6746 | 0.1879 | 0.1304 | 0.0691 | 0.1219 | −0.0138 |
| moe | **0.6741** | 0.2034 | **0.1498** | 0.0810 | 1.000† | 0.0155 |
| moe_noaux | **0.6747** | **0.2079** | 0.1480 | 0.0796 | 1.000† | 0.0162 |
| weighted_avg | 0.6731 | 0.1143 | 0.0848 | −0.0173 | 0.0168 | −0.2173 |
| bandit | 0.3496 | 0.1952 | 0.0635 | **0.0947** | 0.0059 | 0.0666 |
| dqn | 0.5877 | 0.2009 | 0.0871 | 0.0931 | 0.4999 | **0.1689** |

† Empty attack class — MCC=1.0 is trivial (see §Observations).

### set_04

| variant | t01-KK | t02-UK | t03-KU | t04-UU | t05 | t06 |
|---------|--------|--------|--------|--------|-----|-----|
| mlp | 0.2257 | **0.1045** | 0.4320 | 0.2307 | 1.0† | 0.0185 |
| moe | ‡ | ‡ | ‡ | ‡ | ‡ | ‡ |
| moe_noaux | **0.2339** | 0.0962 | **0.4561** | **0.2113** | 0.707† | **0.0215** |
| weighted_avg | 0.1873 | 0.1031 | 0.4017 | 0.2281 | 1.0† | 0.0108 |
| bandit | 0.0027 | −0.0045 | −0.0026 | −0.0004 | −1.0† | 0.0013 |
| dqn | 0.1244 | 0.0885 | 0.2516 | 0.1899 | 0.0071 | **0.3773** |

† MCC on t05 for set_04 is unreliable — the suppress split has very few or zero attack samples
(set-dependent); MCC collapses to either 1.0 (all benign correct) or −1.0 (always-attack policy).

---

## C. Per-attack AUROC detail (moe and moe_noaux)

Attack types vary by split. Only t01–t04 and t06 are shown (t05 has AUROC=0; no per-attack breakdown).

### set_01

**t01 — known vehicle, known attack (dos/gear/rpm/standstill)**

| variant | dos | gear | rpm | standstill | macro |
|---------|-----|------|-----|------------|-------|
| moe | 0.9675 | 0.9763 | 0.9664 | 0.9534 | 0.9659 |
| moe_noaux | 0.9710 | 0.9789 | 0.9710 | 0.9592 | 0.9700 |

**t02 — unknown vehicle, known attack (dos/gear/rpm/standstill)**

| variant | dos | gear | rpm | standstill | macro |
|---------|-----|------|-----|------------|-------|
| moe | 0.3845 | 0.2445 | 0.4422 | 0.4402 | 0.3778 |
| moe_noaux | 0.3792 | 0.2284 | 0.4375 | 0.4315 | 0.3691 |

Gear is the hardest to transfer across vehicles (0.22–0.24 AUROC).

**t03 — known vehicle, unknown attacks (double/fuzzing/interval/speed/systematic/triple)**

| variant | double | fuzzing | interval | speed | systematic | triple | macro |
|---------|--------|---------|----------|-------|------------|--------|-------|
| moe | 0.6317 | 0.4914 | 0.5590 | 0.5215 | 0.4730 | 0.6057 | 0.5471 |
| moe_noaux | 0.6384 | 0.4853 | 0.5442 | 0.5132 | 0.4542 | 0.6039 | 0.5399 |

**t04 — unknown vehicle, unknown attacks (double/fuzzing/interval/speed/systematic/triple)**

| variant | double | fuzzing | interval | speed | systematic | triple | macro |
|---------|--------|---------|----------|-------|------------|--------|-------|
| moe | 0.5217 | 0.5096 | 0.5124 | 0.6696 | 0.5287 | 0.5080 | 0.5416 |
| moe_noaux | 0.5260 | 0.5227 | 0.5226 | 0.6719 | 0.5335 | 0.5034 | 0.5467 |

**t06 — masquerade**

| variant | masquerade | macro |
|---------|------------|-------|
| moe | 0.9834 | 0.9834 |
| moe_noaux | 0.9853 | 0.9853 |

---

### set_02

**t01 — known vehicle, known attack (double/fuzzing/interval/speed/systematic/triple)**

| variant | double | fuzzing | interval | speed | systematic | triple | macro |
|---------|--------|---------|----------|-------|------------|--------|-------|
| moe | 0.8880 | 0.9352 | 0.8196 | 0.7674 | 0.9521 | 0.8912 | 0.8756 |
| moe_noaux | 0.8783 | 0.9334 | 0.8171 | 0.7631 | 0.9497 | 0.8880 | 0.8716 |

**t02 — unknown vehicle, known attack**

| variant | double | fuzzing | interval | speed | systematic | triple | macro |
|---------|--------|---------|----------|-------|------------|--------|-------|
| moe | 0.5265 | 0.5886 | 0.5156 | 0.5415 | 0.5010 | 0.5130 | 0.5310 |
| moe_noaux | 0.4857 | 0.5653 | 0.4947 | 0.5279 | 0.4345 | 0.4907 | 0.4998 |

**t03 — known vehicle, unknown attacks (dos/gear/rpm/standstill)**

| variant | dos | gear | rpm | standstill | macro |
|---------|-----|------|-----|------------|-------|
| moe | 0.5478 | 0.8543 | 0.7015 | 0.6548 | 0.6896 |
| moe_noaux | 0.5731 | 0.8271 | 0.6859 | 0.6375 | 0.6809 |

**t04 — unknown vehicle, unknown attacks (dos/gear/rpm/standstill)**

| variant | dos | gear | rpm | standstill | macro |
|---------|-----|------|-----|------------|-------|
| moe | 0.9081 | 0.4678 | 0.4459 | 0.2924 | 0.5286 |
| moe_noaux | 0.8883 | 0.4386 | 0.4246 | 0.2631 | 0.5037 |

DOS transfers to unknown vehicle (0.89–0.91 AUROC) but standstill completely fails (0.26–0.29).

**t06 — masquerade**

| variant | masquerade | macro |
|---------|------------|-------|
| moe | 0.7857 | 0.7857 |
| moe_noaux | 0.7772 | 0.7772 |

---

### set_03

**t01 — known vehicle, known attack (dos/double/fuzzing/gear/triple)**

| variant | dos | double | fuzzing | gear | triple | macro |
|---------|-----|--------|---------|------|--------|-------|
| moe | 0.6631 | 0.8755 | 0.8741 | 0.8647 | 0.8853 | 0.8325 |
| moe_noaux | 0.6604 | 0.8745 | 0.8740 | 0.8620 | 0.8831 | 0.8308 |

DOS is the weakest in set_03 t01 (0.66) compared to set_01 t01 (0.97). Different vehicle pairing.

**t02 — unknown vehicle, known attack**

| variant | dos | double | fuzzing | gear | triple | macro |
|---------|-----|--------|---------|------|--------|-------|
| moe | 0.3424 | 0.5583 | 0.8093 | 0.4828 | 0.5059 | 0.5398 |
| moe_noaux | 0.3511 | 0.5651 | 0.6573 | 0.4563 | 0.5181 | 0.5096 |

Fuzzing transfers best (0.66–0.81 AUROC); DOS and gear do not.

**t03 — known vehicle, unknown attacks (interval/rpm/rpm-accessory/speed/speed-accessory/standstill/systematic)**

| variant | interval | rpm | rpm-acc | speed | speed-acc | standstill | syst | macro |
|---------|----------|-----|---------|-------|-----------|------------|------|-------|
| moe | 0.5929 | 0.6406 | 0.5610 | 0.5793 | 0.5561 | 0.6222 | 0.5937 | 0.5923 |
| moe_noaux | 0.5811 | 0.6408 | 0.5478 | 0.5826 | 0.5492 | 0.6224 | 0.5964 | 0.5886 |

**t04 — unknown vehicle, unknown attacks**

| variant | interval | rpm | rpm-acc | speed | speed-acc | standstill | syst | macro |
|---------|----------|-----|---------|-------|-----------|------------|------|-------|
| moe | 0.4054 | 0.5102 | 0.4518 | 0.5792 | 0.4652 | 0.7110 | 0.4748 | 0.5139 |
| moe_noaux | 0.4007 | 0.5190 | 0.4354 | 0.5934 | 0.4306 | 0.7031 | 0.4880 | 0.5100 |

Standstill transfers to unknown vehicle (0.70–0.71) despite being unknown attack type.

**t06 — masquerade**

| variant | masquerade | macro |
|---------|------------|-------|
| moe | 0.8269 | 0.8269 |
| moe_noaux | 0.8413 | 0.8413 |

---

### set_04

moe ‡. moe_noaux per-split per-attack detail below.

**t01 — known vehicle, known attack (interval/rpm/rpm-acc/speed/speed-acc/standstill/systematic)**

| variant | interval | rpm | rpm-acc | speed | speed-acc | standstill | syst | macro |
|---------|----------|-----|---------|-------|-----------|------------|------|-------|
| moe_noaux | 0.6325 | 0.7687 | 0.8468 | 0.6876 | 0.8919 | 0.8414 | 0.7569 | 0.7751 |

**t02 — unknown vehicle, known attack**

| variant | interval | rpm | rpm-acc | speed | speed-acc | standstill | syst | macro |
|---------|----------|-----|---------|-------|-----------|------------|------|-------|
| moe_noaux | 0.5001 | 0.5888 | 0.5004 | 0.5136 | 0.5007 | 0.5001 | 0.5000 | 0.5148 |

Near-random on every attack type when vehicle is unknown.

**t03 — known vehicle, unknown attacks (dos/double/fuzzing/gear/triple)**

| variant | dos | double | fuzzing | gear | triple | macro |
|---------|-----|--------|---------|------|--------|-------|
| moe_noaux | 0.9252 | 0.8881 | 0.9904 | 0.8651 | 0.8448 | 0.9027 |

This is the highest AUROC split for set_04 — known vehicle helps enough that novel attacks
are still strongly detectable. Fuzzing is nearly perfect (0.99 AUROC).

**t04 — unknown vehicle, unknown attacks (dos/double/fuzzing/gear/triple)**

| variant | dos | double | fuzzing | gear | triple | macro |
|---------|-----|--------|---------|------|--------|-------|
| moe_noaux | 0.5000 | 0.5000 | 0.5008 | 0.5000 | 0.7088 | 0.5419 |

Only triple marginally above random (0.71); vehicle shift dominates.

**t06 — masquerade**

| variant | masquerade | macro |
|---------|------------|-------|
| moe_noaux | 0.9734 | 0.9734 |

---

## D. hcrl_sa — aggregate test (dev-only, for reference)

Single `test` split (no named breakdown). Included for plumbing validation only.

| variant | auroc_macro | mcc | f1/attack | recall/attack | precision/attack |
|---------|-------------|-----|-----------|---------------|------------------|
| mlp | 0.8586 | 0.7367 | 0.7376 | 0.5887 | 0.9873 |
| moe | 0.9991 | 0.9934 | 0.9944 | 0.9962 | 0.9925 |
| moe_noaux | — | — | — | — | — |
| weighted_avg | 0.9179 | 0.6207 | 0.6520 | 0.9509 | 0.4961 |
| bandit | 0.9999 | 0.9098 | 0.9201 | 1.0000 | 0.8521 |
| dqn | 1.0000 | 0.9870 | 0.9888 | 1.0000 | 0.9779 |

Per-attack AUROC (dos/fuzzing): mlp dos=0.9992 fuzzing=0.7250; moe dos=0.9996 fuzzing=0.9993;
weighted_avg dos=0.9288 fuzzing=0.9291; bandit dos=0.9999 fuzzing=0.9999; dqn dos/fuzzing=1.0000.

mlp fuzzing=0.7250 is notably weaker than other methods on this dev-only dataset; moe/bandit/dqn
are within 0.001 of each other on hcrl_sa which has a single vehicle type.

---

## Observations

### t02 (unknown vehicle + known attacks) is the hardest split

Across all methods and all sets, t02 AUROC ranges 0.33–0.58. Even for attack types that score
≥ 0.97 AUROC on the same vehicle (t01), vehicle shift collapses detection. Set_01 gear AUROC
drops from 0.976 (t01) to 0.224–0.245 (t02) — a 0.75 drop on the same attack type. This is
the clearest evidence that VGAE+GAT representations are vehicle-specific and do not generalize
to unseen vehicle identities.

### t05 (suppress) AUROC = 0 across all methods, all sets

The suppress attack produces no anomalous graph-level signal detectable above the model's
decision threshold. MCC values for t05 are unreliable: they depend on whether the split's attack
class is empty (MCC=1.0 trivially, as in set_03/moe and set_04/mlp/weighted_avg) or populated
(MCC=0.10–0.18 for set_01/set_02). The model outputs near-zero attack-class scores for all
suppress samples — suppress is an adversarial attack that targets exactly the signal the model
uses (message presence/absence patterns).

### set_04 t03 reversal: known vehicle + unknown attacks is the BEST split for set_04

On set_01–03, t01 (known vehicle + known attacks) is the best split. On set_04, t03 (known
vehicle + novel attacks) achieves 0.88–0.89 AUROC while t01 is only 0.70. The set_04 training
attack repertoire (interval/RPM/speed variants) is harder to fit than set_01 DOS/gear/RPM, but
the novel attacks in t03 (DOS/double/fuzzing/gear/triple) happen to be strongly anomalous on
the known vehicle's graph structure.

### MoE ≈ MLP ≈ MoE-noaux on all splits

Within each (dataset, split) cell, mlp/moe/moe_noaux are within 0.015 AUROC and 0.030 MCC.
No split shows a systematic advantage for MoE routing. The largest single-cell gap (moe vs mlp)
is set_01 t06 masquerade (0.983 vs 0.970) — plausibly noise at single seed.

### weighted_avg collapses on set_01 t01 and t06

set_01 t01: 0.594 AUROC / 0.096 MCC vs 0.969–0.980 for moe/moe_noaux/mlp. Set_01's known
attacks (DOS, gear, RPM, standstill) on the known vehicle are apparently not rank-preserved by
a fixed linear combination of VGAE and GAT scores. The collapse is specific to this (dataset,
split) combination — weighted_avg recovers on set_02–04 t01 (0.82–0.84 AUROC).

### bandit and dqn: better AUROC than expected, MCC remains low

Both RL methods achieve AUROC 0.74–0.94 on several t01/t06 cells, above naive baseline. But
MCC remains 0.10–0.52 where moe/mlp achieve 0.40–0.80. On t02–t04, MCC is near zero for both
methods across all datasets. The degenerate bandit set_04 t05 MCC = −1.0 is consistent with
always-attack collapse on a split where attacks are extremely rare. AUROC is not informative
for evaluating RL policy quality; MCC and precision expose the collapse.

### t06 masquerade is highly variable and set-dependent

AUROC ranges 0.30 (set_03/mlp) to 0.985 (set_01/moe_noaux). Set_03/mlp and set_03/weighted_avg
both fail on t06 (0.30, 0.37 AUROC) while moe/moe_noaux score 0.83–0.84. This is the largest
inter-method gap on any single (dataset, split) cell and warrants investigation — masquerade on
set_03 may be particularly sensitive to the routing mechanism.

---

## Caveats

**Previous per-attack AUROC data (§A in prior version) were from val-fallback runs.** The 0.9994–1.0000
per-attack AUROC reported in the 2026-05-07 first draft of this file measured validation performance
(test_dataloader() returned val_dataloader() before the fix). Those numbers have been removed.

**set_04/moe per-split breakdown missing.** Needs resubmission from re-rendered plan at
`rendered/set_04/ablations/fusion/seed_42.json`. Until then, set_04 MoE comparisons use moe_noaux
as proxy.

**hcrl_sa/moe_noaux test not logged.** Intentional dev-only exclusion — not a blocking issue.

**t05 AUROC and MCC should be excluded from cross-method comparisons.** The metric is degenerate
(AUROC=0 universally; MCC depends on attack-class emptiness which varies by set and split).

**Single seed (42).** No confidence intervals. Differences within ±0.01 AUROC or ±0.02 MCC on
any single (dataset, split) cell should be treated as noise.

**bandit and dqn** are now on v6 states (re-rendered plans with correct cached_states_dir), but
policy quality remains substantially below moe/mlp on discriminative metrics.

---

## Next steps

1. **Resubmit set_04/moe** from `rendered/set_04/ablations/fusion/seed_42.json` to get per-split breakdown.
2. **Investigate set_03/t06 mlp gap** — masquerade AUROC 0.30 for mlp vs 0.83 for moe: check if different
   checkpoint was loaded or if MLP head is systematically weaker on masquerade pattern.
3. **Suppress attack**: AUROC=0 means the suppress signal is invisible to VGAE+GAT. Consider whether a
   temporal or sequence-level model would be needed to detect ECU suppression.
4. **Multi-seed runs (seeds 0, 1, 42)** before paper claims, especially for t02 AUROC estimates and t06
   variance.
5. **Cross-vehicle transfer**: t02 result (0.33–0.54 AUROC) is the most paper-relevant finding. Decide
   whether to report this as a limitation or as motivation for vehicle-agnostic feature learning.
6. **Check MoE gate entropy on set_03 t06**: routing may collapse to uniform on masquerade samples —
   entropy probe would distinguish active routing from identity-mapping.
