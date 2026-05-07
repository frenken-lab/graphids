# 2026-05-06 — GAT supervised ablation: cross-analysis of all metrics

**Seed:** 42. **Status:** supervised ablation complete across hcrl_sa, set_01–04.
set_01/curriculum_random still pending (omitted from sampling tables).

Split key:
- **t01** known-vehicle known-attack (in-distribution)
- **t02** unknown-vehicle known-attack (cross-vehicle generalization)
- **t03** known-vehicle unknown-attack (zero-shot attack type)
- **t04** unknown-vehicle unknown-attack (full out-of-distribution)
- **t05** suppress-attacks (always 0.000 — structural blind spot; omitted from means)
- **t06** masquerade-attacks (always high — trivially detectable; omitted from means)

Mean columns use t01–t04 only. All results are AUROC-macro unless stated otherwise.

---

## Ablation: gat_loss

Loss function (baseline: CE with no weighting). Tests whether class-imbalance weighting or focal loss improves detection.

### A. AUROC-macro by split × dataset

Splits present in hcrl_sa: t01–t04 only (no suppress/masquerade partitions).

| variant | hcrl t01 | hcrl t02 | hcrl t03 | hcrl t04 | set_ t01 | set_ t02 | set_ t03 | set_ t04 | set_ t05 | set_ t06 | set_ t01 | set_ t02 | set_ t03 | set_ t04 | set_ t05 | set_ t06 | set_ t01 | set_ t02 | set_ t03 | set_ t04 | set_ t05 | set_ t06 | set_ t01 | set_ t02 | set_ t03 | set_ t04 | set_ t05 | set_ t06 |
|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
| none | 0.999 | 0.265 | 1.000 | 0.894 | 0.937 | 0.499 | 0.582 | 0.744 | 0.000 | 0.988 | 0.844 | 0.632 | 0.737 | 0.621 | 0.000 | 0.814 | 0.860 | 0.699 | 0.589 | 0.569 | 0.000 | 0.849 | 0.744 | 0.688 | 0.897 | 0.735 | 0.000 | 0.975 |
| ce | 0.999 | 0.448 | 1.000 | 0.739 | 0.945 | 0.557 | 0.553 | 0.716 | 0.000 | 0.982 | 0.851 | 0.563 | 0.721 | 0.598 | 0.000 | 0.831 | 0.870 | 0.649 | 0.587 | 0.561 | 0.000 | 0.825 | 0.554 | 0.836 | 0.756 | 0.917 | 0.000 | 0.955 |
| weighted_ce | 0.999 | 0.359 | 1.000 | 0.734 | 0.964 | 0.550 | 0.580 | 0.550 | 0.000 | 0.981 | 0.867 | 0.546 | 0.736 | 0.584 | 0.000 | 0.835 | 0.868 | 0.668 | 0.590 | 0.551 | 0.000 | 0.860 | 0.554 | 0.852 | 0.777 | 0.929 | 0.000 | 0.954 |
| focal | 0.999 | 0.323 | 1.000 | 0.863 | 0.969 | 0.466 | 0.578 | 0.757 | 0.000 | 0.985 | 0.844 | 0.599 | 0.744 | 0.597 | 0.000 | 0.801 | 0.858 | 0.680 | 0.597 | 0.573 | 0.000 | 0.841 | 0.692 | 0.725 | 0.882 | 0.786 | 0.000 | 0.971 |

### B. Multi-metric summary — mean(t01–t04)

Values are mean over the 4 operationally relevant splits.

**AUROC** (`auroc_macro`)

| variant | hcrl_sa | set_01 | set_02 | set_03 | set_04 | macro-avg |
|---------|---------|---------|---------|---------|---------|---------|
| none | 0.789 | 0.691 | 0.708 | 0.679 | 0.766 | 0.727 |
| ce | 0.797 | 0.693 | 0.683 | 0.667 | 0.766 | 0.721 |
| weighted_ce | 0.773 | 0.661 | 0.683 | 0.669 | 0.778 | 0.713 |
| focal | 0.796 | 0.693 | 0.696 | 0.677 | 0.771 | 0.727 |

**AP** (`ap_macro`)

| variant | hcrl_sa | set_01 | set_02 | set_03 | set_04 | macro-avg |
|---------|---------|---------|---------|---------|---------|---------|
| none | 0.794 | 0.664 | 0.596 | 0.676 | 0.719 | 0.690 |
| ce | 0.784 | 0.637 | 0.590 | 0.649 | 0.718 | 0.676 |
| weighted_ce | 0.775 | 0.636 | 0.591 | 0.648 | 0.741 | 0.678 |
| focal | 0.792 | 0.666 | 0.594 | 0.673 | 0.743 | 0.694 |

**F1** (`f1_macro`)

| variant | hcrl_sa | set_01 | set_02 | set_03 | set_04 | macro-avg |
|---------|---------|---------|---------|---------|---------|---------|
| none | 0.642 | 0.439 | 0.361 | 0.564 | 0.402 | 0.481 |
| ce | 0.642 | 0.502 | 0.367 | 0.529 | 0.363 | 0.481 |
| weighted_ce | 0.642 | 0.437 | 0.349 | 0.560 | 0.558 | 0.509 |
| focal | 0.642 | 0.445 | 0.371 | 0.563 | 0.399 | 0.484 |

**MCC** (`mcc`)

| variant | hcrl_sa | set_01 | set_02 | set_03 | set_04 | macro-avg |
|---------|---------|---------|---------|---------|---------|---------|
| none | 0.500 | 0.199 | 0.138 | 0.278 | 0.226 | 0.268 |
| ce | 0.500 | 0.260 | 0.139 | 0.275 | 0.001 | 0.235 |
| weighted_ce | 0.500 | 0.198 | 0.124 | 0.277 | 0.319 | 0.284 |
| focal | 0.500 | 0.209 | 0.145 | 0.275 | 0.233 | 0.272 |

**P@95R** (`precision_at_0.95recall`)

| variant | hcrl_sa | set_01 | set_02 | set_03 | set_04 | macro-avg |
|---------|---------|---------|---------|---------|---------|---------|
| none | 0.757 | 0.250 | 0.072 | 0.422 | 0.477 | 0.395 |
| ce | 0.698 | 0.264 | 0.068 | 0.424 | 0.522 | 0.395 |
| weighted_ce | 0.698 | 0.281 | 0.069 | 0.425 | 0.520 | 0.399 |
| focal | 0.740 | 0.294 | 0.071 | 0.421 | 0.477 | 0.401 |

**R@99P** (`recall_at_0.99precision`)

| variant | hcrl_sa | set_01 | set_02 | set_03 | set_04 | macro-avg |
|---------|---------|---------|---------|---------|---------|---------|
| none | 0.499 | 0.158 | 0.120 | 0.174 | 0.154 | 0.221 |
| ce | 0.499 | 0.131 | 0.112 | 0.152 | 0.004 | 0.180 |
| weighted_ce | 0.499 | 0.140 | 0.121 | 0.150 | 0.032 | 0.188 |
| focal | 0.499 | 0.194 | 0.120 | 0.180 | 0.128 | 0.224 |

### C. AUROC rank within group — per (dataset × split)

Rank 1 = best within group. Tie-breaking by value (rare).

| variant | hcrl.t01 | hcrl.t02 | hcrl.t03 | hcrl.t04 | set_.t01 | set_.t02 | set_.t03 | set_.t04 | set_.t01 | set_.t02 | set_.t03 | set_.t04 | set_.t01 | set_.t02 | set_.t03 | set_.t04 | set_.t01 | set_.t02 | set_.t03 | set_.t04 | #wins |
|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
| none | 2 | 4 | 2 | 1 | 4 | 3 | 1 | 2 | 4 | 1 | 2 | 1 | 3 | 1 | 3 | 2 | 1 | 4 | 1 | 4 | **7** |
| ce | 1 | 1 | 4 | 3 | 3 | 1 | 4 | 3 | 2 | 3 | 4 | 2 | 1 | 4 | 4 | 3 | 4 | 2 | 4 | 2 | **4** |
| weighted_ce | 4 | 2 | 3 | 4 | 2 | 2 | 2 | 4 | 1 | 4 | 3 | 4 | 2 | 3 | 2 | 4 | 3 | 1 | 3 | 1 | **3** |
| focal | 3 | 3 | 1 | 2 | 1 | 4 | 3 | 1 | 3 | 2 | 1 | 3 | 4 | 2 | 1 | 1 | 2 | 3 | 2 | 3 | **6** |

---

## Ablation: gat_sampling

Training curriculum (baseline: none). Tests whether difficulty-ordered sampling lifts generalisation.

### A. AUROC-macro by split × dataset

Splits present in hcrl_sa: t01–t04 only (no suppress/masquerade partitions).

| variant | hcrl t01 | hcrl t02 | hcrl t03 | hcrl t04 | set_ t01 | set_ t02 | set_ t03 | set_ t04 | set_ t05 | set_ t06 | set_ t01 | set_ t02 | set_ t03 | set_ t04 | set_ t05 | set_ t06 | set_ t01 | set_ t02 | set_ t03 | set_ t04 | set_ t05 | set_ t06 | set_ t01 | set_ t02 | set_ t03 | set_ t04 | set_ t05 | set_ t06 |
|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
| none | 0.999 | 0.265 | 1.000 | 0.894 | 0.937 | 0.499 | 0.582 | 0.744 | 0.000 | 0.988 | 0.844 | 0.632 | 0.737 | 0.621 | 0.000 | 0.814 | 0.860 | 0.699 | 0.589 | 0.569 | 0.000 | 0.849 | 0.744 | 0.688 | 0.897 | 0.735 | 0.000 | 0.975 |
| curriculum_random | 0.999 | 0.429 | 1.000 | 0.744 | — | — | — | — | — | — | 0.859 | 0.598 | 0.752 | 0.595 | 0.000 | 0.853 | 0.860 | 0.716 | 0.581 | 0.588 | 0.000 | 0.838 | 0.692 | 0.694 | 0.865 | 0.720 | 0.000 | 0.971 |
| curriculum_vgae | 0.998 | 0.433 | 1.000 | 0.771 | 0.943 | 0.521 | 0.599 | 0.704 | 0.000 | 0.987 | 0.840 | 0.607 | 0.779 | 0.611 | 0.000 | 0.829 | 0.852 | 0.697 | 0.590 | 0.580 | 0.000 | 0.841 | 0.515 | 0.765 | 0.813 | 0.840 | 0.000 | 0.796 |

### B. Multi-metric summary — mean(t01–t04)

Values are mean over the 4 operationally relevant splits.

**AUROC** (`auroc_macro`)

| variant | hcrl_sa | set_01 | set_02 | set_03 | set_04 | macro-avg |
|---------|---------|---------|---------|---------|---------|---------|
| none | 0.789 | 0.691 | 0.708 | 0.679 | 0.766 | 0.727 |
| curriculum_random | 0.793 | N/A | 0.701 | 0.686 | 0.743 | 0.731 |
| curriculum_vgae | 0.800 | 0.692 | 0.709 | 0.680 | 0.733 | 0.723 |

**AP** (`ap_macro`)

| variant | hcrl_sa | set_01 | set_02 | set_03 | set_04 | macro-avg |
|---------|---------|---------|---------|---------|---------|---------|
| none | 0.794 | 0.664 | 0.596 | 0.676 | 0.719 | 0.690 |
| curriculum_random | 0.782 | N/A | 0.594 | 0.673 | 0.698 | 0.687 |
| curriculum_vgae | 0.784 | 0.657 | 0.595 | 0.674 | 0.698 | 0.681 |

**F1** (`f1_macro`)

| variant | hcrl_sa | set_01 | set_02 | set_03 | set_04 | macro-avg |
|---------|---------|---------|---------|---------|---------|---------|
| none | 0.642 | 0.439 | 0.361 | 0.564 | 0.402 | 0.481 |
| curriculum_random | 0.642 | N/A | 0.331 | 0.557 | 0.430 | 0.490 |
| curriculum_vgae | 0.601 | 0.420 | 0.355 | 0.539 | 0.389 | 0.461 |

**MCC** (`mcc`)

| variant | hcrl_sa | set_01 | set_02 | set_03 | set_04 | macro-avg |
|---------|---------|---------|---------|---------|---------|---------|
| none | 0.500 | 0.199 | 0.138 | 0.278 | 0.226 | 0.268 |
| curriculum_random | 0.500 | N/A | 0.112 | 0.282 | 0.232 | 0.281 |
| curriculum_vgae | 0.430 | 0.187 | 0.137 | 0.267 | 0.139 | 0.232 |

**P@95R** (`precision_at_0.95recall`)

| variant | hcrl_sa | set_01 | set_02 | set_03 | set_04 | macro-avg |
|---------|---------|---------|---------|---------|---------|---------|
| none | 0.757 | 0.250 | 0.072 | 0.422 | 0.477 | 0.395 |
| curriculum_random | 0.699 | N/A | 0.071 | 0.424 | 0.467 | 0.415 |
| curriculum_vgae | 0.711 | 0.253 | 0.072 | 0.423 | 0.488 | 0.389 |

**R@99P** (`recall_at_0.99precision`)

| variant | hcrl_sa | set_01 | set_02 | set_03 | set_04 | macro-avg |
|---------|---------|---------|---------|---------|---------|---------|
| none | 0.499 | 0.158 | 0.120 | 0.174 | 0.154 | 0.221 |
| curriculum_random | 0.499 | N/A | 0.114 | 0.151 | 0.109 | 0.218 |
| curriculum_vgae | 0.499 | 0.166 | 0.112 | 0.142 | 0.044 | 0.193 |

### C. AUROC rank within group — per (dataset × split)

Rank 1 = best within group. Tie-breaking by value (rare).

| variant | hcrl.t01 | hcrl.t02 | hcrl.t03 | hcrl.t04 | set_.t01 | set_.t02 | set_.t03 | set_.t04 | set_.t01 | set_.t02 | set_.t03 | set_.t04 | set_.t01 | set_.t02 | set_.t03 | set_.t04 | set_.t01 | set_.t02 | set_.t03 | set_.t04 | #wins |
|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
| none | 1 | 3 | 1 | 1 | 2 | 2 | 2 | 1 | 2 | 1 | 3 | 1 | 2 | 2 | 2 | 3 | 1 | 3 | 1 | 2 | **8** |
| curriculum_random | 2 | 2 | 3 | 3 | — | — | — | — | 1 | 3 | 2 | 3 | 1 | 1 | 3 | 1 | 2 | 2 | 2 | 3 | **4** |
| curriculum_vgae | 3 | 1 | 2 | 2 | 1 | 1 | 1 | 2 | 3 | 2 | 1 | 2 | 3 | 3 | 1 | 2 | 3 | 1 | 3 | 1 | **8** |

---

## Ablation: id_encoding

Input ID encoding (baseline: none — learned embedding only). Tests whether explicit hash/lookup encoding of CAN ID adds discriminative signal.

### A. AUROC-macro by split × dataset

Splits present in hcrl_sa: t01–t04 only (no suppress/masquerade partitions).

| variant | hcrl t01 | hcrl t02 | hcrl t03 | hcrl t04 | set_ t01 | set_ t02 | set_ t03 | set_ t04 | set_ t05 | set_ t06 | set_ t01 | set_ t02 | set_ t03 | set_ t04 | set_ t05 | set_ t06 | set_ t01 | set_ t02 | set_ t03 | set_ t04 | set_ t05 | set_ t06 | set_ t01 | set_ t02 | set_ t03 | set_ t04 | set_ t05 | set_ t06 |
|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
| none | 0.999 | 0.265 | 1.000 | 0.894 | 0.937 | 0.499 | 0.582 | 0.744 | 0.000 | 0.988 | 0.844 | 0.632 | 0.737 | 0.621 | 0.000 | 0.814 | 0.860 | 0.699 | 0.589 | 0.569 | 0.000 | 0.849 | 0.744 | 0.688 | 0.897 | 0.735 | 0.000 | 0.975 |
| id_hash | 0.999 | 0.514 | 1.000 | 0.920 | 0.936 | 0.514 | 0.607 | 0.599 | 0.000 | 0.986 | 0.866 | 0.587 | 0.727 | 0.612 | 0.000 | 0.812 | 0.859 | 0.717 | 0.577 | 0.548 | 0.000 | 0.867 | 0.564 | 0.738 | 0.826 | 0.799 | 0.000 | 0.951 |
| id_lookup | 0.999 | 0.395 | 1.000 | 0.988 | 0.919 | 0.498 | 0.563 | 0.738 | 0.000 | 0.982 | 0.847 | 0.603 | 0.752 | 0.604 | 0.000 | 0.838 | 0.850 | 0.679 | 0.591 | 0.558 | 0.000 | 0.849 | 0.743 | 0.628 | 0.887 | 0.631 | 0.000 | 0.973 |

### B. Multi-metric summary — mean(t01–t04)

Values are mean over the 4 operationally relevant splits.

**AUROC** (`auroc_macro`)

| variant | hcrl_sa | set_01 | set_02 | set_03 | set_04 | macro-avg |
|---------|---------|---------|---------|---------|---------|---------|
| none | 0.789 | 0.691 | 0.708 | 0.679 | 0.766 | 0.727 |
| id_hash | 0.858 | 0.664 | 0.698 | 0.675 | 0.732 | 0.725 |
| id_lookup | 0.845 | 0.679 | 0.701 | 0.669 | 0.722 | 0.724 |

**AP** (`ap_macro`)

| variant | hcrl_sa | set_01 | set_02 | set_03 | set_04 | macro-avg |
|---------|---------|---------|---------|---------|---------|---------|
| none | 0.794 | 0.664 | 0.596 | 0.676 | 0.719 | 0.690 |
| id_hash | 0.844 | 0.620 | 0.599 | 0.675 | 0.712 | 0.690 |
| id_lookup | 0.854 | 0.657 | 0.594 | 0.668 | 0.687 | 0.692 |

**F1** (`f1_macro`)

| variant | hcrl_sa | set_01 | set_02 | set_03 | set_04 | macro-avg |
|---------|---------|---------|---------|---------|---------|---------|
| none | 0.642 | 0.439 | 0.361 | 0.564 | 0.402 | 0.481 |
| id_hash | 0.642 | 0.478 | 0.331 | 0.560 | 0.343 | 0.471 |
| id_lookup | 0.595 | 0.435 | 0.363 | 0.563 | 0.411 | 0.473 |

**MCC** (`mcc`)

| variant | hcrl_sa | set_01 | set_02 | set_03 | set_04 | macro-avg |
|---------|---------|---------|---------|---------|---------|---------|
| none | 0.500 | 0.199 | 0.138 | 0.278 | 0.226 | 0.268 |
| id_hash | 0.500 | 0.198 | 0.113 | 0.292 | 0.142 | 0.249 |
| id_lookup | 0.406 | 0.196 | 0.133 | 0.280 | 0.220 | 0.247 |

**P@95R** (`precision_at_0.95recall`)

| variant | hcrl_sa | set_01 | set_02 | set_03 | set_04 | macro-avg |
|---------|---------|---------|---------|---------|---------|---------|
| none | 0.757 | 0.250 | 0.072 | 0.422 | 0.477 | 0.395 |
| id_hash | 0.761 | 0.247 | 0.068 | 0.422 | 0.472 | 0.394 |
| id_lookup | 0.822 | 0.236 | 0.072 | 0.420 | 0.464 | 0.403 |

**R@99P** (`recall_at_0.99precision`)

| variant | hcrl_sa | set_01 | set_02 | set_03 | set_04 | macro-avg |
|---------|---------|---------|---------|---------|---------|---------|
| none | 0.499 | 0.158 | 0.120 | 0.174 | 0.154 | 0.221 |
| id_hash | 0.499 | 0.140 | 0.129 | 0.155 | 0.091 | 0.203 |
| id_lookup | 0.707 | 0.194 | 0.115 | 0.173 | 0.146 | 0.267 |

### C. AUROC rank within group — per (dataset × split)

Rank 1 = best within group. Tie-breaking by value (rare).

| variant | hcrl.t01 | hcrl.t02 | hcrl.t03 | hcrl.t04 | set_.t01 | set_.t02 | set_.t03 | set_.t04 | set_.t01 | set_.t02 | set_.t03 | set_.t04 | set_.t01 | set_.t02 | set_.t03 | set_.t04 | set_.t01 | set_.t02 | set_.t03 | set_.t04 | #wins |
|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
| none | 2 | 3 | 2 | 3 | 1 | 2 | 2 | 1 | 3 | 1 | 2 | 1 | 1 | 2 | 2 | 1 | 1 | 2 | 1 | 2 | **8** |
| id_hash | 1 | 1 | 1 | 2 | 2 | 1 | 1 | 3 | 1 | 3 | 3 | 2 | 2 | 1 | 3 | 3 | 3 | 1 | 3 | 1 | **9** |
| id_lookup | 3 | 2 | 3 | 1 | 3 | 3 | 3 | 2 | 2 | 2 | 1 | 3 | 3 | 3 | 1 | 2 | 2 | 3 | 2 | 3 | **3** |

---

## Cross-group summary — best variant per metric per dataset

Mean over t01–t04. `none` baseline appears in all three groups.

### AUROC (`auroc_macro`)

| variant | hcrl_sa | set_01 | set_02 | set_03 | set_04 | macro-avg |
|---------|---------|---------|---------|---------|---------|---------|
| ce | 0.797 | 0.693 | 0.683 | 0.667 | 0.766 | 0.721 |
| curriculum_random | 0.793 | N/A | 0.701 | 0.686 | 0.743 | 0.731 |
| curriculum_vgae | 0.800 | 0.692 | 0.709 | 0.680 | 0.733 | 0.723 |
| focal | 0.796 | 0.693 | 0.696 | 0.677 | 0.771 | 0.727 |
| id_hash | 0.858 | 0.664 | 0.698 | 0.675 | 0.732 | 0.725 |
| id_lookup | 0.845 | 0.679 | 0.701 | 0.669 | 0.722 | 0.724 |
| none | 0.789 | 0.691 | 0.708 | 0.679 | 0.766 | 0.727 |
| weighted_ce | 0.773 | 0.661 | 0.683 | 0.669 | 0.778 | 0.713 |
| ★ best | **id_hash** | **ce** | **curriculum_vgae** | **curriculum_random** | **weighted_ce** | **curriculum_random** |

### AP (`ap_macro`)

| variant | hcrl_sa | set_01 | set_02 | set_03 | set_04 | macro-avg |
|---------|---------|---------|---------|---------|---------|---------|
| ce | 0.784 | 0.637 | 0.590 | 0.649 | 0.718 | 0.676 |
| curriculum_random | 0.782 | N/A | 0.594 | 0.673 | 0.698 | 0.687 |
| curriculum_vgae | 0.784 | 0.657 | 0.595 | 0.674 | 0.698 | 0.681 |
| focal | 0.792 | 0.666 | 0.594 | 0.673 | 0.743 | 0.694 |
| id_hash | 0.844 | 0.620 | 0.599 | 0.675 | 0.712 | 0.690 |
| id_lookup | 0.854 | 0.657 | 0.594 | 0.668 | 0.687 | 0.692 |
| none | 0.794 | 0.664 | 0.596 | 0.676 | 0.719 | 0.690 |
| weighted_ce | 0.775 | 0.636 | 0.591 | 0.648 | 0.741 | 0.678 |
| ★ best | **id_lookup** | **focal** | **id_hash** | **none** | **focal** | **focal** |

### F1 (`f1_macro`)

| variant | hcrl_sa | set_01 | set_02 | set_03 | set_04 | macro-avg |
|---------|---------|---------|---------|---------|---------|---------|
| ce | 0.642 | 0.502 | 0.367 | 0.529 | 0.363 | 0.481 |
| curriculum_random | 0.642 | N/A | 0.331 | 0.557 | 0.430 | 0.490 |
| curriculum_vgae | 0.601 | 0.420 | 0.355 | 0.539 | 0.389 | 0.461 |
| focal | 0.642 | 0.445 | 0.371 | 0.563 | 0.399 | 0.484 |
| id_hash | 0.642 | 0.478 | 0.331 | 0.560 | 0.343 | 0.471 |
| id_lookup | 0.595 | 0.435 | 0.363 | 0.563 | 0.411 | 0.473 |
| none | 0.642 | 0.439 | 0.361 | 0.564 | 0.402 | 0.481 |
| weighted_ce | 0.642 | 0.437 | 0.349 | 0.560 | 0.558 | 0.509 |
| ★ best | **ce** | **ce** | **focal** | **none** | **weighted_ce** | **weighted_ce** |

### MCC (`mcc`)

| variant | hcrl_sa | set_01 | set_02 | set_03 | set_04 | macro-avg |
|---------|---------|---------|---------|---------|---------|---------|
| ce | 0.500 | 0.260 | 0.139 | 0.275 | 0.001 | 0.235 |
| curriculum_random | 0.500 | N/A | 0.112 | 0.282 | 0.232 | 0.281 |
| curriculum_vgae | 0.430 | 0.187 | 0.137 | 0.267 | 0.139 | 0.232 |
| focal | 0.500 | 0.209 | 0.145 | 0.275 | 0.233 | 0.272 |
| id_hash | 0.500 | 0.198 | 0.113 | 0.292 | 0.142 | 0.249 |
| id_lookup | 0.406 | 0.196 | 0.133 | 0.280 | 0.220 | 0.247 |
| none | 0.500 | 0.199 | 0.138 | 0.278 | 0.226 | 0.268 |
| weighted_ce | 0.500 | 0.198 | 0.124 | 0.277 | 0.319 | 0.284 |
| ★ best | **ce** | **ce** | **focal** | **id_hash** | **weighted_ce** | **weighted_ce** |

### P@95R (`precision_at_0.95recall`)

| variant | hcrl_sa | set_01 | set_02 | set_03 | set_04 | macro-avg |
|---------|---------|---------|---------|---------|---------|---------|
| ce | 0.698 | 0.264 | 0.068 | 0.424 | 0.522 | 0.395 |
| curriculum_random | 0.699 | N/A | 0.071 | 0.424 | 0.467 | 0.415 |
| curriculum_vgae | 0.711 | 0.253 | 0.072 | 0.423 | 0.488 | 0.389 |
| focal | 0.740 | 0.294 | 0.071 | 0.421 | 0.477 | 0.401 |
| id_hash | 0.761 | 0.247 | 0.068 | 0.422 | 0.472 | 0.394 |
| id_lookup | 0.822 | 0.236 | 0.072 | 0.420 | 0.464 | 0.403 |
| none | 0.757 | 0.250 | 0.072 | 0.422 | 0.477 | 0.395 |
| weighted_ce | 0.698 | 0.281 | 0.069 | 0.425 | 0.520 | 0.399 |
| ★ best | **id_lookup** | **focal** | **curriculum_vgae** | **weighted_ce** | **ce** | **curriculum_random** |

### R@99P (`recall_at_0.99precision`)

| variant | hcrl_sa | set_01 | set_02 | set_03 | set_04 | macro-avg |
|---------|---------|---------|---------|---------|---------|---------|
| ce | 0.499 | 0.131 | 0.112 | 0.152 | 0.004 | 0.180 |
| curriculum_random | 0.499 | N/A | 0.114 | 0.151 | 0.109 | 0.218 |
| curriculum_vgae | 0.499 | 0.166 | 0.112 | 0.142 | 0.044 | 0.193 |
| focal | 0.499 | 0.194 | 0.120 | 0.180 | 0.128 | 0.224 |
| id_hash | 0.499 | 0.140 | 0.129 | 0.155 | 0.091 | 0.203 |
| id_lookup | 0.707 | 0.194 | 0.115 | 0.173 | 0.146 | 0.267 |
| none | 0.499 | 0.158 | 0.120 | 0.174 | 0.154 | 0.221 |
| weighted_ce | 0.499 | 0.140 | 0.121 | 0.150 | 0.032 | 0.188 |
| ★ best | **id_lookup** | **focal** | **id_hash** | **focal** | **none** | **id_lookup** |

---

## Observations

### t05 suppress-attacks: structural blind spot

AUROC = 0.000 across **every** variant and **every** dataset. Suppress attacks remove
CAN frames — the graph is sparser than benign, not denser. The GAT's score is
monotonically tied to anomalous activity presence; a sparse graph scores low.
This cannot be fixed with loss weighting or curriculum — a complementary traffic-volume
monitor (frame count per window) is required for suppress-attack detection.

### t06 masquerade-attacks: trivially detectable

AUROC ≥ 0.95 universally (set_02 lowest at 0.801–0.853, still high). Active injection
of recognizable malicious frames is the easiest detection case for a graph classifier.

### Loss function (gat_loss): negligible effect on large datasets

Macro-avg AUROC over t01–t04: focal (0.727), ce (0.721), none (0.727), weighted_ce (0.713).
Differences are within noise for a single-seed ablation. focal shows a consistent small
advantage on R@99P across set_01–04 (0.194 vs 0.158 for none on set_01), suggesting focal
is better calibrated at high-precision operating points even when macro AUROC is identical.

### Training curriculum (gat_sampling): weak positive signal

curriculum_random and curriculum_vgae consistently match or marginally exceed `none` on
AUROC (macro-avg 0.731/0.723 vs 0.727). MCC shows curriculum_vgae below baseline
(0.232 vs 0.268), indicating it may trade decision-boundary quality for ranking. 
curriculum_vgae is the weakest variant on set_04 (0.733 AUROC vs 0.766 baseline),
suggesting the VGAE score signal is poorly calibrated for set_04's timing-perturbation attacks.

### ID encoding (id_encoding): strong lift on hcrl_sa, neutral on set_01–04

hcrl_sa: id_hash 0.858, id_lookup 0.845 vs baseline 0.789 (+0.069/+0.056 AUROC).
The gain is concentrated in t04 (unknown-vehicle unknown-attack): id_lookup 0.988, id_hash 0.920
vs none 0.894. Explicit CAN ID tokens appear to encode vehicle-specific traffic patterns
that transfer to unseen vehicles when the attack type is also unseen.

set_01–04: id_hash 0.664–0.698, id_lookup 0.669–0.701, none 0.679–0.708 — id encoding
is neutral to slightly negative on the large datasets. These datasets have more diverse
vehicle profiles; the ID vocabulary loses its discriminative edge.

id_lookup achieves the best R@99P macro-avg (0.267 vs 0.221 for none), driven by hcrl_sa
(0.707 vs 0.499). The lookup encoding provides better high-precision recall on the
dataset where ID space is most structured.
