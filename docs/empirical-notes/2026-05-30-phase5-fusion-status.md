# Phase 5 fusion status — hcrl_sa seed 42

Context: resumed after the May 9 representation-first refactor. The current
shell did not export `GRAPHIDS_LAKE_ROOT`, so this audit queried the known
lake-root MLflow store directly:

```python
mlflow.set_tracking_uri("sqlite:////fs/ess/PAS1266/graphids/mlflow.db")
experiment = "graphids/hcrl_sa/fusion"
```

## Latest completed test runs

Latest by MLflow start time for each `graphids.variant` in
`graphids/hcrl_sa/fusion`.

| Variant | Run ID | MCC | AUROC | fuzzing AUROC | DOS AUROC | Attack precision | Attack recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| mlp | `c1be46063eb74a54954de70d26e2d34f` | 0.737 | 0.859 | 0.725 | 0.999 | 0.987 | 0.589 |
| moe | `3ff996809e9e4a77b883b8449a4605d2` | **0.993** | 0.999 | 0.999 | 1.000 | 0.992 | 0.996 |
| weighted_avg | `90df5aedf5f7444788806c2c52349faf` | 0.621 | 0.918 | 0.929 | 0.929 | 0.496 | 0.951 |
| bandit | `890a674a8471468e9e731173a2a19b58` | 0.910 | 1.000 | 1.000 | 1.000 | 0.852 | 1.000 |
| dqn | `33033c9617b44db7bf573eb68c2e6a78` | 0.987 | **1.000** | **1.000** | **1.000** | 0.978 | 1.000 |

## Interpretation

Phase 5 acceptance is met on hcrl_sa: MoE beats MLP on both stated gates.

- MCC: `0.993` vs `0.737`
- fuzzing AUROC: `0.999` vs `0.725`

DQN is now competitive after the later rerun: AUROC is effectively perfect
and MCC is `0.987`, below MoE but far above the original collapse runs
(`MCC≈0.04`). That means the old decision tree should not assume RL is still
collapsed on hcrl_sa.

MoE diagnostics indicate a router-collapse caveat. Latest MoE fit run
`fbb9499aaa2844e795ce0bd8792338d2` logged:

- `val/gate_entropy = 0.0000013`
- `val/expert_usage_0 ≈ 0`
- `val/expert_usage_1 = 0.0059`
- `val/expert_usage_2 = 0.9941`

So MoE wins the metric gate, but not by demonstrating healthy per-sample
routing. It behaves like a single selected expert. Treat this as "supervised
nonlinear fusion works" rather than proof that mixture routing is carrying
the improvement.

## Next decision

Do not spend effort on Phase 3 rich feature extraction for hcrl_sa first:
the hcrl_sa fuzzing gate is already closed by MoE/DQN on the existing v6
fusion state.

Next experiment should be cross-dataset validation:

1. Query/latest-run audit for `set_01..set_04` fusion variants.
2. If MoE/DQN already close fuzzing/MCC there, update the paper narrative
   around supervised nonlinear fusion and calibrated DQN.
3. If set_01/04 remain weak, then Phase 3 rich features are justified on
   those datasets, not because of hcrl_sa.

## Cross-dataset audit

Queried latest completed test runs for `set_01..set_04` in the same MLflow
store after the hcrl_sa audit.

| Dataset | Best latest variant by MCC | Best MCC | Best AUROC | Notes |
|---|---|---:|---:|---|
| set_01 | dqn | 0.274 | 0.661 | All methods weak; MLP close at MCC 0.274 / AUROC 0.623. |
| set_02 | dqn | 0.175 | 0.678 | All methods weak; bandit essentially tied. |
| set_03 | moe | 0.365 | 0.736 | MoE best but still moderate. |
| set_04 | moe | **0.763** | **0.960** | MoE is the only strong run; fuzzing AUROC 0.940, attack macro 0.948. |

Latest-run details:

| Dataset | Variant | Run ID | MCC | AUROC | fuzzing AUROC |
|---|---|---:|---:|---:|---:|
| set_01 | mlp | `2e328604584440cc916714ab307abce2` | 0.274 | 0.623 | n/a |
| set_01 | moe | `9ff2aa71551341d984922ce09965f977` | 0.140 | 0.646 | n/a |
| set_01 | dqn | `9e80108cc9334a138aa59234109adb53` | **0.274** | **0.661** | n/a |
| set_02 | mlp | `53059e30df0c47b7b7507e991b579913` | 0.143 | **0.689** | n/a |
| set_02 | moe | `4d8b6bf6c13c469dbace544aef02df0e` | 0.143 | 0.680 | n/a |
| set_02 | dqn | `e1a62e148cbc4f5d8e3098fc3471dbbf` | **0.175** | 0.678 | n/a |
| set_03 | mlp | `4728537f3fda4d97983f1bad51604c95` | 0.361 | 0.643 | n/a |
| set_03 | moe | `a0e1b7d9c4384df8afb32ca214264b82` | **0.365** | **0.736** | n/a |
| set_03 | dqn | `f362cfa9ed1c4b28aa60c42e48f5c361` | 0.330 | 0.735 | n/a |
| set_04 | mlp | `f0f904741bf6479d83704c86dfc065d0` | -0.137 | 0.507 | n/a |
| set_04 | moe | `6e0223418c114c54b2e6c934250865de` | **0.763** | **0.960** | 0.940 |
| set_04 | dqn | `bb37e2300e5d41ddaa1c7c1eb10659a2` | 0.133 | 0.455 | n/a |

Conclusion: Phase 5 is validated on hcrl_sa and set_04, but not enough for
set_01/set_02/set_03. Phase 3 rich fusion-state features are still justified,
but the target datasets are set_01/set_02/set_03. Also, many set_01..set_03
test runs do not expose per-attack AUROC; before interpreting attack-family
failure modes, rerun or re-evaluate those tests through the post-attack-type
metric path.
