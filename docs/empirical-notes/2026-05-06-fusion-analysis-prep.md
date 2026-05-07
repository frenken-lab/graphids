# 2026-05-06 — Fusion ablation: analysis prep and extracted-state reference

Jobs submitted: plan_ids `019e0028-a390` through `019e0028-aac2` (hcrl_sa, set_01–04,
seed 42). Four fusion methods per dataset: bandit, dqn, mlp, weighted_avg.

---

## What are "extracted states"?

The fusion pipeline does not run VGAE and GAT live during training. Instead, an
**extract** job runs both models in inference mode over the full dataset once, collects
per-graph feature tensors, and writes them to a TensorDict cache at
`{states_dir(dataset, seed)}/fusion_states/{train,val}_states.pt`.

Every fusion model then trains on this frozen cache — it never sees raw graphs.
The cached tensor is called the **extracted state** (or fusion state).

---

## Feature layout — 18-dimensional flat vector

`flatten_features(td)` sorts nested TensorDict keys lexicographically and concatenates
along the last dim. The resulting flat vector is always in this exact order:

| position | key | shape | source | meaning |
|----------|-----|-------|--------|---------|
| 0 | `gat/conf` | [N,1] | GAT | normalized confidence: `1 − entropy/log(2)`. 0 = max-entropy (GAT has no idea), 1 = certain |
| 1–4 | `gat/emb_stats` | [N,4] | GAT | final graph-embedding statistics: mean, std, max, min across the embedding dim |
| 5–6 | `gat/probs` | [N,2] | GAT | softmax output: `[prob_benign, prob_attack]`. `probs[:,1]` is the GAT attack score used in RL reward |
| 7 | `vgae/affinity` | [N,1] | VGAE | TAM per-graph mean affinity — measures how tightly a graph's nodes cluster in latent space relative to the benign manifold |
| 8 | `vgae/conf` | [N,1] | VGAE | `1/(1+recon_mean)` — high when reconstruction is low (graph looks benign) |
| 9 | `vgae/errors[0]` | [N,1] | VGAE | `recon` — mean per-node reconstruction MSE across the graph |
| 10 | `vgae/errors[1]` | [N,1] | VGAE | `mahal` — Mahalanobis distance of the graph's latent z from the prior |
| 11 | `vgae/errors[2]` | [N,1] | VGAE | `kl` — KL divergence per graph |
| 12 | `vgae/rq` | [N,1] | VGAE | Rayleigh quotient — input-space spectral smoothness; low = graph signal varies sharply across edges (anomalous topology) |
| 13 | `vgae/spike` | [N,1] | VGAE | `recon_max` — per-graph **maximum** masked-node MSE. Captures spike-pattern attacks where 1–N malicious frames in a 100-frame window have very high node-level error even when the mean is low |
| 14–17 | `vgae/z_stats` | [N,4] | VGAE | latent z statistics per graph: mean, std, max, min across the latent dim |

Total: 1+4+2+1+1+3+1+1+4 = **18 dims** (`_state_dim = 18` in `fusion.py`).

Note: `vgae/errors` is stored as a single [N,3] tensor in the TensorDict; `flatten_features`
concatenates it as positions 9–11 in the order [recon, mahal, kl].

---

## What each model does with these states

### MLP (`mlp`)
Fully connected net over the flat 18-dim vector → binary classification. Standard
`BCEWithLogitsLoss`. The direct supervised baseline: maximum AUROC potential from the
given features. If bandit/DQN can't beat MLP on AUROC, the RL reward isn't aligning
with the ranking objective.

### WeightedAvg (`weighted_avg`)
Learns a single scalar `w = sigmoid(θ)` and combines:
`score = w · prob_attack_gat + (1−w) · vgae_score`
where `vgae_score = dot(errors, [0.4, 0.3, 0.3])` (recon weighted most).
`θ` initializes at 0 → `w = 0.5` (equal weight at start).
**Interpretable**: the trained `w` directly tells you how much the fusion model
trusts GAT vs VGAE on each dataset. Watch this value in MLflow logs.

### Bandit (`bandit`)
LinUCB contextual bandit. Maintains a ridge-regularized precision matrix `A_inv`
per arm (benign/attack) and selects actions via UCB bonus `α·√(x^T A_inv x)`.
Sherman-Morrison rank-1 updates — no gradient descent. Trains in a single online
pass; `max_epochs=1500` means 1500 full passes through the shuffled training buffer.
Does **not** directly optimize AUROC — optimizes cumulative reward.

### DQN (`dqn`)
Deep Q-network (torchrl). State = 18-dim flat features; actions = {benign=0, attack=1}.
Trains from a replay buffer via TD(0) target updates. Same reward function as bandit.
Highest capacity RL method but also most sensitive to reward shaping.

---

## RL reward structure (bandit and DQN)

Defined in `REWARD` (primitives.py), applied via `reward.derive_scores()`:

```
vgae_score  = dot(errors, [0.4, 0.3, 0.3])   # recon:mahal:kl weighted
gat_score   = probs[:, 1]                      # attack probability

reward = correct_bonus                          # +3.0 if action matches label
       + incorrect_penalty                      # −3.0 if wrong
       + confidence_weight * combined_conf      # +0.5 × mean(gat_conf, vgae_conf)
       + combined_conf_weight * agreement       # +0.3 when both models agree
       + disagreement_penalty (if disagree)     # −1.0 when models conflict
       + overconf_penalty (if wrong+confident)  # −1.5 when confidently wrong
       + balance_weight * class_balance         # +0.3 for balanced episode sampling
```

The disagreement penalty is the key design decision: the RL agent is explicitly
rewarded for not committing when VGAE and GAT disagree, and penalized for
confident wrong predictions more than unconfident wrong predictions.

---

## What to look for in results

**Per metric:**
- **AUROC**: MLP should be highest or tied — it's directly trained on a ranking-compatible
  loss. If bandit/DQN approach MLP, the reward is well-aligned.
- **MCC / F1**: RL methods may show better decision-boundary calibration than MLP if the
  reward's overconfidence penalty improves threshold behaviour.
- **P@95R / R@99P**: Most operationally relevant. WeightedAvg may punch above its weight
  here — a well-tuned `w` that defers to the higher-precision model at high-precision
  thresholds would show up in R@99P.

**Per method:**
- `weighted_avg.w` in MLflow: values near 0 = trust VGAE, near 1 = trust GAT.
  Expected: GAT dominates on hcrl_sa (id-rich); VGAE may contribute more on set_04
  (timing attacks where GAT had calibration issues).
- If bandit underperforms weighted_avg significantly: the contextual policy hasn't
  outperformed the simple linear blend, suggesting the 18-dim state isn't adding
  much over the two scalar scores.
- DQN worse than bandit: usually means the replay buffer is too small or the
  TD targets are noisy — not a signal about the features.

**Structural caveat:**
t05 (suppress attacks) will be 0.000 on every fusion method. Neither VGAE nor GAT
produces a useful score for suppress attacks (both score sparse graphs as benign).
The fusion model receives two near-zero inputs for suppress samples and cannot recover.
This is expected, not a regression.
