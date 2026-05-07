# Fusion Model Design, Policy Design & Reward Shaping for CAN Bus Anomaly Detection

> Context: GAT (supervised, discriminative) + VGAE (unsupervised, generative) feature fusion
> for graph-level anomaly detection. 18-dim state: `gat/{conf, emb_stats[4], probs[2]}` +
> `vgae/{affinity, conf, errors[3], rq, spike, z_stats[4]}`. Methods: MLP, WeightedAvg
> (scalar α), LinUCB, DQN. Findings: α→1.0 universally; AUROC≈1, MCC≈0; reward contains
> anti-optimal shaping terms.

## 1. Fusion Taxonomy and Why α→1.0 Is Not What It Seems

### 1.1 The Three-Level Fusion Hierarchy

Decision fusion (Ross & Jain, 2003; Kittler et al., 1998) partitions multi-source
combination into three architecturally incomparable levels:

| Level | Architecture | Learns | What α→1.0 falsifies |
|---|---|---|---|
| **Feature fusion** | MLP over raw 18-dim vector | Joint representation | Nothing — MLP never computed α |
| **Score fusion** | WeightedAvg scalar blend | Linear combination of two pre-designated scalars | Only that `vgae_anom` adds nothing *to this blend* |
| **Decision fusion** | Bandit / DQN over 18-dim state | Policy over {benign, attack} | Nothing about feature utility |

α→1.0 applies only to **score fusion**. Collapses to `score = gat/probs[:,1]` — 16 of 18
features inaccessible to `WeightedAvg`; `vgae/rq`, `vgae/spike`, `vgae/affinity`,
`vgae/z_stats` never entered the blend.

### 1.2 Why Supervised Subsumes Unsupervised Scalars (Theory)

By no-free-lunch on anomaly detection (Emmott et al., 2015), a labeled classifier subsumes
an unsupervised proxy. `gat/probs[:,1]` is the Bayes-optimal decision boundary;
`vgae_anom = recon_mean/(1+recon_mean)` is monotone in reconstruction error, no label
signal. On set_02/03 (AUROC>0.99) GAT is near-optimal. On set_01/04 (recall 0.46–0.51) GAT
fails systematically, but scalar `vgae_anom` fails in the same region (both graph-level
aggregates suppress spatial structure). Orthogonal VGAE signal lives in `rq` (spectral),
`spike` (max masked-node MSE), `z_stats` (latent drift) — none exposed in two-scalar blend
or reward path.

**Implication:** keep VGAE in the feature vector; remove agreement term from reward;
expose `rq`/`spike`/`affinity` through richer fusion.

**Sources:** Ross & Jain (2003), PRL 24(13). Kittler et al. (1998), IEEE TPAMI 20(3).
Emmott et al. (2015), KDD workshop.

## 2. Fusion Architecture Design

### 2.1 Failure of Static Score Fusion

Kittler et al. (1998): sum rule beats product/min/max when components are (a) calibrated,
(b) independent, (c) near chance. All three fail here: not in shared probability space,
correlated on benign (both ≈0), GAT alone hits AUROC>0.99. When component reliability
varies by sample, use **gated/attention-weighted fusion** (Hazarika et al., 2020) — weights
as functions of input context, not a global scalar.

### 2.2 Attention-Gated Fusion (Recommended Replacement for WeightedAvg)

```python
class AttentionFusion(nn.Module):
    """Per-sample gating over GAT and VGAE feature blocks."""
    def __init__(self, gat_dim=6, vgae_dim=12, hidden=32):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(gat_dim + vgae_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 2), nn.Softmax(dim=-1))   # [w_gat, w_vgae]
        self.gat_proj  = nn.Linear(gat_dim, hidden)
        self.vgae_proj = nn.Linear(vgae_dim, hidden)
        self.head      = nn.Linear(hidden, 1)

    def forward(self, x):
        gat_x, vgae_x = x[..., :6], x[..., 6:]
        weights = self.gate(x)                           # [B, 2]
        fused = weights[..., 0:1] * self.gat_proj(gat_x) \
              + weights[..., 1:2] * self.vgae_proj(vgae_x)
        return self.head(fused).squeeze(-1), weights
```

Testable hypothesis: suppress (sparse topology, high `rq` gap) → elevated `w_vgae`;
injection → `w_gat≈1`.

**Sources:** Hazarika et al. (2020), ACM MM 2020. Vaswani et al. (2017), NeurIPS 2017.

### 2.3 Multi-View Learning Frame

GAT/VGAE as two views (Xu et al., 2013). Co-training (Blum & Mitchell, 1998): views
maximally complementary when conditionally independent given label. GAT/VGAE share topology
(not CI), but error modes may be: GAT fails on low-feature-density attacks, VGAE fails on
high-variance benign topology. CI is sufficient, not necessary.

**Design principle:** evaluate per-subtype AUROC (inject, suppress, fuzzy, timing), not
aggregate. Aggregate on set_01/04 conflates subtypes — can't tell if VGAE provides
orthogonal signal on suppress.

**Sources:** Xu et al. (2013), AI Review 42(2). Blum & Mitchell (1998), COLT 1998.

## 3. RL Policy Design

### 3.1 Why Discrete DQN Is the Wrong Action Space

DQN with actions={benign=0, attack=1} is a classifier trained via RL. Q-values learn
discounted reward, not calibrated probabilities. `arg max Q(s,·) ≡ Q(s,1)>Q(s,0)` — no
reason Q is calibrated such that `Q(s,1)−Q(s,0)=0` is optimal, especially under biased
reward. AUROC≈1.0, MCC≈0 = well-ranked but uncalibrated; `balance` term forces blend off
the data-optimum, displacing the threshold without the policy able to compensate.

**Short-term fix (no retraining):** Platt scaling on val — fit logistic regression on
`Q(s,1) − Q(s,0)` against labels (Platt, 1999).

**Sources:** Platt (1999), Adv. Large Margin Classifiers 10(3). Niculescu-Mizil & Caruana
(2005), ICML 2005.

### 3.2 Continuous Action Space: PPO with Continuous α

Policy outputs scalar `α ∈ [0,1]`; `score(s, α) = α · gat_attack(s) + (1−α) · vgae_anom(s)`.
Policy learns when to trust GAT vs VGAE per-sample.

**PPO** (Schulman et al., 2017): state=18-dim frozen cache; action=α via sigmoid;
reward=ranking or asymmetric FN/FP (§4); objective
`L^CLIP(θ) = E[min(r_t A_t, clip(r_t,1±ε) A_t)]`. Over vanilla PG: trust-region clipping
stabilizes under biased dense reward; entropy bonus (`L = L^CLIP − c₁L^VF + c₂S[π_θ]`)
discourages premature α→0/1 collapse, replacing the broken `balance` term; PG over
rollouts handles class imbalance better than DQN's TD errors.

```python
class AlphaPolicy(nn.Module):
    """PPO actor-critic for continuous fusion weight."""
    def __init__(self, state_dim=18, hidden=64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),   nn.Tanh())
        self.alpha_mean   = nn.Linear(hidden, 1)
        self.alpha_logvar = nn.Linear(hidden, 1)
        self.value        = nn.Linear(hidden, 1)

    def forward(self, s):
        h = self.shared(s)
        a = F.softplus(self.alpha_mean(h)) + 1.0
        b = F.softplus(self.alpha_logvar(h)) + 1.0
        return torch.distributions.Beta(a, b), self.value(h)
```

**Beta over clipped Gaussian:** bounded support, correct boundary asymptotics (deterministic
α≈1 without clipping), interpretable mode `(α−1)/(α+β−2)`. GAE-λ=0.95 averages ~20 future
steps — better than TD(0) for temporal patterns (fuzzy attacks over multiple graphs).

**Sources:** Schulman et al. (2017), arXiv:1707.06347. Schulman et al. (2015) GAE, ICLR
2016. Chou et al. (2017), ICML 2017.

### 3.3 SAC with Continuous α — When to Prefer It Over PPO

**SAC** (Haarnoja et al., 2018) maximizes `J(π) = Σ E[r_t + α_ent H(π(·|s_t))]`.
(1) Entropy maximization prevents α collapse — principled alternative to broken `balance`.
(2) Off-policy from frozen replay buffer: fusion uses frozen `train_states.pt`; SAC learns
from it directly, PPO would need re-rollouts.

```python
class SACAlphaActor(nn.Module):
    def __init__(self, state_dim=18, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),   nn.ReLU())
        self.c1 = nn.Linear(hidden, 1)  # Beta concentration1
        self.c0 = nn.Linear(hidden, 1)  # Beta concentration0

    def sample(self, s):
        h = self.net(s)
        c1 = F.softplus(self.c1(h)) + 1.0
        c0 = F.softplus(self.c0(h)) + 1.0
        dist = torch.distributions.Beta(c1, c0)
        alpha = dist.rsample()                      # reparameterized
        log_prob = dist.log_prob(alpha.clamp(1e-6, 1-1e-6))
        score = alpha * gat_attack + (1 - alpha) * vgae_anom
        return alpha, score, log_prob
```

| Criterion | PPO | SAC |
|---|---|---|
| Data regime | On-policy, needs rollouts | Off-policy, works on frozen buffer |
| Entropy exploration | Bonus (coef tuned) | Maximization (principled) |
| Sample efficiency | Lower | Higher (replays buffer) |
| Stability | High (clipped surrogate) | Medium (sensitive to α_ent) |
| α collapse prevention | Via entropy coefficient | Via max-entropy objective |
| **Frozen cache** | ✗ Less natural | ✓ Preferred |

**Recommendation:** SAC for frozen-cache; PPO if moving online/streaming.

**Sources:** Haarnoja et al. (2018), ICML 2018; arXiv:1812.05905. Christodoulou (2019),
arXiv:1910.07207.

### 3.4 What to Train the Continuous Policy On (Beyond α)

Two-scalar blend can't exploit the 16 unused features. Three extensions:

**A. 18-dim feature attention.** `w ∈ Δ^17`, `score = w · f`. RL-trained linear classifier;
manageable for SAC with care in Beta/Dirichlet parameterization.

**B. Pre-trained MLP ensemble + gate logits.** K heads on feature subsets (GAT-only [0–6],
VGAE-errors [9–11], VGAE-topology [7,12,13]); policy weights specialists per-sample.
Decomposes into supervised MLP + combinatorial policy. MoE (Jacobs et al., 1991).

**C. Policy controls threshold τ, not fusion weight.** AUROC≈1.0 ⇒ ranking already correct.
Action = `τ ∈ [0,1]` on GAT score; reward = asymmetric FN/FP. Decouples ranking (solved by
GAT) from threshold calibration (unsolved by DQN). State can include batch attack rate,
confidence stats, recent FP/FN — **non-stationary threshold adaptation** for deployment
under concept drift.

**Sources:** Jacobs et al. (1991), Neural Computation 3(1). Hendrycks & Gimpel (2017).
Lakshminarayanan et al. (2017), NeurIPS 2017.

### 3.5 LinUCB Misspecification and NeuralUCB Replacement

LinUCB assumes `E[r | a, x] = x^T θ_a`. Reward has nonlinear interaction terms ⇒
misspecified; ridge `A_inv` fits a linear approximation. AUROC≈1.0 means ranking is easy
enough for the linear approximation; threshold calibration fails because the linear model
can't represent the boundary.

**NeuralUCB** (Zhou et al., 2020):
`UCB(a, x; θ) = f_θ(x, a) + β · ||∇_θ f_θ(x, a)||_{Z_t^{-1}}`, `Z_t` = empirical gradient
covariance. Handles nonlinear reward, uncertainty-driven exploration in attack-minority
regime, proper confidence estimates.

**Sources:** Zhou et al. (2020), ICML 2020. Li et al. (2010), WWW 2010 (LinUCB original).
Zhang et al. (2021), ICLR 2021 (Neural TS).

## 4. Reward Shaping

### 4.1 Theoretical Foundation: Potential-Based Reward Shaping (PBRS)

Ng, Harada, Russell (1999): the only shaping `F(s,a,s')` guaranteeing **policy invariance**
is `F = γΦ(s') − Φ(s)`. Current reward violates PBRS in two places:
1. **`balance` `0.3 × (1 − |α−0.5|×2)`:** function of α alone, not a state-potential diff.
   Peaks at α=0.5, defining a new optimum at α≈0.5 regardless of data. Stated goal
   (penalize α=1.0) is wrong.
2. **Agreement bonus/penalty `+0.3 / −1.0`:** function of two actions, not a state
   potential. On 86%-benign data, always-benign maximizes the bonus (both ≈0).

**Minimum PBRS-compliant reward:**

```python
def pbrs_reward(labels, preds, gat_prob, fn_cost=-6.0, fp_cost=-1.5,
                tp_reward=3.0, tn_reward=1.5):
    """Φ(s)=0 everywhere; pure transition reward. PBRS by construction."""
    r = torch.zeros_like(preds, dtype=torch.float32)
    r[(labels == 1) & (preds == 1)] =  tp_reward
    r[(labels == 0) & (preds == 0)] =  tn_reward
    r[(labels == 0) & (preds == 1)] =  fp_cost
    r[(labels == 1) & (preds == 0)] =  fn_cost      # missed attack: 4× FP cost
    # Confidence bonus gated to attack predictions only (no benign inflation)
    return r + 0.3 * gat_prob * (preds == 1).float()
```

**Sources:** Ng, Harada, Russell (1999), ICML 1999. Devlin & Kudenko (2012), AAMAS 2012
(PBRS, non-stationary).

### 4.2 Asymmetric Cost Design for IDS

For CAN bus (missed attacks → physical harm), FN:FP reflects attack vs. investigation cost.
**F2 (β=2):** recall 2× precision ≡ FN ≈ 4× FP; `fn_cost=-6.0, fp_cost=-1.5` implements
this. **Cost-sensitive SVM baseline:** `class_weight={0:1, 1:4}` gives upper bound on
cost-sensitive non-temporal classifiers without RL.

On hcrl_sa (86% benign), uniform-reward + always-benign earns `E[r] ≈ 1.29`; perfect policy
earns `0.86×1.5 + 0.14×3.0 = 1.71`. Asymmetric costs break the majority-class equilibrium.

**Sources:** Davis & Goadrich (2006), ICML 2006. Elkan (2001), IJCAI 2001.

### 4.3 Pairwise Ranking Reward (AUROC Surrogate)

WMW=AUROC: `AUROC = P(score_attack > score_benign)`. Differentiable surrogate (Yan et al., 2003):

```python
def pairwise_ranking_reward(scores, labels, tau=0.1, n_pairs=256):
    """Majority-class neutral AUROC surrogate. O(N) via random sampling."""
    atk = (labels == 1).nonzero(as_tuple=True)[0]
    ben = (labels == 0).nonzero(as_tuple=True)[0]
    n = min(n_pairs, len(atk), len(ben))
    a = atk[torch.randperm(len(atk))[:n]]
    b = ben[torch.randperm(len(ben))[:n]]
    return torch.sigmoid((scores[a] - scores[b]) / tau).mean() - 0.5  # 0=random,+0.5=perfect
```

Properties: (1) majority-class neutral; (2) expectation = AUROC; (3) differentiable;
(4) τ controls sharpness. Add as a component, not replacement — combine with asymmetric
FN/FP for ranking (AUROC) + calibration (MCC).

**Sources:** Yan et al. (2003), ICML 2003. Narasimhan & Agarwal (2013), NeurIPS 2013.
Eban et al. (2017), AISTATS 2017.

### 4.4 Reward Decomposition for Diagnosing Policy Behavior

Juozapaitis et al. (2019): train separate `V_i(s)` per component `r_i`. Reveals which
component the policy maximizes. Likely finding: DQN optimizes the agreement bonus (large,
frequent, benign-biased), not the ±3.0 classification term.

Simpler: log per-component reward in MLflow (`r_classification`, `r_confidence`,
`r_agreement`, `r_balance`); inspect per-epoch trajectories. If `r_agreement` dominates
while `r_classification` plateaus, diagnosis confirmed.

**Sources:** Juozapaitis et al. (2019), IJCAI-XAI Workshop 2019. Abramson et al. (2020),
arXiv:2012.05672.

## 5. The Suppress Attack Problem and Multi-Task Fusion

### 5.1 Why t05 (Suppress) Will Be 0.000 Across All Methods

Suppress attacks *remove* edges → graph topologically *more benign* by most measures. Both
`gat/probs[:,1]` and `vgae_anom` are trained assuming attacks add anomalous structure;
suppress inverts this — signal is *absence* of expected edges. Features with theoretical
sensitivity:
- `vgae/rq` — Rayleigh quotient `x^T L x / x^T x`; low RQ ⇒ sparse/disconnected
- `vgae/spike` — max masked-node MSE; suppressed-node masking → low MSE (directionally wrong)
- `vgae/z_stats` — latent drift; suppressed graph may project to unusual z

None used in reward path (`derive_scores()` uses only `errors[recon, mahal, kl]`). Visible
to MLP/DQN in flat state, but reward doesn't incentivize their use.

### 5.2 Multi-Task Fusion Architecture

**MoE with learned routing** (Jacobs et al., 1991; Shazeer et al., 2017):

```python
# Expert 1: injection/fuzzy (GAT-dominant)
# Expert 2: suppress (VGAE-topology: rq, z_stats)
# Expert 3: timing (VGAE-temporal: z_stats drift, affinity shift)
class AttackTypeAwareFusion(nn.Module):
    def __init__(self, state_dim=18, hidden=32, n_experts=3):
        super().__init__()
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))
            for _ in range(n_experts)])
        self.router = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, n_experts), nn.Softmax(dim=-1))

    def forward(self, x):
        gates = self.router(x)                                      # [B, n_experts]
        scores = torch.stack([e(x).squeeze(-1) for e in self.experts], dim=-1)
        return (gates * scores).sum(-1), gates
```

`gates` inspectable at test — high `gates[:,1]` ⇒ suppress-type evidence, interpretable
without ground-truth labels. Train: BCE on all experts; router specializes via gradient
routing. Attack-type labels needed for training only, not inference.

**Sources:** Jacobs et al. (1991), Neural Computation 3(1). Shazeer et al. (2017), ICLR
2017. Eigen et al. (2014), ICLR 2014.

## 6. Consolidated Action Plan

**Immediate (before next submission):**
1. **Drop `balance` + `agreement` + `disagreement_penalty`** — violate PBRS, induce
   majority-class bias. Replace with PBRS-compliant asymmetric cost (§4.1).
2. **Platt scaling post-hoc on existing DQN Q-values** — `Q(s,1)−Q(s,0)` ranks correctly;
   recovers calibrated probs without retraining (§3.1).
3. **Resubmit MLP** (Bug-2-era had 1 val epoch only) — clean supervised baseline.

**Short-Term (current sprint):**
4. **Add pairwise ranking reward component** (§4.3).
5. **Replace WeightedAvg with AttentionFusion** (§2.2) — exposes all 18 features per-sample.
6. **Stratify results by attack subtype** — aggregate on set_01/04 conflates subtypes.

**Medium-Term:**
7. **Replace DQN with SAC + continuous α** (§3.3) — off-policy, entropy-maximizing.
8. **Replace LinUCB with NeuralUCB or Neural TS** (§3.5).
9. **Log per-component reward in MLflow** (§4.4).

**Long-Term:**
10. **Multi-task expert fusion for suppress** (§5.2) — needs attack-type training labels.
11. **Online threshold adaptation policy** (§3.4, Option C) — non-stationary attacks.

## 7. Key Sources

| Citation | Relevance |
|---|---|
| Ross & Jain (2003), PRL 24(13) | Fusion taxonomy: feature/score/decision |
| Kittler et al. (1998), IEEE TPAMI 20(3) | Sum rule optimality conditions |
| Ng, Harada, Russell (1999), ICML | PBRS: policy invariance under shaping |
| Yan et al. (2003), ICML | Pairwise AUROC surrogate (WMW) |
| Schulman et al. (2017), arXiv:1707.06347 | PPO: clipped surrogate, GAE |
| Haarnoja et al. (2018), ICML | SAC: max-entropy off-policy continuous |
| Chou et al. (2017), ICML | Beta distribution for bounded actions |
| Zhou et al. (2020), ICML | NeuralUCB |
| Zhang et al. (2021), ICLR | Neural Thompson Sampling |
| Hazarika et al. (2020), ACM MM | Attention-gated fusion |
| Xu et al. (2013), AI Review 42(2) | Multi-view learning survey |
| Jacobs et al. (1991), Neural Computation | Mixture-of-experts |
| Platt (1999), Adv. Large Margin Classifiers | Post-hoc calibration |
| Elkan (2001), IJCAI | Cost-sensitive learning |
| Juozapaitis et al. (2019), IJCAI-XAI | Reward decomposition |
| Dulac-Arnold et al. (2021), JMLR | Real-world RL challenges |
| Emmott et al. (2015), KDD Workshop | Anomaly detection benchmarks |
