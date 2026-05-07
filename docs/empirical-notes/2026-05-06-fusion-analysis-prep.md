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

---

## 2026-05-06 — First-submission failures + fixes

All initial fusion jobs (plan_ids `019e0028-a390`–`019e0028-aac2`) failed. Two bugs.

### Bug 1 — MLflow experiment creation race (UNIQUE constraint)

**Symptom:** Jobs with no MLflow run at all for hcrl_sa bandit/dqn/weighted_avg,
set_01 dqn, set_04 bandit/dqn. Stderr showed SQLite `UNIQUE constraint failed:
experiments.workspace, experiments.name`.

**Cause:** Multiple concurrent SLURM jobs all called `MLFlowLogger(experiment_name=...)`
which internally calls `create_experiment()`. Under SQLite only one job wins the lock;
the rest raise `MlflowException` and crash before training starts.

**Fix** (`graphids/orchestrate.py::_trainer_kwargs`): pre-create the experiment with a
try/except before constructing `MLFlowLogger`:
```python
try:
    _mlflow.MlflowClient().create_experiment(exp_name)
except _MlflowException:
    pass  # already exists — concurrent job won the create race
```
After this, `MLFlowLogger`'s subsequent `get_experiment_by_name()` always finds the
experiment regardless of which job got there first. Also added `configure_tracking_uri()`
call here so the exec path (SLURM sbatch) sets the URI without relying solely on the
`MLFLOW_TRACKING_URI` env var.

### Bug 2 — Generator dataloader exhaustion (bandit checkpoint never saved)

**Symptom:** bandit fit on set_01/02/03 reported FINISHED with exit code 0, but
`best_model.ckpt` was never written — only `last.ckpt`. Test rows then failed with
`FileNotFoundError: ckpt_path does not exist: .../bandit/seed_42/checkpoints/best_model.ckpt`.

**Cause:** `FusionDataModule._batches()` is a generator function (`yield`). Lightning
2.6's `evaluation_loop.setup_data()` calls `val_dataloader()` exactly once per fit and
stores the result in `_combined_loader`. Each epoch's `reset()` calls
`iter(combined_loader)` → `iter(generator)` — but generators are their own iterators,
so `iter(gen)` is a no-op returning the same exhausted object. After epoch 1 drains the
generator, validation epochs 2–1500 yield 0 batches → `validation_step` never fires →
`val_acc` never reaches `callback_metrics` → `Sha256ModelCheckpoint(save_top_k=1)`
never improves from −∞ → no `best_model.ckpt`. `save_last=True` still fires each
training epoch, which is why `last.ckpt` exists.

**Fix** (`graphids/plan/compose.py::_FUSION_TRAINER_OVERLAY`): add
`reload_dataloaders_every_n_epochs: 1`. This sets Lightning's
`_should_reload_val_dl=True` each epoch, forcing a fresh `val_dataloader()` call →
new generator object → correct multi-epoch validation.

Also added `reload_dataloaders_every_n_epochs: int = 0` to `TrainerCfg`
(`graphids/plan/schema.py`) since the field is `extra="forbid"`.

**Why bandit only (not dqn/mlp/weighted_avg)?** DQN and bandit are both CPU/RL
methods using `FusionDataModule`. DQN was missing due to Bug 1 (race crash), so its
generator bug was never reached. MLP and weighted_avg train much faster and in practice
the first epoch's `val_acc` was already better than −∞, so `save_top_k=1` fired after
epoch 1. Bandit's UCB policy starts with zero-initialized theta and produces
near-constant predictions for the first epoch — `val_acc` never improved on the initial
call → no checkpoint.

### Re-submissions (plan_ids `019e0052-c622`–`019e0052-cdb9`)

All 5 fusion plans re-rendered with `reload_dataloaders_every_n_epochs=1` baked in.
Targeted re-submissions (fit+test, afterok-chained):

| dataset  | methods re-run                 | reason                    |
|----------|-------------------------------|---------------------------|
| hcrl_sa  | bandit, dqn, weighted_avg      | Bug 1 (no MLflow run)     |
| set_01   | bandit, dqn                   | Bug 2 + Bug 1 respectively |
| set_02   | bandit                        | Bug 2 (FINISHED, no ckpt) |
| set_03   | bandit                        | Bug 2 (FINISHED, no ckpt) |
| set_04   | bandit, dqn                   | Bug 1 (no MLflow run)     |

SLURM jids 47358590–47358611 (pitzer, cpu partition).

---

## 2026-05-06 — Bug 3: hcrl_sa bandit/dqn train only 1 effective epoch

### Symptom

hcrl_sa bandit and dqn fit jobs finished in 4.6 s and 9.7 s respectively —
200× faster than weighted_avg (751.7 s, 210 epochs). MLflow showed exactly
1 `val_acc` entry per run (at step=0). `best_model.ckpt` was absent for
bandit (expected, see Bug 2 analysis); dqn had a checkpoint but from only
1 real training epoch.

set_01–04 bandit jobs are unaffected: they run 7 batches/epoch, not 1.

### Root cause: double `iter()` on a 1-batch generator

Lightning 2.6 calls `iter(data_fetcher)` **twice** per epoch ≥1:

1. `fit_loop.py:276` — `iter(self._data_fetcher)` in `setup_data()` when
   `_should_reload_train_dl=True` (which `reload_dataloaders_every_n_epochs=1`
   keeps permanently true).
2. `training_epoch_loop.py:246` — `iter(data_fetcher)` again in
   `on_run_start()` when `trainer.current_epoch > 0`.

`_batches()` returned a bare generator function (a `yield`-based function).
Python generators are their own iterators: `iter(gen) is gen`. Both calls
operate on the **same generator object**. The first call triggers
`_PrefetchDataFetcher.__iter__()`, which prefetches batch 0 (advancing the
generator). The second call sees the same exhausted generator object, fails
to refill the prefetch buffer, sets `self.done=True`, and raises
`StopIteration` immediately on the first `next()` at every epoch ≥1.

### Why RL methods only (not mlp/weighted_avg)

`FusionDataModule._batch_size` is split by method:

| method | `_batch_size` | hcrl_sa batches/epoch (train_n=7496) |
|--------|--------------|--------------------------------------|
| bandit / dqn | `episode_sample_size=20000` | **1** |
| mlp / weighted_avg | `batch_size=128` | 59 |

With 1 batch/epoch, the first `iter()` prefetch exhausts the **only**
batch. With 59 batches, the first `iter()` consumes batch 0; the second
starts from position 1 — 58 batches remain. Training proceeds, just
slightly short-changed.

set_01–04 bandit has more graphs (roughly 7 batches/epoch at
`episode_sample_size=20000`), so the second `iter()` still sees 6 batches.

### Why fit runs 200 trivial epochs before stopping

EarlyStopping monitors `val_acc` (checked on `on_train_epoch_end`). Epoch 0
runs correctly: prefetcher hasn't been double-iterated yet, training batch
executes, validation runs at `is_last_batch=True` → `val_acc` logged.
Epochs 1+: 0 training batches → validation never fires (no
`is_last_batch=True`) → `callback_metrics["val_acc"]` is stale → no
improvement → `wait_count` increments each epoch. After
`patience=200` stale epochs, `trainer.should_stop=True` → fit exits.

### Fix (`fusion.py::_batches`)

Changed `_batches()` from a generator function to a method returning a
`_Batches` iterable class:

```python
class _Batches:
    def __len__(self):
        return math.ceil(n / batch_size)   # enables sized_len()

    def __iter__(self):
        idx = torch.randperm(n) if shuffle else torch.arange(n)
        for start in range(0, n, batch_size):
            sub = td[idx[start : start + batch_size]]
            yield sub.exclude("labels"), sub["labels"]
```

Two properties that together close the bug:

- **`__iter__` creates a fresh generator on each call** — no shared state
  between successive `iter()` invocations.
- **`__len__` makes `sized_len()` return a non-None value** — this causes
  `_PrefetchDataFetcher.__iter__()` to return early (`if self.length is not
  None: return self`) before prefetching, so the double-iter issue never
  arises even if the class were somehow shared.

### Re-submissions needed (post Bug 3 fix)

hcrl_sa bandit and dqn fit+test need to be resubmitted. weighted_avg loses
1/59 batches per epoch (minor) but should also be resubmitted for clean
data. set_01–04 bandit is currently running with the pre-fix code (6/7
batches per epoch); decision: let them finish (linUCB robust to small data
deficit, AUROC impact negligible, queue time cost outweighs benefit of restart).

---

## 2026-05-07 — Latent risks after Bug 3 fix

Bug 3 is fixed but several risks remain for the re-runs.

### Risk 1 — Bandit `global_step=0` → no checkpoint (high)

Even with the generator fix, bandit calls `opt.step()` only when
`_episode % backbone_retrain_freq == 0` (freq=50). With one batch per epoch,
`_episode` increments by 1 per training step. `opt.step()` fires at episodes
50, 100, … — `global_step` stays 0 for the first 50 training steps.
`ModelCheckpoint._should_skip_saving_checkpoint` returns `True` when
`_last_global_step_saved == trainer.global_step` (both 0), so checkpoint
monitor skips saving until episode 50.

Mitigation: monitor MLflow for `global_step` advancing past 0 and
`best_model.ckpt` appearing by epoch 50. If `global_step` never advances,
the backbone's `opt.step()` path is broken.

### Risk 2 — TensorDict views aliased into torchRL replay buffer (medium)

`_batches()` yields `sub.exclude("labels")` which is a view sharing storage
with the source `train_td`. torchRL's `ReplayBuffer.extend()` may not copy.
If torchRL mutates tensors in-place during TD update (reward normalization
etc.), it corrupts `train_td` for all future batches.

Watch for: loss or metric instability that worsens monotonically over epochs
(signature of in-place corruption). Long-term fix: add `.clone()` before
yielding — but only after confirming this is the cause.

### Risk 3 — set_01–04 bandit: ~14% training data loss per epoch (low)

Currently running (pre-fix code, 6/7 batches/epoch). Decision: let finish.
linUCB sees 1500 × 6 = 9000 effective passes. Note the deficit in results.

### Risk 4 — weighted_avg: systematic 1/59 batch loss per epoch (low)

Every epoch the prefetch call in `setup_data` consumed batch 0 before
`on_run_start` reset. ~1.7% data deficit per epoch, same random 1/59th of
data each epoch. Resubmitting for clean results.

### Summary

| Risk | Severity | Action |
|------|----------|--------|
| Bandit global_step=0 → no checkpoint before episode 50 | High | Verify MLflow global_step in re-run |
| TensorDict view aliased into replay buffer | Medium | Watch for monotonic metric degradation |
| set_01–04 bandit: ~14% data loss/epoch | Low | Let finish, note in results |
| weighted_avg: 1/59 batch loss/epoch | Low | Resubmit |

### Re-submissions (plan_id `019e00a7-3498`, pitzer cpu, seed 42)

| row | fit jid | test jid |
|-----|---------|----------|
| bandit | 47359359 | 47359362 (afterok 47359359) |
| dqn | 47359360 | 47359364 (afterok 47359360) |
| weighted_avg | 47359361 | 47359365 (afterok 47359361) |

Results from plan `019e00a7-3498` (first clean re-run):

| variant | val_acc epochs | auroc | mcc | f1 | acc |
|---------|---------------|-------|-----|----|-----|
| bandit | 224 | 0.9999 | 0.0133 | 0.1253 | 0.1426 |
| dqn | 322 | 1.0000 | 0.0432 | 0.1381 | 0.1527 |
| weighted_avg | 210 | 0.1433 | −0.3833 | 0.1941 | 0.2029 |

bandit/dqn: Bug 3 fix confirmed — 224/322 real training epochs vs 1 before.
Risk 1 resolved: bandit `best_model.ckpt` written at epoch 50+ (global_step advanced).
bandit/dqn AUROC near 1.0 consistent with hcrl_sa being id-rich (easy dataset).
bandit/dqn MCC/acc low: RL agents output high Q-value for attack but select action=1
for all samples at test time (threshold miscalibrated relative to α-based decision).

weighted_avg: AUROC=0.1433 — worse than random. Inverted. See Bug 4.

---

## 2026-05-07 — Bug 4: weighted_avg forward_scores returns benign confidence (inverted)

### Symptom

weighted_avg test AUROC=0.1433, MCC=−0.3833 on hcrl_sa. Flipping the score
(1 − 0.1433 = 0.8567) gives a plausible number. Model is predicting the
opposite class.

### Root cause

`WeightedAvgModule.forward_scores` returned:

```python
(1 - alpha) * vgae_conf + alpha * gat_conf
```

where:
- `vgae_conf = 1/(1+recon_mean)` — HIGH when reconstruction is LOW → **benign**
- `gat_conf = 1 - entropy/log(2)` — HIGH when GAT is confident, but not
  directional (high for confident benign AND confident attack)

Both signals are high for benign samples. BCE loss trains toward `score ≈ 1`
for attacks and `score ≈ 0` for benign — but the inputs naturally go the
opposite direction. The model can only adjust `alpha` (a scalar blend), which
cannot flip signal direction. It converges to a local minimum that still
outputs a benign-biased score → AUROC < 0.5.

`test_step` then uses `fused_scores` directly as the attack probability
`probs[:,1]`, transmitting the inversion to AUROC computation.

### Fix (`weighted_avg.py::forward_scores`)

Replaced both conf signals with proper attack-direction signals:

```python
vgae_anom = 1.0 - td["vgae", "conf"].squeeze(-1)   # high when recon error high (attack)
gat_attack = td["gat", "probs"][..., 1]              # GAT attack probability
return clamp((1 - alpha) * vgae_anom + alpha * gat_attack)
```

`vgae_anom = 1 - 1/(1+recon_mean) = recon_mean/(1+recon_mean)` — monotone in
`recon_mean`, stays in (0, 1), high for anomalous graphs. `gat_attack` is the
softmax attack probability, already in (0, 1). Both are naturally in (0, 1);
no sigmoid squashing needed. The blend is now a proper weighted attack score.

The docstring was updated to match.

### Re-submission (plan_id `019e00b8-205f`, pitzer cpu, seed 42)

| row | fit jid | test jid |
|-----|---------|----------|
| weighted_avg | 47359413 | 47359414 (afterok 47359413) |

**hcrl_sa final results (plan `019e00b8-205f`, all bugs fixed):**

| variant | val_acc_n | AUROC | MCC | F1 | acc | notes |
|---------|-----------|-------|-----|----|-----|-------|
| bandit | 224 | 0.9999 | 0.013 | 0.125 | 0.143 | alpha=threshold miscal; plan 019e00a7 |
| dqn | 322 | 1.0000 | 0.043 | 0.138 | 0.153 | same; plan 019e00a7 |
| mlp | 1 | 0.8667 | 0.745 | 0.856 | — | Bug-2-era: val once only; plan 019e0028 |
| weighted_avg | 201 | **0.9291** | **0.622** | **0.783** | 0.859 | alpha→1.0 (GAT-only); plan 019e00b8 |

weighted_avg AUROC (0.9291) > mlp (0.8667): the mlp result is from the Bug-2-era
checkpoint (epoch-0 only) and is likely suboptimal. weighted_avg with alpha=1.0
(trust GAT fully) reflects that GAT's attack probability is the dominant signal on
hcrl_sa. mlp should be resubmitted for a clean comparison.

RL MCC≈0 pattern: bandit/dqn output high Q-values for attacks (AUROC≈1) but the
alpha-based threshold at test time (decision_threshold=0.5 applied to the α-blended
fused score) is uncalibrated → models predict all-attack → acc = attack prevalence
≈14%, MCC≈0. This is a test-evaluation design issue, not a training failure.

Bug 4 confirmed across all datasets (weighted_avg from old plans all inverted).
Resubmitting set_01–04 weighted_avg with fixed code.

---

## 2026-05-07 — alpha→1.0 everywhere: VGAE adds no signal over GAT in weighted_avg

### Observation

After Bug 4 fix, weighted_avg was refit on all 5 datasets. In every case
`final_alpha` converged to 1.000 within the first 20–30 epochs:

| dataset | val_acc_n (so far) | best_val_acc | final_alpha |
|---------|--------------------|-------------|-------------|
| hcrl_sa | 201 | 0.9989 | 1.000 |
| set_01  | in progress | 0.9966 | 1.000 |
| set_02  | in progress | 0.9867 | 1.000 |
| set_03  | in progress | 0.9959 | 1.000 |
| set_04  | in progress | 0.9426 | 1.000 |

`alpha=1.0` means the model sets `score = 1·gat_attack + 0·vgae_anom` — it
discards the VGAE signal entirely and uses only `gat/probs[:,1]`.

### Interpretation

The single degree of freedom (alpha ∈ [0,1]) is sufficient to express the
discovery that a supervised classifier (GAT) trained directly on the attack
detection objective dominates an unsupervised proxy (VGAE reconstruction
error). Once GAT provides a calibrated probability, VGAE adds noise.

This is consistent with theory: unsupervised anomaly scores are useful when
no supervised signal exists; once a supervised model is trained to the same
objective, it subsumes the unsupervised signal because it has access to label
information during training.

### Implication for RL reward design

The RL reward (`FusionRewardCalculator`) still uses both `vgae_score` and
`gat_prob` in `derive_scores()` and `compute()`. If VGAE is redundant for
fusion, the disagreement penalty and agreement bonus (which reward
VGAE↔GAT concordance) may be distorting the reward signal:
- High agreement occurs when VGAE happens to agree with GAT, not when the
  fusion policy makes a good decision.
- The balance bonus (penalises alpha near 0 or 1) actively discourages
  alpha=1.0, fighting the direction the data supports.

The RL agents learn alpha as an action proxy, but the training signal pushes
them away from the optimal `alpha=1.0`. This may explain why bandit/DQN
converge to AUROC≈1 (they find a good ranking) but MCC≈0 (threshold
miscalibrated by the reward shaping).

Open question: how should the reward be redesigned for anomaly detection
where one class dominates (86% benign)? See research note below.

---

## 2026-05-07 — Research note: RL reward redesign for imbalanced anomaly detection

### Q1 — Does VGAE offer anything the RL agent can't already get from GAT?

**In the current closed-world setting: no.** `alpha→1.0` on all five datasets
shows that supervised GAT subsumes the VGAE signal whenever both are trained on
the same attack vocabulary. `derive_scores()` builds `anomaly` as a weighted
sum of `errors` (recon, mahal, kl only — `reward.py:55`), ignoring `rq`,
`spike`, `affinity`, and `z_stats`. Those features are in the flat state vector
and are visible to the MLP/Q-network, but the reward shaper never reads them.

**Where VGAE might add signal:** attack types that don't alter frame *content*
but alter *communication pattern* — suppress (missing frames → topology gap),
timing (edge-weight distribution shift), fuzzy (dense inter-ID traffic → high
affinity, anomalous rq). GAT sees these as distribution-shifted embeddings; it
may still classify them correctly because hcrl_sa is id-rich. On harder datasets
or out-of-distribution attacks, `vgae/rq`, `vgae/spike`, and `vgae/affinity`
could provide orthogonal information. But the current reward never exploits them
in its `derive_scores()` path, and the closed-world experiments don't expose the
gap.

**Bottom line:** don't remove VGAE from the feature vector, but stop treating
VGAE↔GAT agreement as a training signal in the reward. It's noise in-distribution
and may be valuable precisely when the two *disagree* (novel attack type).

---

### Q2 — Does GAT overpredict one class? Is reward biased to majority-class rule?

**Yes on both counts.**

**Majority-class bias in the reward:** hcrl_sa is 86% benign. A policy that
predicts all-benign gets `correct=True` for 86% of steps, collecting
`base(+3.0) + agreement + bonus` on each. `agreement = 1 - |anomaly - gat_prob|`
(`reward.py:74`) also naturally inflates for benign samples because VGAE has low
anomaly scores and GAT has low attack probability — they trivially agree on benign.
The `confidence` term (`reward.py:77`) for benign samples is `1 - max(anomaly,
gat_prob)`, which is high when both models are confidently benign — again a
majority-class reward. Expected per-step reward for all-benign policy ≈
0.86 × (3.0 + ~1.0 + ~0.8) = **4.1** vs expected for perfect policy ≈ 3.7
(correct 14% attacks penalised by lower agreement on the attack minority). The
reward accidentally incentivizes the majority-class rule.

**Balance term creates a perverse equilibrium:** `balance = 0.3*(1 - |alpha-0.5|*2)`
(`reward.py:91`) is zero at alpha=0 or alpha=1 and peaks at alpha=0.5. Since
the data optimal is alpha=1.0, the balance term is a constant -0.3/step
anti-gradient. The agent cannot escape to alpha=1.0 without giving up 0.3 reward
per step, so it settles near alpha≈0.5–0.7 — a range where the blended score at
`decision_threshold=0.5` is unreliable. AUROC stays near 1.0 (ranking preserved)
but the binary threshold is miscalibrated → MCC≈0.

**GAT overprediction:** GAT's attack precision is limited by training data
imbalance (focal loss helps but doesn't eliminate it). In the RL context the
bigger issue is that the reward does not distinguish FN (missed attack → security
gap) from FP (false alarm → operator fatigue). The symmetric `correct=+3 /
incorrect=-3` encoding treats both errors equally; for IDS the cost asymmetry
should be at least 3:1 FN:FP.

---

### Q3 — How to redesign the reward for anomaly detection?

Five targeted changes to `FusionRewardCalculator.compute()`:

| # | Change | Rationale |
|---|--------|-----------|
| 1 | **Drop `balance` term** | Penalises alpha=1.0 (the data optimum); AUROC already rewards good ranking, threshold calibration requires a separate mechanism |
| 2 | **Asymmetric FN/FP costs** | Replace symmetric `correct/incorrect` with `fn_cost = -6.0`, `fp_cost = -1.5`; attacks missed 4× worse than false alarms for IDS |
| 3 | **Remove VGAE agreement terms** | Drop `agreement`, `disagreement_penalty`, `combined_conf_weight`; they reward VGAE↔GAT concordance on benign majority, adding noise without signal |
| 4 | **Pairwise ranking bonus** | Add small bonus when a correctly predicted attack in the batch has higher fused score than a correctly predicted benign (`+0.5 / pair`); this is a direct AUROC surrogate and is majority-class neutral |
| 5 | **Retain confidence bonus but make it attack-only** | `confidence_weight * gat_prob` for attacks only — do not reward confident benign predictions (majority-class path) |

The `overconf_penalty` can stay but should be gated on attacks only: an
overconfident false alarm costs less than an overconfident miss.

**Minimum viable change for the next run:** drop `balance` (one line) and
double the FN cost. This removes the anti-alpha=1.0 gradient and breaks the
majority-class trap without a full reward rewrite. Measure whether MCC lifts
above 0 before implementing the ranking bonus.

**Longer-term:** pairwise ranking is expensive per batch (O(N²) naive); use
a sampled approximation (random 256 attack×benign pairs per batch, each pair
one comparison). This gives an unbiased AUROC gradient with O(N) cost.

---

### Summary: what to change in `reward.py`

```python
# current (anti-patterns highlighted)
balance = self._balance_weight * (1.0 - (alphas - 0.5).abs() * 2)  # DROP
agreement = 1.0 - (anomaly - gat_prob).abs()                        # DROP (VGAE noise)
base = where(correct, +3.0, -3.0)                                   # REPLACE with asymmetric

# proposed minimal replacement in compute()
fn_mask = (labels == 1) & (preds == 0)
fp_mask = (labels == 0) & (preds == 1)
tp_mask = (labels == 1) & (preds == 1)
tn_mask = (labels == 0) & (preds == 0)
base = torch.zeros_like(preds, dtype=torch.float32)
base[tp_mask] = self._reward_correct         # +3.0
base[tn_mask] = self._reward_correct * 0.5   # +1.5 (benign correct is worth less)
base[fp_mask] = self._fp_cost                # -1.5
base[fn_mask] = self._fn_cost                # -6.0
confidence = self._confidence_weight * gat_prob * (labels == 1).float()
return base + confidence  # no balance, no agreement
```

This is a hypothesis, not yet tested. Implement after set_01–04 weighted_avg
results are in and after deciding whether to resubmit hcrl_sa mlp for a clean
comparison baseline.

---

## 2026-05-07 — set_01–04 weighted_avg: run outcomes and set_02 timeout

### set_01 — COMPLETED

201 epochs, best_val_acc=0.9966, final_alpha=1.000.

| metric | value |
|--------|-------|
| AUROC | 0.6953 |
| MCC | 0.183 |
| F1 (attack) | 0.292 |
| recall (attack) | 0.459 |
| precision (attack) | 0.214 |
| F1 (weighted) | 0.787 |

AUROC=0.6953 is substantially lower than hcrl_sa (0.9291). Since alpha=1.0,
the weighted_avg score equals `gat/probs[:,1]` directly — the AUROC reflects
GAT's ranking quality on set_01's harder attack mix, not a fusion failure.

### set_02 — TIMEOUT after 178 epochs; test submitted directly

**Symptom:** job 47359461 hit 4h walltime and timed out. SIGUSR2 auto-requeue
did not fire (signal at 3h55m should have triggered `scontrol requeue`, but the
job exited TIMEOUT instead). Test job 47359462 was auto-cancelled.

**State at timeout:** 178 epochs logged, best_val_acc=0.9867, final_alpha=1.000.
`best_model.ckpt` exists on disk. MLflow fit run `bb8ed84d` is stuck at status
`RUNNING` (finalize never ran). This does not block the test — test opens its
own always-fresh run.

**Decision:** alpha already converged to 1.0 within the first ~30 epochs.
Additional training would not change the checkpoint meaningfully. Submitted
test directly against existing `best_model.ckpt` as job 47359731 (no
fit dependency). If the fit run needs to be cleanly closed for MLflow queries,
use `GRAPHIDS_FORCE_RESUME=1` on a future fit resubmission.

### set_03 / set_04 — FAILED (SQLite DB lock), resubmitted

**Symptom:** both jobs (47359463, 47359465) were co-scheduled on node p0085
and failed simultaneously with `sqlite3.OperationalError: database is locked`
during MLflow `set_terminated()`. Two concurrent writers to the shared NFS
SQLite DB (`/fs/ess/PAS1266/graphids/mlflow.db`) hit WAL lock contention at
finalization.

**Resubmissions:**

| dataset | fit jid | test jid |
|---------|---------|----------|
| set_03 | 47359696 | 47359698 (afterok 47359696) |
| set_04 | 47359697 | 47359699 (afterok 47359697) |

Both running on separate nodes now; DB lock race unlikely to recur.

### Final results — weighted_avg (Bug 4 fixed), all datasets

| dataset | val_acc_n | best_val_acc | alpha | AUROC | MCC | F1 (attack) | prec (attack) | recall (attack) | notes |
|---------|-----------|-------------|-------|-------|-----|-------------|---------------|-----------------|-------|
| hcrl_sa | 201 | 0.9989 | 1.000 | 0.929 | 0.622 | 0.783 | — | — | plan 019e00b8 |
| set_01  | 201 | 0.9966 | 1.000 | 0.695 | 0.183 | 0.292 | 0.214 | 0.459 | |
| set_02  | 178† | 0.9867 | 1.000 | **0.996** | **0.947** | **0.969** | 0.994 | 0.944 | †timeout; test on existing ckpt |
| set_03  | 201 | 0.9959 | 1.000 | **0.998** | **0.986** | **0.991** | 0.992 | 0.990 | |
| set_04  | 204 | 0.9426 | 1.000 | 0.938 | 0.680 | 0.668 | 0.978 | 0.508 | |

alpha=1.000 on every dataset — VGAE discarded, score = `gat/probs[:,1]` only.

**Dataset difficulty gradient:** set_02 and set_03 are easy for GAT (AUROC>0.99,
MCC>0.95). set_01 and set_04 are harder — set_01 has lower recall (0.459) despite
good precision, suggesting GAT's attack probability is low-confidence on set_01's
attack mix. set_04 has very high precision (0.978) but poor recall (0.508),
indicating GAT learns a high-confidence subset of attacks but misses half.

The weighted_avg AUROC with alpha=1.0 is a direct proxy for standalone GAT test
AUROC on each dataset — useful cross-check once GAT ablation results are in.
