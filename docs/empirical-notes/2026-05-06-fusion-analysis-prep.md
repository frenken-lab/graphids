# 2026-05-07 — Fusion ablation: bugs, fixes, and results

Jobs submitted: plan_ids `019e0028-a390` through `019e0028-aac2` (hcrl_sa, set_01–04,
seed 42). Four fusion methods: bandit, dqn, mlp, weighted_avg. Four bugs found and
fixed. Key finding: **alpha→1.0 on all datasets** — VGAE adds no signal over GAT;
weighted_avg score degenerates to `gat/probs[:,1]` alone.

---

## Architecture reference

### Extracted states

The fusion pipeline does not run VGAE/GAT live. An extract job runs both in inference
mode and writes per-graph feature tensors to
`{states_dir(dataset, seed)}/fusion_states/{train,val}_states.pt`. Every fusion model
trains on this frozen cache — it never sees raw graphs.

### Feature layout — 18-dimensional flat vector

`flatten_features(td)` sorts TensorDict keys lexicographically and concatenates
along the last dim. Always in this exact order:

| pos   | key             | shape | source | meaning                                                                     |
| ----- | --------------- | ----- | ------ | --------------------------------------------------------------------------- |
| 0     | `gat/conf`      | [N,1] | GAT    | `1 − entropy/log(2)`. 0=max-entropy, 1=certain                              |
| 1–4   | `gat/emb_stats` | [N,4] | GAT    | graph-embedding statistics: mean, std, max, min                             |
| 5–6   | `gat/probs`     | [N,2] | GAT    | softmax: `[prob_benign, prob_attack]`; `probs[:,1]` is the GAT attack score |
| 7     | `vgae/affinity` | [N,1] | VGAE   | TAM per-graph mean affinity — latent clustering relative to benign manifold |
| 8     | `vgae/conf`     | [N,1] | VGAE   | `1/(1+recon_mean)` — high when reconstruction is low (benign)               |
| 9–11  | `vgae/errors`   | [N,3] | VGAE   | `[recon, mahal, kl]`; stored as one [N,3] tensor                            |
| 12    | `vgae/rq`       | [N,1] | VGAE   | Rayleigh quotient — spectral smoothness; low = anomalous topology           |
| 13    | `vgae/spike`    | [N,1] | VGAE   | `recon_max` — max masked-node MSE; catches spike attacks where mean is low  |
| 14–17 | `vgae/z_stats`  | [N,4] | VGAE   | latent z: mean, std, max, min                                               |

Total: **18 dims** (`_state_dim = 18`).

### Models

**MLP** — Fully-connected net over the flat 18-dim vector, `BCEWithLogitsLoss`. Direct
supervised baseline; maximum AUROC potential from the given features.

**WeightedAvg** — Single scalar `w = sigmoid(θ)` blends attack-direction signals:
`score = w·gat_attack + (1−w)·vgae_anom`. Initialises at `w=0.5` (θ=0). `alpha→1.0`
means score equals `gat/probs[:,1]` and VGAE is discarded entirely.

**Bandit** — LinUCB contextual bandit. Maintains ridge-regularized precision matrix
`A_inv` per arm; Sherman-Morrison rank-1 updates — no gradient descent. `max_epochs=1500`
means 1500 passes over the shuffled buffer. Optimises cumulative reward, not AUROC.

**DQN** — Deep Q-network (torchrl). State=18-dim; actions={benign=0, attack=1}.
TD(0) replay buffer. Highest capacity RL method but most sensitive to reward shaping.

### RL reward structure (bandit and DQN)

Defined in `primitives.py::REWARD`, applied via `reward.derive_scores()`:

```
vgae_score  = dot(errors, [0.4, 0.3, 0.3])   # recon:mahal:kl weighted
gat_score   = probs[:, 1]

reward = ±3.0 (correct/incorrect)
       + 0.5 × mean(gat_conf, vgae_conf)       # confidence bonus
       + 0.3 if models agree                   # agreement bonus
       − 1.0 if disagree                       # disagreement penalty
       − 1.5 if wrong + confident              # overconfidence penalty
       + 0.3 × (1 − |alpha−0.5|×2)            # balance (peaks at alpha=0.5)
```

The `balance` term penalises alpha=1.0 (the data-optimal direction). The agreement
terms inflate reward on the benign majority. See reward redesign section.

---

## Bugs and fixes

### Bug 1 — MLflow experiment creation race (UNIQUE constraint)

**Symptom:** Jobs crash before training: SQLite `UNIQUE constraint failed:
experiments.workspace, experiments.name`. Multiple concurrent SLURM jobs call
`MLFlowLogger(experiment_name=...)`, which calls `create_experiment()` internally.
Only one wins the SQLite lock; the rest crash before training starts. Affected:
hcrl_sa bandit/dqn/weighted_avg, set_01 dqn, set_04 bandit/dqn.

**Fix** (`orchestrate.py::_trainer_kwargs`): pre-create the experiment with
`try/except MlflowException` before constructing `MLFlowLogger`. All jobs then
find the experiment via `get_experiment_by_name()` regardless of which won the
race. Also wired `configure_tracking_uri()` so the exec path (sbatch) sets the
URI without relying solely on `MLFLOW_TRACKING_URI`.

### Bug 2 — Generator dataloader exhaustion (no best_model.ckpt)

**Symptom:** bandit fit on set_01/02/03 finished (exit code 0) with no
`best_model.ckpt` — only `last.ckpt`. Test rows then failed with `FileNotFoundError`.

**Cause:** `FusionDataModule._batches()` was a `yield`-based generator function.
Lightning 2.6 calls `val_dataloader()` once per fit and stores the result in
`_combined_loader`. Each epoch's `reset()` calls `iter(combined_loader)` →
`iter(generator)`. But generators are their own iterators: `iter(gen) is gen`,
a no-op returning the same object. After epoch 1 drains the generator, all
subsequent validation epochs yield 0 batches → `val_acc` never logged →
`ModelCheckpoint` never improves from −∞ → no `best_model.ckpt`. `save_last=True`
still fires each training epoch, which is why `last.ckpt` exists.

**Why bandit only:** DQN was missing due to Bug 1. MLP/weighted_avg train fast enough
that epoch-1 `val_acc` beat −∞ (the initial value), so `save_top_k=1` fired after
epoch 1. Bandit's UCB policy starts zero-initialized and produces near-constant
predictions; epoch-1 val_acc never beat −∞.

**Fix** (`compose.py::_FUSION_TRAINER_OVERLAY`): `reload_dataloaders_every_n_epochs: 1`.
Forces a fresh `val_dataloader()` call each epoch → new generator → correct
multi-epoch validation. Field added to `TrainerCfg` (`extra="forbid"`).

### Bug 3 — Double `iter()` on 1-batch generator (RL methods train 1 effective epoch)

**Symptom:** hcrl_sa bandit/dqn finished in 4.6 s / 9.7 s vs 751 s for weighted_avg.
MLflow showed exactly 1 `val_acc` entry per run.

**Cause:** Bug 2's fix (`reload_dataloaders_every_n_epochs=1`) causes Lightning 2.6
to call `iter(data_fetcher)` twice per epoch ≥1:

1. `fit_loop.py:276` — `iter(self._data_fetcher)` in `setup_data()` when
   `_should_reload_train_dl=True`.
2. `training_epoch_loop.py:246` — `iter(data_fetcher)` again in `on_run_start()`
   when `trainer.current_epoch > 0`.

Both calls get the same generator object. The first call prefetches and exhausts the
only batch; the second sees the exhausted generator, sets `done=True`, and raises
`StopIteration` immediately. With 0 training steps, `is_last_batch` never fires →
validation skipped → `callback_metrics["val_acc"]` is stale → EarlyStopping waits
`patience=200` stale epochs before stopping.

**Why RL methods only:**

| method             | hcrl_sa batches/epoch                        | effect                                     |
| ------------------ | -------------------------------------------- | ------------------------------------------ |
| bandit / dqn       | 1 (episode_sample_size=20000 ≫ train_n=7496) | first `iter()` exhausts the only batch     |
| mlp / weighted_avg | 59 (batch_size=128)                          | first `iter()` consumes batch 0; 58 remain |

set_01–04 RL runs minimally affected (≥7 batches/epoch → lose batch 0 only).
Decision: let those runs finish (linUCB robust to small data deficit; resubmitting
costs more than the 1/7 batch deficit).

**Fix** (`fusion.py::_batches`): replaced generator function with a `_Batches`
iterable class:

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

`__iter__` creates a fresh generator on each call — no shared state between
successive `iter()` invocations. `__len__` makes `sized_len()` return non-None →
`_PrefetchDataFetcher.__iter__()` returns early before prefetching (`if self.length
is not None: return self`), so the double-iter is a no-op regardless.

### Bug 4 — weighted_avg forward_scores returns benign confidence (inverted AUROC)

**Symptom:** weighted_avg test AUROC=0.1433, MCC=−0.3833 on hcrl_sa. Flipping the
score gives 0.857 — model predicts the wrong class.

**Root cause** (`weighted_avg.py::forward_scores` before fix):

```python
(1 - alpha) * vgae_conf + alpha * gat_conf
```

- `vgae_conf = 1/(1+recon_mean)` — HIGH for benign (low reconstruction error)
- `gat_conf = 1 − entropy/log(2)` — undirectional (high for confident benign AND attack)

BCE trains toward `score≈1` for attacks, but both inputs naturally peak for benign
samples. The scalar `alpha` cannot flip signal direction → converges to a
benign-biased local minimum. `test_step` passes `fused_scores` directly as
`probs[:,1]` → AUROC < 0.5.

**Fix** (`weighted_avg.py::forward_scores`):

```python
vgae_anom = 1.0 - td["vgae", "conf"].squeeze(-1)   # high when recon error high (attack)
gat_attack = td["gat", "probs"][..., 1]              # GAT attack probability
return clamp((1 - alpha) * vgae_anom + alpha * gat_attack)
```

`vgae_anom = recon_mean/(1+recon_mean)` — monotone in `recon_mean`, high for anomalous
graphs. `gat_attack` is the softmax attack probability. Both naturally in (0,1) and
attack-direction. Bug 4 affected all datasets — all weighted_avg results from plans
`019e0028-*` are invalid.

---

## Results

### hcrl_sa — all methods

Plans: bandit/dqn from `019e00a7`; weighted_avg from `019e00b8`; mlp from `019e0028`.

| variant      | val_acc_n | AUROC      | MCC       | F1        | acc   | notes                                   |
| ------------ | --------- | ---------- | --------- | --------- | ----- | --------------------------------------- |
| bandit       | 224       | 0.9999     | 0.013     | 0.125     | 0.143 | threshold miscal; all-attack prediction |
| dqn          | 322       | 1.0000     | 0.043     | 0.138     | 0.153 | same                                    |
| mlp          | 1         | 0.8667     | 0.745     | 0.856     | —     | Bug-2-era (1 val epoch only) — resubmit |
| weighted_avg | 201       | **0.9291** | **0.622** | **0.783** | 0.859 | alpha=1.0 (GAT-only)                    |

bandit/dqn AUROC≈1.0 (good ranking) but MCC≈0 — see RL threshold miscalibration below.
mlp AUROC is a lower bound; resubmit for a clean comparison baseline.

### All datasets — weighted_avg (Bug 4 fixed, alpha=1.0 everywhere)

| dataset | val_acc_n | best_val_acc | final_alpha | AUROC     | MCC       | F1 (atk)  | prec  | recall | notes                              |
| ------- | --------- | ------------ | ----------- | --------- | --------- | --------- | ----- | ------ | ---------------------------------- |
| hcrl_sa | 201       | 0.9989       | 1.000       | 0.929     | 0.622     | 0.783     | —     | —      | plan 019e00b8                      |
| set_01  | 201       | 0.9966       | 1.000       | 0.695     | 0.183     | 0.292     | 0.214 | 0.459  |                                    |
| set_02  | 178†      | 0.9867       | 1.000       | **0.996** | **0.947** | **0.969** | 0.994 | 0.944  | †timeout 4h; test on existing ckpt |
| set_03  | 201       | 0.9959       | 1.000       | **0.998** | **0.986** | **0.991** | 0.992 | 0.990  |                                    |
| set_04  | 204       | 0.9426       | 1.000       | 0.938     | 0.680     | 0.668     | 0.978 | 0.508  |                                    |

alpha converged to 1.000 on every dataset within 20–30 epochs. Score = `gat/probs[:,1]`
only; weighted_avg AUROC is a direct proxy for standalone GAT AUROC on each dataset.

Dataset difficulty: set_02/03 easy (AUROC>0.99, MCC>0.94). set_01 low recall (0.459)
despite good precision — GAT attack probability low-confidence on set_01's attack mix.
set_04 high precision (0.978) but low recall (0.508) — GAT learns a high-confidence
subset but misses half.

Structural caveat: t05 (suppress attacks) will be 0.000 across all methods — neither
VGAE nor GAT produces a useful score for sparse suppress graphs. Expected.

---

## Findings

### VGAE adds no signal over GAT (alpha→1.0 everywhere)

Consistent with theory: a supervised classifier trained on the detection objective
subsumes an unsupervised proxy because it has access to labels during training.

The reward shaper (`derive_scores()`) builds `vgae_score` from only `errors[recon,
mahal, kl]` (reward.py:55), ignoring `rq`, `spike`, `affinity`, and `z_stats`. Those
features are visible to MLP/Q-network in the flat state but are not exploited in the
reward path. They may provide orthogonal signal on out-of-distribution attacks:
suppress (sparse topology → high rq gap), timing (edge-weight shift → z_stats drift),
fuzzy (dense inter-ID traffic → high affinity, anomalous rq). Do not remove VGAE
from the feature vector; stop using VGAE↔GAT agreement as a reward signal.

### RL threshold miscalibration (AUROC≈1, MCC≈0)

bandit/dqn rank attacks well but `decision_threshold=0.5` applied to the alpha-blended
score is uncalibrated. Root cause: the `balance` term penalises alpha=1.0, keeping
alpha near 0.5–0.7 where the blended score is unreliable for threshold decisions.

Additional structural bias: hcrl_sa is 86% benign. A policy predicting all-benign
collects +3.0 for 86% of steps plus trivial agreement/confidence bonuses ≈ 4.1
expected reward/step — higher than the expected reward for a perfect policy (≈3.7,
because detecting the attack minority earns fewer agreement bonuses). The reward
accidentally incentivizes the majority-class rule.

### Reward redesign proposal

Minimum viable change (`reward.py::compute()`):

```python
fn_mask = (labels == 1) & (preds == 0)
fp_mask = (labels == 0) & (preds == 1)
tp_mask = (labels == 1) & (preds == 1)
tn_mask = (labels == 0) & (preds == 0)
base = torch.zeros_like(preds, dtype=torch.float32)
base[tp_mask] = self._reward_correct         # +3.0
base[tn_mask] = self._reward_correct * 0.5   # +1.5 (benign correct worth less)
base[fp_mask] = self._fp_cost                # -1.5
base[fn_mask] = self._fn_cost                # -6.0
confidence = self._confidence_weight * gat_prob * (labels == 1).float()
return base + confidence  # drop: balance, agreement, disagreement_penalty
```

Changes: (1) drop `balance` — removes anti-alpha=1.0 gradient; (2) asymmetric FN/FP
costs — FN 4× FP matches IDS cost asymmetry; (3) drop `agreement`/`disagreement_penalty`/
`combined_conf_weight` — VGAE↔GAT concordance on the benign majority is noise; (4)
confidence bonus gated to attacks only.

Longer-term: pairwise ranking bonus — sample 256 attack×benign pairs/batch and award
+0.5 when attack score > benign score. Direct AUROC surrogate, O(N) cost, majority-class
neutral. Implement after set_01–04 bandit/dqn results are in.

---

## Open risks

| Risk                                                                                             | Severity | Action                                                                              |
| ------------------------------------------------------------------------------------------------ | -------- | ----------------------------------------------------------------------------------- |
| Bandit `global_step=0` → no checkpoint before episode 50                                         | High     | Verify MLflow `global_step` in re-runs; `best_model.ckpt` should appear by epoch 50 |
| TensorDict view aliased into torchRL replay buffer (in-place corruption of `train_td`)           | Medium   | Watch for monotonic metric degradation; add `.clone()` before yielding if confirmed |
| set_01–04 bandit: ~14% data loss/epoch (pre-Bug-3 code, currently running)                       | Low      | Let finish; note deficit in results                                                 |
| set_02 fit run `bb8ed84d` stuck at `RUNNING` in MLflow (SIGUSR2 requeue did not fire at timeout) | Low      | `GRAPHIDS_FORCE_RESUME=1` on future fit resubmission if clean close needed          |

---

## Submission history

| plan_id                            | scope                                                                                                     | outcome                                         |
| ---------------------------------- | --------------------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| `019e0028-a390`–`aac2`             | initial, all datasets                                                                                     | all failed (Bugs 1+2)                           |
| `019e0052-c622`–`cdb9`             | Bug 1+2 targeted re-runs                                                                                  | partial; Bug 3 surfaced on hcrl_sa RL           |
| `019e00a7-3498`                    | Bug 3 fix; hcrl_sa bandit/dqn/weighted_avg (jids 47359359–47359365)                                       | bandit/dqn clean; weighted_avg inverted (Bug 4) |
| `019e00b8-205f`                    | Bug 4 fix; hcrl_sa weighted_avg (jids 47359413–47359414)                                                  | hcrl_sa final; used in results table            |
| set_01–04 weighted_avg (Bug 4 fix) | set_01 clean; set_02 timeout → test job 47359731; set_03/04 SQLite lock → resubmit jids 47359696–47359699 | all in final results table                      |

## Convo Claude 5.07.26

WeightedAvg — alpha IS a single static scalar. self.weight is one nn.Parameter. torch.sigmoid(self.weight) broadcasts to all samples identically.
The "alpha→1.0" finding here is a clean, meaningful result about a learned scalar.

Bandit and DQN — alpha is per-sample by design. alpha_values = linspace(0, 1, 21) is a lookup table of 21 discrete blend values.
select_action_batch (base.py:228) returns self.alpha_values[actions] where actions has shape [N] — each sample selects its own arm from the 18-dim
feature context. Sample 1 could get alpha=0.2, sample 2 alpha=1.0, sample 3 alpha=0.65, etc. That per-sample contextual selection is the entire
value proposition of the RL approach.

The problem is what the logged avg_alpha=1.000 actually means. Because linspace(0,1,21) has 1.0 as the maximum, an average of exactly 1.000 means
every single sample selected arm 20 — the policy didn't learn differentiated selection, it collapsed to a constant. That's worse than the
weighted_avg finding: weighted_avg learned a scalar that happens to equal 1.0 (informative about VGAE vs GAT), while bandit/DQN lost all per-sample
adaptivity entirely and degenerated to a constant policy.

The empirical note's phrasing "alpha→1.0 everywhere" treats both cases as the same finding. They aren't:

┌──────────────┬───────────────────────────────────────┬───────────────────────────────────────────────────────────────────────────────────┐
│ Model │ Alpha design │ What "avg_alpha=1.0" means │
├──────────────┼───────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────┤
│ weighted_avg │ Single scalar parameter │ Learned value, informative about VGAE vs GAT signal strength │
├──────────────┼───────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────┤
│ bandit / DQN │ Per-sample arm selection from 21 arms │ Policy collapsed — all samples got arm 20, contextual selection failed completely │
└──────────────┴───────────────────────────────────────┴───────────────────────────────────────────────────────────────────────────────────┘

The reward biases described in the note (balance term, majority-class incentive) caused the collapse. But the framing should be: RL methods failed
to learn any contextual policy, not just that they "converged to alpha=1.0."

---

## 2026-05-07 — set_01–04 results inventory (was missing from this writeup)

This writeup focused on the hcrl_sa results table; **the bigger-set runs were
not captured here even though they finished**. They live in MLflow
(`/fs/ess/PAS1266/graphids/mlflow.db`). To avoid reconstructing this table
from scratch every session, use the canonical query tool:

```bash
python scripts/results.py --view fusion                    # all datasets, all variants
python scripts/results.py --view fusion --dataset set_01   # filter
python scripts/results.py --view gat --variant focal       # different model group
python scripts/results.py --list-views                     # see profiles
```

View profiles are config-driven in `configs/result_views.yml`. Adding a new
model group means editing YAML, not Python.

### Most-recent FINISHED test row per (dataset, variant), as of 2026-05-07 13:38

| dataset | variant | AUROC | **MCC** | F1(atk) | recall | precision | plan_id | git_sha |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hcrl_sa | bandit | 0.9999 | 0.0133 | 0.2481 | 1.0000 | 0.1416 | 019e0380 | 43ba3b2 |
| hcrl_sa | dqn | 1.0000 | 0.0491 | 0.2511 | 1.0000 | 0.1436 | 019e0380 | 43ba3b2 |
| hcrl_sa | mlp | 0.8586 | **0.7367** | 0.7376 | 0.5887 | 0.9873 | 019e034a | 1fad328 |
| hcrl_sa | weighted_avg | 0.9291 | 0.6217 | 0.6545 | 0.9434 | 0.5010 | 019e00b8 | 6546750 |
| set_01 | bandit | 0.9992 | 0.0076 | 0.2004 | 1.0000 | 0.1114 | 019e0052 | 427c9af |
| set_01 | dqn | 0.9925 | 0.0302 | 0.2022 | 0.9937 | 0.1126 | 019e0052 | 427c9af |
| **set_01** | **mlp** | **0.9992** | **0.9821** | **0.9841** | **0.9808** | **0.9873** | 019e0028 | 9d8da38 |
| set_01 | weighted_avg | 0.6953 | 0.1825 | 0.2916 | 0.4591 | 0.2137 | 019e00c5 | 6546750 |
| set_02 | bandit | 0.9987 | 0.0397 | 0.5952 | 0.9999 | 0.4237 | 019e0052 | 427c9af |
| set_02 | dqn | 0.9989 | −0.0018 | 0.5943 | 1.0000 | 0.4227 | 019e0028 | 9d8da38 |
| **set_02** | **mlp** | **0.9988** | **0.9735** | **0.9847** | **0.9819** | **0.9875** | 019e0028 | 9d8da38 |
| set_02 | weighted_avg | 0.9956 | 0.9475 | 0.9686 | 0.9443 | 0.9942 | 019e00c5 | 6546750 |
| set_03 | bandit | 0.9990 | 0.1368 | 0.5497 | 0.9997 | 0.3790 | 019e0052 | 427c9af |
| set_03 | dqn | 0.9997 | −0.0032 | 0.5370 | 1.0000 | 0.3671 | 019e0028 | 9d8da38 |
| **set_03** | **mlp** | **0.9994** | **0.9907** | **0.9941** | **0.9941** | **0.9941** | 019e0028 | 9d8da38 |
| set_03 | weighted_avg | 0.9982 | 0.9862 | 0.9912 | 0.9902 | 0.9923 | 019e00c5 | 6546750 |
| set_04 | bandit | 0.9393 | −0.0146 | 0.2208 | 1.0000 | 0.1241 | 019e0052 | 427c9af |
| set_04 | dqn | 0.8291 | 0.0900 | 0.2319 | 1.0000 | 0.1312 | 019e0052 | 427c9af |
| **set_04** | **mlp** | **0.9582** | **0.7534** | **0.7697** | **0.6686** | **0.9067** | 019e0028 | 9d8da38 |
| set_04 | weighted_avg | 0.9380 | 0.6796 | 0.6684 | 0.5077 | 0.9778 | 019e00c5 | 6546750 |

### Findings

1. **MLP is the winner across all datasets.** Set_01–04 MCC are 0.98 / 0.97 / 0.99 / 0.75 —
   the only fusion variant that recovers when GAT alone fails (set_01: GAT≈WAVG MCC=0.18,
   MLP=0.98). On hcrl_sa MLP MCC=0.74 is lower — hcrl_sa is the small dev-set, *not the
   representative case*.
2. **Bandit / DQN show predict-all-attack across every dataset** (recall ≈ 1.0,
   precision tiny, MCC ≈ 0). This is the same pattern as today's hcrl_sa runs and is
   consistent with the saturating `derive_scores` bug — repeatable across all four
   bigger datasets, not a hcrl_sa quirk.
3. **All RL rows above were run with the broken (saturating) `derive_scores`.** The
   patch is `ccd0ab9` (2026-05-07): replaces `clamp(0,1)` with the Möbius transform
   `x/(1+x)`. The right next experiment is bandit/DQN re-runs on set_01–04 with the
   patch — single most valuable RL result outstanding.
4. **Per-attack stratification is `nan` on these rows** because they predate Phase 0.3
   (`auroc_per_attack/{name}` wiring + `attack_type` cache propagation in CACHE_VERSION
   5→6). Any new fusion run on the v6 cache will produce these keys automatically.
5. **weighted_avg numbers don't need re-running** — separate code path, unaffected by
   the `derive_scores` fix. Sets a stable baseline for "GAT-alone-via-α=1.0" performance.

### Implication for the plan

`docs/drafts/fusion-improvement-plan.md` has been updated to reflect that MLP set_01–04
is already in hand. The Phase 5 (MoE+BCE per-sample gating) framing is reinforced by data:
**MLP averaging MCC ≈ 0.92 across set_01–04 is the strongest fusion result on record.**
The remaining open question is whether RL recovers under the patched `derive_scores` —
gating decision for keeping bandit/DQN as a paper section.
