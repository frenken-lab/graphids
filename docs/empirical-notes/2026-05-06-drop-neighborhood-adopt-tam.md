# 2026-05-06 — Drop `neighborhood_decoder`, swap in TAM affinity

**Prior log:** `docs/empirical-notes/2026-05-05-vgae-cublas-overflow-v100.md` — cuBLAS V100 fp32-overflow fix → set_01..04 cross-dataset diagnosis (Q1-Q12) → literature alignment with the GAD literature (Q13).

## Ported from yesterday's Q13b — the literature answer

TAM (Qiao & Pang, *Truncated Affinity Maximization*, NeurIPS 2023) gives a closed-form, training-free answer to the question `neighborhood_decoder` was approximating with a 1791-dim BCE head:

```python
def neighbor_affinity(z, edge_index):
    """Per-node 1 - mean cosine similarity to neighbors. High = anomalous."""
    src, dst = edge_index
    sim = F.cosine_similarity(z[src], z[dst], dim=-1)              # [E]
    per_node = scatter(sim, src, reduce='mean', dim_size=z.size(0))  # [N]
    return 1 - per_node                                             # [N]
```

Per-node, vocab-free, computable on the `z` the encoder already produces. TAM empirically beats reconstruction-based GAD methods by >10 AUROC across 10 datasets per the paper.

The standing `neighborhood_decoder`:
1. Was a workaround for classical Kipf adjacency reconstruction being random under chain-windowed topology (98 deterministic edges, nothing to predict).
2. Inherits vocab dependency — UNK cliff on OOD; learns an arbitrary `embedding(ID_X) → bag-of-neighbor-IDs` lookup with no semantic prior.
3. Was one of the two 1791-dim matmuls saturated in the V100 fp32-overflow at the top of yesterday's log.
4. Produces uniform output when attacks are in-vocab — set_04, the case we most need signal on.

## Decision

Drop `neighborhood_decoder` + `nbr_logits` BCE term + `nbr_weight` hparam. Replace with TAM `neighbor_affinity` computed at inference time, wired into both `_score` and `extract_features`.

Combined with the parallel Q13a-derived plan to drop `mahal` + `kl` from `score`, the final inference score becomes:

```
score(G) = max-σ(recon, recon_max, tam_affinity)
```

Three components, all tied to actual data signal. No vocab dependence in any axis. No "premature" detector layers (`mahal` over an unconstrained latent, `kl` as a proxy for prior fit) cluttering the score.

## Code change manifest

| File | Change |
|---|---|
| `graphids/core/models/autoencoder/vgae.py` | Remove `neighborhood_decoder` Sequential + masking helper. Add `_tam_affinity(z, edge_index)` (~5 LOC, uses `torch_scatter`). Wire into `_score`. Forward 6-tuple → 5-tuple `(cont_out, canid_logits, z, kl_per_node, edge_logits)`. `extract_features` gains an `affinity` per-graph key (additive — same pattern as `spike` in Q12, doesn't break the frozen `[N, 3]` fusion-reward weight). `validation_step` logs `val_node_affinity_{benign,attack}_{p50,p95,p99,max}` alongside the existing `val_node_recon_*`. |
| `graphids/core/losses/autoencoder.py` | Remove `nbr_recon` BCE term + `nbr_weight` hparam from `VGAETaskLoss`. Loss = `recon + α·canid + β·kl + γ·edge_recon`. |
| `graphids/core/losses/build.py` | Remove `nbr_weight` from `_VGAE_LOSS_KEYS`. |
| `graphids/core/losses/distillation.py:181, 184` | 6-tuple → 5-tuple unpack at teacher + student forward. |
| `graphids/core/curriculum.py:66`, `graphids/core/data/preprocessing/curriculum.py:43` | Same unpack adjustment. |
| `tests/core/models/test_vgae.py:54-93` | `test_forward_shapes` expects 5-tuple; `test_gradient_flow` removes `nbr_logits` from required-grads list. Add `test_tam_affinity_shapes` — output shape `[N]`, finite, in `[0, 2]`. |
| Plan modules referencing `nbr_weight` | Drop the kwarg. |

Net delete ≈ 80 LOC (head + loss term + masking helper + unpack sites). Net add ≈ 15 LOC (TAM helper + score wiring + per-node logging). Bonus: one fewer 1791-dim matmul, half of the V100 fp32-overflow surface gone.

Old VGAE checkpoints will not load (missing `neighborhood_decoder` weights, forward arity changes). Per `feedback_no_backward_compat_wrappers.md`, retrain from scratch — same posture as the Q12 changes from yesterday.

## Composition with shipped changes (yesterday's Q12)

- **`recon_max`** — orthogonal: per-node recon spike pattern. TAM is per-node latent-affinity inconsistency. Both summarize per-graph in `_score`.
- **`edge_decoder`** — orthogonal: edge_attr regression for set_04 timing perturbations. TAM doesn't see edge_attr.
- **Per-node histograms (val)** — extended with TAM affinity histograms; lets us verify per-node distribution shape on attack vs benign empirically per dataset.

## Hypotheses for the validation ablation

1. **TAM lifts set_01.** UNK-injected nodes (DoS / fuzzing) will have anomalous `z` vs in-vocab neighbors → low cosine → high TAM score. Should reproduce or exceed `neighborhood_decoder`'s contribution to set_01's gap=0.20.
2. **TAM near-zero on set_04.** In-vocab attacks have neighbor-similar `z`. Set_04 leverage is `edge_decoder`, not TAM.
3. **Components identifiable.** A 4-row ablation (each of `{recon_max, edge_decoder, tam_affinity}` on/off, plus baseline) localizes the gap-lift contribution per dataset.

If TAM shows neither (1) nor (2), revert. The principled fallback is full MuSE `phi(G)` extraction + downstream OC-SVM (yesterday's Q13a) — bigger change.

## Handoff

1. Implement the manifest above (single PR).
2. Render two VGAE rows under the same plan: new arch + baseline (`neighborhood_decoder` re-enabled via flag for the comparison row only).
3. Submit on Pitzer with EarlyStopping Candidate A (`val_discrimination_gap, patience=30, mode='max'` — yesterday's decision).
4. Compare `val_discrimination_gap`, per-component score axes, and per-node TAM-affinity histograms across set_01..04.
5. If hypothesis (1) holds and (2) holds → ship as the new VGAE baseline. If both fail → fall back to full phi(G) refactor.

---

## Q14 — Does dropping `neighborhood_decoder` change preprocessing (vocab, scaler, cache)?

**No.** Preprocessing artifacts are bit-identical. Three signal paths to verify:

1. **Vocab.** The vocab is built by `BaseGraphSource.build()` from the raw CSV inputs and partitioned at `{LAKE_ROOT}/cache/v{PREPROCESSING_VERSION}/{dataset}/voc_{scope}/` (cite: `CLAUDE.md` "Lake-root data layout"; `.claude/rules/data-layout.md` "Cache partitioning"). The encoder's `id_encoder` (embedding lookup over vocab) is unchanged; only the *output-side head that used vocab as labels* is removed. Removing a label space doesn't alter the vocabulary that defines it.
2. **Scaler.** Continuous-feature scaler stats live in the same cache. The encoder still consumes scaled `x` identically. The TAM swap touches `z` and `edge_index` — neither depends on scaler params.
3. **Cache digest.** MLflow's `Dataset` entity is keyed by cache digest (cite: `.claude/rules/data-layout.md` "Store ownership" table — `dataset.digest` is the search filter for dataset identity). Same cache files → same digest → same MLflow `Dataset` entity. Old and new runs will share the same dataset id when filtered.

**`PREPROCESSING_VERSION` does NOT bump.** That counter is reserved for changes to the cache-producing code (vocab construction rules, edge-construction rules, feature columns). Architecture changes downstream of the cache don't touch it. (Mechanism: `voc_{scope}` partitioning was added 2026-05-04 as the only recent reason to bump — see `CLAUDE.md` "Lake-root data layout".)

**Caveat — masking.** `canid_classifier` still requires masked-node targets, so the masking augmentation in `dataset.__getitem__` stays. If a future change drops *both* heads, masking becomes dead code and can be removed. Not in scope today.

## Q15 — On re-run with the new arch, does MLflow assign a different run id, and does that break cross-comparison?

**Yes, different run id. No, it does not break cross-comparison — that's by design.**

### Mechanism

Each `mlflow.start_run()` call without an explicit `run_id=` argument allocates a fresh `run_id` (cite: MLflow Python API — `mlflow.start_run(run_id: Optional[str] = None)`, https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.start_run). `_mlflow.start_training_run` follows this contract: a fit run is opened in `orchestrate.run_row` before `trainer.fit` (cite: `.claude/rules/config-system.md` "Observability wiring — Lifecycle"). The new arch produces a code change → new git SHA → resume gating routes to a fresh run rather than reusing the prior one (cite: `.claude/rules/config-system.md` "Resume gating: git-SHA change → new run").

### Why fresh run id is actually correct here

Cross-run comparison in this project is **not** done by `run_id`. It's done by the **five identity tags** on each run (cite: `.claude/rules/chassis-invariants.md` Invariant 4 — "Reproduction contract — five MLflow tags"):

```
graphids.plan_id        graphids.plan_module     graphids.plan_args
graphids.git_sha        graphids.row_name
```

For baseline (with `neighborhood_decoder`) vs TAM (without):

| Tag | Baseline run | TAM run | Same? |
|---|---|---|---|
| `plan_id` | sha derived from full plan JSON | … | depends on whether both rows live in the same rendered plan |
| `plan_module` | `ablations.supervised` | `ablations.supervised` | **same** |
| `plan_args` | `{dataset, seed}` | `{dataset, seed}` | **same** |
| `row_name` | e.g. `vgae_baseline_…` | e.g. `vgae_tam_…` | **different** (different variants) |
| `git_sha` | pre-PR SHA | post-PR SHA | **different** |
| `dataset.digest` (Dataset entity) | cache hash | cache hash | **same** (Q14) |

The cross-comparison query path is `_mlflow.build_search_filter(...)` filtering on these tags (cite: `.claude/rules/config-system.md` "Observability wiring — Query API: always `_mlflow.build_search_filter(...)`" and `.claude/rules/data-layout.md` "Key rules" §4 "Query path is MLflow"). The user-facing surface is `gx plans show <plan_id>` (cite: `CLAUDE.md` "Key Commands"), which aggregates rows across `git_sha` for the same plan.

### Concretely

To cross-compare baseline and TAM on set_03/seed_42:
- Render both rows in one plan (different `row_name`, same `plan_args`). One `plan_id` covers both.
- `gx plans show <plan_id>` lists both as separate runs under one plan, with `row_name` distinguishing them.
- `dataset.digest` matches across runs (Q14) → search filter `dataset.digest = '<hash>'` confirms same training data.

If they were rendered as separate plans with separate `plan_id`s, comparison still works via tag filter on `plan_module` + `plan_args` + `row_name` patterns; just less ergonomic.

**The reproduction contract is preserved across the swap.** A reader given the five tags of either run can `git checkout <git_sha> && gx run … --filter <row_name>` to regenerate the exact row JSON (cite: `.claude/rules/chassis-invariants.md` Invariant 4 reproduction snippet).

## Q16 — If we drop `neighborhood_decoder` at train and replace at inference (TAM), why not the same for `canid_classifier`?

**Because no closed-form inference equivalent for the canid task is published, and `canid_classifier`'s removal would be net-loss without one.** The asymmetry is in the literature, not in our code.

### What "swap" actually means for `neighborhood_decoder`

Strictly: we **delete** a training-time auxiliary loss (`nbr_recon` BCE) and **add** an inference-time score (`tam_affinity`) that targets the same property the head was approximating (local neighbor consistency). Note that `nbr_logits` was never in `_score` (cite: yesterday's §Q8 — `score = max-σ(recon, mahal, kl)` does not include canid or nbr); neither head was directly producing the inference signal. Both were latent-shaping auxiliaries. The TAM swap works because:

1. The latent-shaping auxiliary objective (predict neighbor IDs) was a **proxy** for the actual scoring property of interest (neighbor consistency in `z`).
2. TAM (Qiao & Pang, NeurIPS 2023) shows the property is computable directly from `z` without the proxy objective — `1 - mean cos(z_v, z_u)` over neighbors, no training.
3. Empirically TAM dominates reconstruction-based methods by >10 AUROC across 10 datasets (yesterday's Q13b).

So: drop the proxy, compute the property directly. The latent-shaping that the proxy provided is replaced by TAM's loss-free inference, validated empirically by the paper.

### Why `canid_classifier` does not have the same path

The canid task is "predict the ID of this masked node." A closed-form inference equivalent would need to answer "is this node's `z` typical for its ID?" without a trained head. The candidates:

| Approach | Closed-form? | Issue |
|---|---|---|
| Per-ID centroid `‖z_v − μ_id(v)‖` | Requires accumulating per-ID `z` mean over training (extra training-time pass + state). Not closed-form on `z` alone. | Vocabulary-large state; UNK has no centroid; centroids drift if the encoder updates |
| Compare `z_v` to `id_encoder.embed(id_v)` | Cheap | The two live in different spaces — `id_encoder` output is one input to GAT, not a target. No reason for `z` to align with the embedding. |
| TAM-by-ID (cosine within same-ID nodes in window) | Closed-form on `z` | Most windows have ≤1-2 nodes per ID; mean over <2 elements is degenerate |
| GGAD pseudo-anomaly objective (Qiao et al., NeurIPS 2024) | Trained, not closed-form | Different track entirely; relevant for the GAT stage, not VGAE auxiliaries (yesterday's Q13e) |
| Drop entirely, no replacement | n/a | Loses the latent-shaping signal that helps set_01 (vocab-OOD detection) |

There is no published "TAM-for-IDs" with the same training-free property. The literature alignment exercise in yesterday's Q13 surfaced TAM (Q13b) and MuSE (Q13a) but no equivalent for the canid task.

### The principled position for now

`canid_classifier` stays as a training-time auxiliary, weighted at 0.1 (yesterday's "canid_weight decision" section). The reasons:

1. **No closed-form replacement exists.** Following the literature alignment principle, we shouldn't delete a useful auxiliary just because we deleted a different one — they're independent decisions on independent merits.
2. **Set_01 leans on it.** §"Why set_01 separates and set_04 doesn't" (yesterday's log) attributes set_01's gap=0.20 partly to UNK-embedding distance signal that the canid head leverages during training. Removing it without replacement loses set_01 lift.
3. **The cuBLAS argument is no longer load-bearing.** Yesterday's permanent strict-reductions fix (`_ensure_runtime`) eliminated the V100 fp32-overflow path; `canid_classifier`'s 1791-dim matmul is now numerically safe. "It saturated cuBLAS" no longer argues for removal.
4. **MuSE doesn't argue against auxiliaries, only against tuning their weights.** "Tuning alpha/beta loss weights … will not fix this" (yesterday's Q13a) targets *per-dataset* hyperparameter tuning of weights, not the existence of auxiliary terms. Static weights are fine.

### What would change the position

If the validation ablation shows TAM + `recon_max` + `edge_decoder` *fully* recovers the set_01 gap with `canid_classifier` removed (i.e., the per-component contribution is zero or negative), drop it in a follow-up PR. The data answers it. Don't drop preemptively.

---

## Implementation reality (amendment, end of session 2026-05-06)

The plan above said "drop neighborhood, adopt TAM (3 axes)." What actually shipped is broader: **drop the vocab-bag neighborhood, adopt TAM AND restore neighborhood the literature-correct way (GAD-NR), AND add Rayleigh quotient as a 4th axis.** The shifts came from cross-referencing the literature on `neighborhood`-style mechanisms after deletion.

### Final architecture

VGAE forward returns a **6-tuple** (NOT the 5-tuple in the original plan): `(cont_out, canid_logits, nbr_pred, z, kl_per_node, edge_logits)`. Position 2 is the same slot the deleted vocab-bag head occupied; the **semantics** of position 2 changed (vocabulary categorical → continuous neighbor-mean prediction in latent space).

Score axes are **4** (not 3): `max-σ(recon, recon_max, affinity, rq)` — all calibrated z-norm. `mahal`/`KL` no longer enter scoring but are retained in `extract_features['errors'][N, 3]` so fusion's frozen reward weight is preserved.

Loss is **5 components**: `recon + α·canid + β·nbr + γ·kl + δ·edge_recon` — `nbr` came back, with GAD-NR semantics.

### Why neighborhood came back

After deletion, library inventory (pygod, PyOD, PyG `metrics`/`contrib`/`graphgym`) found GAD-NR (Roy et al., WSDM 2024 — arXiv:2306.01951) — the *literature-correct* "neighborhood reconstruction" mechanism. Side-by-side audit of the deleted code vs GAD-NR's `KL_neighbor_loss`:

| Axis | Deleted vocab-bag | GAD-NR (now) |
|---|---|---|
| Output dim | `[N, num_ids]` (1791) | `[N, latent_dim]` (64) |
| Output space | Categorical vocab | Continuous latent |
| Mechanism | NCE on neighbor-ID bag | Closed-form Gaussian KL on neighbor latent distribution |
| Vocab dependence | Yes → UNK cliff | No → vocab-free |
| 1791-dim matmul (V100 fp32 overflow) | Yes | No |
| Output for in-vocab attacks | Uniform (set_04 failure) | Distributional (works) |

GAD-NR addresses ALL FOUR critiques the original plan cited against the deleted code. Cross-checked with the upstream paper repo (`Graph-COM/GAD-NR/GAD-NR_inj_cora.ipynb`) and pygod's port — confirmed the deleted code was a different mechanism, not a botched GAD-NR.

### Sign-convention bug in upstream KL math

Both pygod (`pygod.nn.functional.KL_neighbor_loss`) and the original GAD-NR notebook use `log(det Σ_predictions / det Σ_targets)` — the **inverse** of the textbook KL form. Empirically this produced `last_nbr ≈ −5.96` in our smoke run, which a proper KL divergence cannot do. Minimizing the upstream form would reward the encoder for collapsing prediction variance (the opposite of distribution matching).

We use `log(det Σ_targets / det Σ_predictions)` — the standard convention. After the flip: `KL(x, x) == 0`, `last_nbr ≈ 1.97` on a fresh forward, monotone in the right direction. Documented in the docstring of `kl_neighbor_loss` for paper-readers who will compare to upstream.

### Code organization

`tam_affinity` and `rayleigh_quotient` lifted out of VGAE into `graphids/core/models/_score_primitives.py` as **stateless free functions**. Direct response to the "VGAE is a monolith" concern from earlier in the session — the model imports primitives instead of housing them. New primitives slot in the same file without architectural surgery on VGAE itself; if a third primitive arrives, the lift to a registry pattern is a 30-line refactor.

`kl_neighbor_loss` lives in `graphids/core/losses/autoencoder.py` next to `VGAETaskLoss` (where the deleted `neighborhood_loss_negsampled` lived). Type-fit: it's a loss function, not a stateless primitive.

### Final code-change manifest (replaces the table at line 40)

| File | Net effect |
|---|---|
| `graphids/core/models/autoencoder/vgae.py` | `_tam_affinity` static method removed (lifted out). `_SCORE_COMPONENTS = ("recon", "recon_max", "affinity", "rq")` (4-axis). Forward returns 6-tuple with `nbr_pred` at position 2 (same slot, new semantics — `[N, latent_dim]` GAD-NR mean prediction). `_score`, `extract_features`, `_fit_score_norm`, `validation_step` updated for 4 score axes + nbr_pred + RQ logging. `train_nbr` logged in `training_step`. |
| `graphids/core/models/_score_primitives.py` | New file. Stateless free functions: `tam_affinity(z, edge_index)` and `rayleigh_quotient(x, edge_index, batch=None)`. |
| `graphids/core/losses/autoencoder.py` | `VGAETaskLoss.__init__` adds `nbr_weight=0.1` (default). Forward unpacks 6-tuple, computes empirical neighbor-mean target inline (`scatter` with clamped count for isolated nodes), calls `kl_neighbor_loss(nbr_pred, nbr_targets)`. New free function `kl_neighbor_loss` with full GAD-NR docstring + sign-fix. |
| `graphids/core/losses/build.py` | `nbr_weight` re-added to `_VGAE_LOSS_KEYS`. |
| `graphids/core/losses/distillation.py` | 5-tuple → 6-tuple unpack at student + teacher forward. |
| `graphids/core/curriculum.py`, `graphids/core/data/preprocessing/curriculum.py` | Same unpack adjustment. |
| `graphids/core/budget.py` | NaN-debug `names` tuple gets `nbr_pred` back. |
| `graphids/orchestrate.py` | Stale comment cleanup (no behavior change). |
| `tests/core/models/test_vgae.py` | `test_forward_shapes` and `test_gradient_flow` expect 6-tuple; `test_tam_affinity_shape` rewired to free function; new `test_rayleigh_quotient_per_graph`. |

Net delta in `vgae.py` is roughly +30 LOC vs HEAD — most of the file's growth predates this session (`recon_max`, per-node histogram, `edge_decoder`, `_per_graph_masked_recon` expansion). My contribution net is small; primitives lifted out, GAD-NR + RQ wired in.

### Hypothesis update for validation ablation

Add a 5th component to the on/off matrix: **rq**. Hypothesized contributions:

1. **RQ lifts set_01.** OOD payload values from injected ECUs → high feature-smoothness violation on raw `x`. RQ is closed-form, no learned weights — should produce a clean signal where TAM (on `z`) might smooth across attacks.
2. **GAD-NR `nbr_loss` shapes z.** Encourages encoder to produce z's where each node's z predicts its neighbors' mean. If TAM is the weakest axis at test, `nbr_loss` is what would lift it (training-time signal that pulls z's into a "predictable-from-self" structure).
3. **RQ near-zero on set_04.** In-vocab attacks may not perturb feature smoothness much (they're using known IDs with known features). Set_04 leverage stays `edge_decoder`.
4. **Cardinal short fit (jids 47344547 / 47344548) is a wiring smoke**, not a paper-quality result. Watch for: (a) NaN on the new GAD-NR Gaussian KL term — `nbr_weight=0.1` is the default, drop to `0.01` if first epoch NaNs; (b) RQ scale before calibration — should be O(1) but unbounded above for highly non-smooth windows.

### Submitted

`gx run ablations.unsupervised --dataset hcrl_sa --seed 42 -o /tmp/vgae_cardinal.json`
`gx plans submit -p /tmp/vgae_cardinal.json -C cardinal -L short --filter 'vgae*'`

- `vgae` (fit): jid 47344547
- `vgae-test` (test, afterok=47344547): jid 47344548
- `plan_id`: `019dfe48-1e53-72a8-9cf0-a6cb2d1909fa`
