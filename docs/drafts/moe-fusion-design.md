# MoE+BCE per-sample gated fusion — design

> Companion to `docs/drafts/fusion-improvement-plan.md` §Phase 5. That
> file owns the *when* and *why-now*; this file owns the *what* and
> *which variant*. Every architectural claim cites a source — internal
> file:line for graphids facts, external URL for literature claims.
> No claim is asserted from training memory.

## 1. Hypothesis

Per-sample gating over the existing 18-dim fusion feature vector should
beat a single dense MLP on **either** MCC **or**
`auroc_per_attack/<subtype>` (binding subtype on hcrl_sa is `fuzzing`).

The hypothesis is grounded in two empirical facts already on disk:

1. **Supervised over the full feature vector beats score-fusion's
   two-scalar blend.** Phase 0.2 hcrl_sa MLP run:
   `AUROC=0.859, MCC=0.737`; weighted_avg comparator: `MCC=0.622`
   (`docs/drafts/fusion-improvement-plan.md:21-23, 29`). So "use all
   features" already pays. The remaining question is whether different
   *samples* benefit from different *combinations* of those features.
2. **Per-attack AUROC is highly asymmetric.** Same hcrl_sa MLP run:
   `auroc/dos=0.999, auroc/fuzzing=0.725`
   (`docs/drafts/fusion-improvement-plan.md:23`). DOS is essentially
   solved; fuzzing carries the residual error. A model that can route
   "DOS-shaped" samples through one expert and "fuzzing-shaped" samples
   through another has, in principle, capacity the MLP lacks — the MLP
   has one decision boundary for both regimes simultaneously.

If MoE matches MLP within noise, the extra gating mechanism didn't buy
capability and we keep MLP. The acceptance criterion is in
`docs/drafts/fusion-improvement-plan.md:457-460`.

## 2. Variant survey — four real designs

Per [Wikipedia: Mixture of experts](https://en.wikipedia.org/wiki/Mixture_of_experts)
and [Hugging Face: Mixture of Experts Explained](https://huggingface.co/blog/moe),
the design space has two orthogonal axes — **router timing** (parallel
with experts vs. selecting before them) and **gating density** (soft
weighted sum vs. sparse top-k vs. hard top-1). A third axis —
**expert symmetry** (architecture and input) — interacts with both.

| # | Design | Router | Experts run | Diff'able | Reference |
|---|---|---|---|---|---|
| A | **Dense soft-gated** | softmax over K | all K, in parallel | end-to-end | Jacobs & Jordan 1991 (per Wikipedia §History) |
| B | Sparse top-k | softmax → top-k | only k of K | via straight-through / noise | Shazeer et al. 2017 (per HF blog) |
| C | Hard top-1 + load-balance aux | argmax | only 1 of K | needs ST estimator + aux loss | Switch Transformer 2021 (per HF blog) |
| D | Asymmetric expert input | any of A/B/C | per-expert feature subset | depends on router | non-canonical; not in surveyed sources |

### 2.1 What each source says about router timing

- **Dense (A) — original formulation:**
  Per Wikipedia, the canonical formula is
  `f(x) = Σᵢ w(x)ᵢ · fᵢ(x)` — gate weights and expert outputs combine
  in a single weighted sum, *all* experts evaluated. Wikipedia describes
  Jacobs & Jordan's update rule: *"the weighting function is changed to
  increase the weight on all experts that performed above average, and
  decrease the weight on all experts that performed below average."*
  Specialization is **emergent** from gradient flow through the gate;
  no expert is hand-assigned a region.
- **Sparse (B) — Shazeer et al. 2017:**
  Per Wikipedia, *"they achieve sparsity by a weighted sum of only the
  top-k experts, instead of the weighted sum of all of them."* Per the
  HF blog, the gating function `G(x) = Softmax(x · W_g)` is computed
  *first*, then *"determines which experts receive each token before
  their computation."* This is the "router first, experts second"
  framing.
- **Hard (C) — Switch Transformer 2021:**
  Per HF blog: *"Switch Transformers uses a simplified single-expert
  strategy."* One expert is selected per token via argmax routing.
  Requires a load-balancing auxiliary loss (HF blog: load balance is
  added to prevent gate collapse onto a single expert).

### 2.2 Why sparse routing exists

Per the HF MoE blog, the four advantages of sparse routing are:
- *"The router computation is reduced"*
- *"The batch size of each expert can be at least halved"*
- *"Communication costs are reduced"*
- *"Quality is preserved"*

All four are **conditional-compute scaling arguments** — the goal is
to run a 1T-param model at the FLOPs of a 100B-param model. Sparse
routing is unequivocally the modern transformer-MoE default, but for a
reason that does not apply to small-K shallow MoEs.

## 3. Decision: variant **A** (dense soft-gated, symmetric same-input)

### 3.1 Why dense (A) over sparse (B/C)

Sparse routing's stated benefits (§2.2) are conditional-compute savings
at scale. For our problem:

- **K=3, expert size = `[18→64→32→1]` MLP.** Total expert FLOPs are
  negligible vs. the upstream VGAE+GAT extract that already ran offline
  and is cached at `cache/v6/{dataset}/voc_*/`
  (`graphids/CLAUDE.md:96-100`). There is no compute to save.
- **Sparse top-1 over 3 experts ≡ a hard-gated classifier-per-regime.**
  That loses the soft per-sample blending which is the actual
  hypothesis (§1) — different attack subtypes prefer different
  anomaly/GAT *combinations*, and the boundary between regimes is
  unlikely to be crisp. Hard routing answers a different question
  ("can we partition samples by regime?"), not the one we want
  ("does soft blending help?").
- **Switch-style hard routing (C) requires a load-balance aux loss**
  (HF blog) plus a routing-temperature schedule. Adds two
  hyperparameters and a known failure mode (gate collapse) for
  marginal interpretability gain at K=3.

Variant A directly tests "does per-sample blending beat single-MLP"
with one loss term and no auxiliary tuning.

### 3.2 Why symmetric experts, same input

Both Wikipedia and the HF blog describe canonical MoE as **identical
expert architecture, full input to every expert**. Per HF blog on
Switch Transformer: *"each expert in the FFN layer of the Switch
Transformer is a single FFN with the same architecture."*
Specialization is **not engineered** — it falls out of how gradients
flow through the gate during training.

For graphids the alternative would be **variant D** — assign each
expert a feature subset (e.g., expert 1 sees only `("vgae", *)` leaves,
expert 2 sees only `("gat", *)`, expert 3 sees joint statistics). The
TensorDict structure makes this trivial because tuple keys already
namespace the features
(`graphids/core/models/fusion/weighted_avg.py:33-35`).

We reject D for v0:

1. **It's non-canonical** — neither Wikipedia nor HF blog describes
   asymmetric-input as the standard MoE pattern. We have no prior to
   cite for it.
2. **It encodes our hypothesis as architecture.** D presupposes the
   correct decomposition is VGAE-vs-GAT-vs-joint. If the actual
   right decomposition is by attack subtype (DOS-shaped vs.
   fuzzing-shaped), we've handcuffed the model. A symmetric design
   lets the gate discover whichever decomposition pays.
3. **Diagnosing failure conflates two causes.** If D loses to MLP, did
   mixture not help, or was the partition wrong? A's failure mode is
   unambiguous.
4. **18-dim is too small to subset cleanly.** Splitting an 18-vector
   into ~6-dim per-namespace slices means each expert sees fewer
   features than it has hidden units. The "savings" story doesn't
   apply at this scale.
5. **Symmetry breaks naturally.** Each `nn.Linear` initializes from
   its own random sample; gradients diverge from epoch 1. Engineered
   asymmetry is unnecessary to get specialization — only a learned
   routable signal is.

If A's gate-entropy diagnostic (§5) shows specialization fails to
emerge, **then** D becomes the principled escalation — but the
diagnostic gates the choice, not the hypothesis.

## 4. Architecture spec

```
Input:  td (TensorDict)  →  flatten_features(td) →  x ∈ R^{N×18}

Experts (K=3, identical):
  hᵢ : R¹⁸ → R¹     [Linear(18,64), ReLU, Dropout(0.2),
                     Linear(64,32),  ReLU, Dropout(0.2),
                     Linear(32, 1)]
  expert_score_i = sigmoid(hᵢ(x))   ∈ R^{N×1}

Gate (router):
  g : R¹⁸ → R³      [Linear(18,32), ReLU, Linear(32, 3)]
  w(x) = softmax(g(x))             ∈ R^{N×3}, rows sum to 1

Mixed prediction:
  s(x) = Σᵢ wᵢ(x) · expert_score_i(x)
  s(x) ∈ (0, 1) by clamp + softmax convex combination
  L = BCE(s(x), y)
```

- **Loss:** `nn.functional.binary_cross_entropy` on the clamped mix —
  same as `MLPFusionModule._supervised_loss` via the inherited base
  class path (`graphids/core/models/fusion/base.py:113-116`).
- **No auxiliary loss in v0.** Load-balance aux is the principled
  add-on if gate entropy collapses; we don't pre-add it because (a)
  HF blog adds it specifically because hard routing collapses, and
  (b) collecting the diagnostic *first* tells us whether we even need
  it.
- **`automatic_optimization = True`** — Lightning runs the supervised
  path in `FusionModuleBase` (`base.py:118-126`) which calls
  `_supervised_loss` → `forward_scores`. No RL hooks, no replay
  buffer, no `_store_init_kwargs` quirks beyond the standard
  `_ModelBase` pattern (`graphids/core/models/base.py:75-95`).

### Default hyperparameters

| Param | Value | Rationale |
|---|---|---|
| `state_dim` | 18 | Set by `flatten_features` over current cache v6 (`graphids/plan/plans/ablations/fusion.py:33-35`). |
| `num_experts` | 3 | Per Phase 5 framing in `fusion-improvement-plan.md:447`. |
| `expert_hidden` | `(64, 32)` | Matches `MLPFusionModule` (`graphids/core/models/fusion/mlp.py:21`) — same per-expert capacity as the baseline. |
| `gate_hidden` | `(32,)` | Smaller than experts; gate decides routing, not classification. |
| `lr` | `1e-3` | Matches `MLPFusionModule` (`graphids/core/models/fusion/mlp.py:23`). |
| `decision_threshold` | `0.5` | Matches `FusionModuleBase` default (`graphids/core/models/fusion/base.py:83`). |

## 5. Diagnostics — how we know if A worked

Logged every step via `self.log` in `training_step` / `validation_step`
(stored on instance attribute by `forward_scores`, read after the
super-class step):

- **`{phase}/gate_entropy`** = `-Σ wᵢ log(wᵢ)`, batch-averaged.
  - `→ log(K) ≈ 1.099` (uniform): gate isn't routing on anything.
  - `→ 0` (one-hot): collapsed to a single expert; if MCC also
    matches MLP, the mixture buys nothing (one expert is doing the
    MLP's job).
  - In between with non-trivial variance across samples: routing
    *something*; check expert_usage to see what.
- **`{phase}/expert_usage_i`** = `mean(wᵢ)` per expert. Detects dead
  experts (one usage `→ 0`).
- **`{phase}/expert_disagreement`** = `var(expert_scores, dim=K).mean()`.
  If ~0, all experts predict the same thing and gating is meaningless;
  if non-trivial, experts have diverged and the gate is actually
  selecting from real alternatives. (Optional v0; cheap.)

Existing test-phase plumbing already gives us `auroc_per_attack/<name>`
via `FusionModuleBase.test_step` → `_record_test_batch`
(`graphids/core/models/fusion/base.py:145-155`); no MoE-specific
test-side wiring needed.

## 6. Acceptance + escalation

Per `docs/drafts/fusion-improvement-plan.md:457-460`:

> MoE must beat MLP on either calibration (MCC) OR per-attack-type
> AUROC (`auroc_per_attack/fuzzing` is the binding subtype on
> hcrl_sa). If MoE matches MLP exactly, the gating mechanism added
> complexity without buying capability — keep MLP and put the
> implementation effort into Phase 3/4 features instead.

### Escalation paths if A loses

| Diagnostic at the loss | Read | Next move |
|---|---|---|
| `gate_entropy → log(K)` (uniform) | gate isn't routing — no signal in features it can use to discriminate regimes | escalate to **D** (asymmetric input by namespace) — inject the prior the gate failed to learn |
| `gate_entropy → 0` + `expert_usage` collapses to one | gate found a single expert sufficient — mixture is redundant | keep MLP; close Phase 5 |
| `gate_entropy` healthy, `expert_disagreement → 0` | experts are duplicates — capacity wasted | drop K to 2; if still no win, close Phase 5 |
| `gate_entropy` healthy, MCC ≈ MLP, fuzzing AUROC < MLP | mixture preserves dense capacity but doesn't specialize on fuzzing | escalate to **C** (hard top-1 + load-balance aux) — force commitment |

## 7. File-touch inventory (v0)

| Step | File | What |
|---|---|---|
| 1 (now) | `graphids/core/models/fusion/moe.py` | new — `MoEFusionModule` leaf |
| 2 | `graphids/core/models/fusion/__init__.py` | export `MoEFusionModule` |
| 3 | `graphids/plan/primitives.py` | add `MOE_FUSION = "graphids.core.models.fusion.moe.MoEFusionModule"` |
| 4 | `graphids/plan/__init__.py` | re-export `MOE_FUSION` if other primitives do |
| 5 | `graphids/plan/plans/ablations/fusion.py` | add `moe = fuse("moe", spec(MOE_FUSION, state_dim=_state_dim))` row + its `.fit("moe")` / `.test("moe")` entries |
| 6 | smoke on hcrl_sa | render → `gx submit --row-name moe_fit/moe_test` (gated on `derive_scores` fix Step 1, per handoff) |

Steps 2–6 are deferred until step 1's leaf compiles + imports cleanly.

## 8. Sources

External (literature):
- [Mixture of experts — Wikipedia](https://en.wikipedia.org/wiki/Mixture_of_experts)
- [Mixture of Experts Explained — Hugging Face blog](https://huggingface.co/blog/moe)
- Jacobs, Jordan, Nowlan, Hinton (1991), *Adaptive Mixtures of Local
  Experts*, Neural Computation 3(1):79–87 (cited via Wikipedia, not
  fetched directly).
- Shazeer et al. (2017), *Outrageously Large Neural Networks: The
  Sparsely-Gated Mixture-of-Experts Layer*, arXiv:1701.06538 (cited via
  Wikipedia + HF blog, not fetched directly).
- Fedus, Zoph, Shazeer (2021), *Switch Transformers: Scaling to
  Trillion Parameter Models with Simple and Efficient Sparsity*
  (cited via HF blog, not fetched directly).

Internal (graphids):
- `docs/drafts/fusion-improvement-plan.md:21-35` — Phase 0.2/0.3 results
- `docs/drafts/fusion-improvement-plan.md:432-466` — Phase 5 framing
- `graphids/core/models/fusion/mlp.py` — supervised baseline
- `graphids/core/models/fusion/weighted_avg.py` — score-fusion comparator
- `graphids/core/models/fusion/base.py:73-156` — `FusionModuleBase`
  contract
- `graphids/core/models/base.py:75-95` — `_store_init_kwargs` /
  hparam mirroring
- `graphids/plan/plans/ablations/fusion.py:33-35` — current state_dim=18
