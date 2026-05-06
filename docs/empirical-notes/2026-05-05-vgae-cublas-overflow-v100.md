# VGAE budget probe — fp32 overflow in `nn.Linear` on V100

**Date:** 2026-05-06 (UTC ~01:50)
**Hardware:** OSC Pitzer V100 16GB
**Build:** PyTorch 2.8 + CUDA 12.6, model state `git_sha=05f8a59860bb`
**Plan:** `ablations.supervised --filter 'vgae*' -d set_03 -s 42` (VGAE on can-train-and-test-v1.5 set_03, `label_filter='benign'`)

## TL;DR

`nn.Linear` produces fp32-overflow (`absmax=3.40282347e+38` = `torch.finfo(torch.float32).max`) and NaN/Inf outputs when applied to a finite latent `z` (absmax=11.68) at matmul shape `[300793, 64] @ [64, 1791]` on V100. There is no algebraic path from inputs of that magnitude to fp32 max — this is a **cuBLAS-level numerical defect at this specific (M, N, K) on Volta**, not a model bug.

## Reproducible diagnostic

After patching the budget probe to:
- isolate RNG via `torch.random.fork_rng()` + fixed seed `GRAPHIDS_PROBE_SEED=20260506`
- snapshot CPU + CUDA RNG state immediately before the failing fwd+bwd
- replay the failing forward via `torch.set_rng_state` / `torch.cuda.set_rng_state`

we get bit-deterministic NaN every run. Sample `nan_debug_intermediates` log line:

```json
{
  "tag": "sanity",
  "V": 300793,
  "E": 652974,
  "bad_params": [],
  "z_finite": true,           "z_absmax": 11.68,
  "cont_out_finite": true,    "cont_out_absmax": 6.97,
  "kl_per_node_finite": true, "kl_per_node_absmax": 1.11,
  "canid_logits_has_nan": true, "canid_logits_has_inf": true,
    "canid_logits_absmax": NaN,
    "canid_logits_shape": [300793, 1791],
  "nbr_logits_has_nan": true,   "nbr_logits_has_inf": true,
    "nbr_logits_absmax": 3.40282347e+38,
    "nbr_logits_shape": [300793, 1791]
}
```

`bad_params: []` rules out weight corruption (`isfinite(p).all()` over every named parameter). `z` is healthy. The single `Linear(64→1791)` (`canid_classifier`) and the 3-layer MLP (`neighborhood_decoder`) on the same z both produce non-finite outputs — pointing at the matmul kernel itself rather than any model-side numerics.

## Why this is *not* algebraic

- `nn.Linear` default init: Kaiming uniform with bound `sqrt(1/fan_in)` for fan_in=64 ⇒ `|W| ≤ 0.125`.
- Per-output element of `Linear(64, 1791)`: `out[i,j] = Σ_k W[j,k]·z[i,k] + b[j]`.
- Worst case |out| ≤ 64 × 0.125 × 11.68 ≈ **93**. Reaching fp32 max (3.40e38) requires `>10^36×` amplification across a matmul that algebraically caps at ~93.
- The 3-layer MLP nominally amplifies further but is bounded by ReLU/Dropout(p=0.1) and same Kaiming weights — saturation impossible from real-valued accumulation.

## Hypothesis: cuBLAS GEMM numerical defect at Volta

V100 cuBLAS chooses an algorithm based on (M, N, K, dtype, layout). Some heuristics select kernels that use reduced-precision intermediate reductions even for fp32 inputs (e.g., split-K reductions accumulating to fp16 buffers, or pre-Ampere code paths with looser numerical guarantees). For `[300793, 64] @ [64, 1791]` there is at least one path that returns saturated/NaN for inputs in our range.

This is observable because:
- Smaller batches (the candidate probes at V=57) succeed with the same model and weights.
- The output saturates at exactly fp32 max — characteristic of accumulator overflow in a low-precision intermediate.
- `bad_params: []` and `z` finite at the entrance to the matmul rule out everything upstream.

## Workarounds

In order of preference:

1. **Cap the probe / pack budget so this matmul shape never appears.** With `max_num ≈ 64K`, the GEMM becomes `[64K, 64] @ [64, 1791]` — well inside cuBLAS-safe territory. Other ablations don't hit this because they have full label distribution and pack denser smaller batches naturally; VGAE's `label_filter='benign'` produces unusually large packed batches (105K benign graphs available; entire benign pool packs into a few large bins).
2. **Force strict-precision reductions:** set
   ```python
   torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
   torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
   ```
   before `pl.Trainer.fit`. Forces cuBLAS to keep accumulators in fp32. Worth one job to confirm; cleaner than capping budget.
3. **Move VGAE training to Cardinal H100 / Ascend A100.** Ampere/Hopper cuBLAS kernels are different code paths and typically don't exhibit this. (Requires logging into the right cluster — sbatch can only target the local cluster on OSC, see `reference/osc_gpu_clusters` memory.)

## Probe instrumentation in place (v5)

`graphids/core/budget.py` now:

- Wraps the probe in `torch.random.fork_rng()` so probe RNG consumption doesn't pollute the training-time draws — required for reproducibility AND for not silently shifting batch sampling order during fit.
- Seeds with `GRAPHIDS_PROBE_SEED` (default `20260506`) — same draw → same outcome across runs. Bisecting flaky NaN by varying this env var is the official debug workflow.
- Snapshots CPU + CUDA RNG state right before each fwd+bwd. On `ValueError` from `loss_fn` non-finite check, restores state and replays through `_dump_intermediates` for an exact-replay forward (no longer rolls fresh randomness — the prior `_dump_intermediates` was running a different draw than the one that failed, which is why earlier dumps showed contradictory `_finite=True` + `_absmax=NaN`).
- Uses `isnan(t).any()` / `isinf(t).any()` (scalar reductions) instead of `.sum()` (allocates `[N]→int64` ~4 GB for the 540M-element `canid_logits`, OOM'd the dump itself on V100).

## Reproducing the failure manually

```bash
source .env && source .venv/bin/activate
gx run ablations.supervised -d set_03 -s 42 --filter 'vgae*' -o /tmp/vgae.json
gx plans submit --plan /tmp/vgae.json -C cardinal   # routed to local cluster (pitzer)
# Fit fails ~3 min in at sanity probe with deterministic NaN. Stdout/stderr:
#   /fs/ess/PAS1266/graphids/dev/rf15/set_03/ablations/unsupervised/vgae/seed_42/.parsl_scripts/
```

To bisect non-failing seeds:
```bash
GRAPHIDS_PROBE_SEED=42 gx plans submit --plan /tmp/vgae.json -C cardinal
```

## Open

- Whether this affects VGAE training itself (post-probe, inside the fit loop) once the probe passes is **unconfirmed**. The previous "lucky" run trained to completion; the model checkpoint exists. May or may not have hit the same matmul issue silently and produced subtly bad gradients. Workaround #2 (now applied — see Update below) eliminates the matmul path entirely so this is moot going forward.
- No issue filed with PyTorch / cuBLAS yet. Should reproduce on a minimal `[300793, 64] @ [64, 1791]` matmul with random input/weight tensors before reporting.

## Update — 2026-05-06 ~02:01 UTC — workaround #2 confirmed

In `graphids/orchestrate.py::_ensure_runtime`:
```python
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
```

Resubmitted the same plan with the same `GRAPHIDS_PROBE_SEED=20260506` (so the
matmul shape, RNG draw, and inputs are bit-identical to the failing run). The
sanity probe now passes:

```
budget_probed: sanity_V=300793 sanity_peak_mb=6209 (no nan_debug, no nan_replay)
```

Bit-deterministic same draw → no NaN with strict reductions. The defect lives
in the cuBLAS code path that picks reduced-precision intermediate accumulation
for fp32 GEMM at this shape on Volta. Disabling it falls back to a strictly-fp32
accumulation kernel that handles the shape correctly.

Full fit ran to completion: jid 47317870 COMPLETED in 11:56, `last-v3.ckpt`
(3.9 MB) saved, test row 47317871 auto-chained on afterok and started cleanly.

**Permanent change.** Strict reductions stay on in `_ensure_runtime` for all
runs — they only marginally affect throughput on Volta, never affect Ampere or
Hopper meaningfully (those architectures' default kernels already accumulate
in fp32), and make Volta numerically uniform with the newer clusters. Net cost
is negligible; net benefit is one fewer source of intermittent NaN.

Workaround #1 (cap budget to ~64K) and #3 (move to Cardinal) not pursued.

## Update — 2026-05-06 ~02:15 UTC — fit results + diagnosis + improvements

### Results (plan_id `019dfb01-f45c-7dbe-b272-c3b2ffa236be`, set_03/seed_42)

| metric | first (ep 12) | last (ep 178) | max | source |
|---|---|---|---|---|
| `val_discrimination_ratio` | 1.111 | 1.151 | **1.166** | benign-vs-attack score ratio (>1 = better) |
| `val_loss` | 1.225 | 0.911 | — | reconstruction + canid CE + nbr BCE + KL |

178 epochs (out of `max_epochs=600`). `EarlyStopping(monitor='val_discrimination_ratio', patience=100, mode='max')` fired — best metric stopped improving for 100 consecutive epochs.

Test phase running at time of writing; AUROCs not yet appended.

### Diagnosis — why ratio plateaus at ~1.17

`val_discrimination_ratio = mean_score_attack / mean_score_benign` where score = mahalanobis-style distance in latent space. Random would give ratio ≈ 1.0. **1.17 is essentially noise above chance.** The model is barely separating benign from attack on val.

Three plausible drivers, in order of likelihood:

1. **Benign pool starvation.** `cache_metadata.json::train.attack_balance.benign = 1099` raw samples → ~105K windowed benign graphs. But benign-labeled samples in set_03 come from a single attack-free recording; **graph-level diversity is low**. VGAE overfits the benign manifold within a few dozen epochs, then `val_discrimination_ratio` plateaus because val benign and val attack both end up in the same dense reconstruction-error region. Compare to hcrl_sa where benign messages span multiple recordings and CAN-ID populations.

2. **set_03 attack design = "stealth" attacks.** Per the can-train-and-test-v1.5 split design, set_03 includes `interval`, `rpm`, `speed`, `standstill` — timing/value perturbations on legitimate IDs, not novel injection. VGAE built on `(node_id, x_continuous)` reconstruction will struggle: the IDs are all in-vocab, the continuous-feature distribution shift is subtle. Set_01 (DoS, fuzzing) would exercise a stronger signal.

3. **Defaults not tuned for set_03 graph sizes.** Train graphs have V_max=57, E_max=98 (per metadata). With `latent_dim=64` and `hidden_dim=64`, the encoder compresses to a vector roughly the same size as the input — minimal bottleneck, weak inductive pressure. On hcrl_sa with denser graphs the same dim choice is fine; here it's effectively an autoencoder with no compression.

## Update — 2026-05-06 ~02:50 UTC — VGAE diagnosis across set_01..04

All four can-train-and-test-v1.5 datasets fitted (set_01/04 needed `CUDA_LAUNCH_BLOCKING=1` to dodge an intermittent V100 vectorized-gather race; same code/data succeeded under sync execution).

### Per-dataset training trajectory

| ds      | epochs | train_recon ↓     | train_canid ↓     | train_nbr ↓       | train_kl ↑        | val_loss_benign | val_loss_attack | gap (att-ben) | ratio (att/ben) |
|---------|-------:|-------------------|-------------------|-------------------|-------------------|----------------:|----------------:|--------------:|----------------:|
| set_01  |    351 | 0.73 → **0.20**   | 2.95 → **1.47**   | 0.57 → **0.26**   | 0.92 → 1.74       |          0.704  |          0.894  |    **0.200**  |        **1.327** |
| set_02  |    103 | 0.70 → **0.19**   | 3.45 → **2.33**   | 0.14 → **0.08**   | 1.26 → 2.23       |          0.735  |          0.819  |     0.076     |        1.130     |
| set_03  |    179 | 0.78 → **0.22**   | 4.16 → **3.45**   | 0.19 → **0.05**   | 1.08 → 2.04       |          0.868  |          0.987  |     0.131     |        1.166     |
| set_04  |    281 | 0.74 → **0.19**   | 3.48 → **2.19**   | 0.13 → **0.07**   | 1.17 → 1.91       |          0.754  |          0.773  |    **0.019**  |        **1.057** |

### Q1: Is there actual separation between benign and attack?

**Yes for set_01, marginal for set_02/set_03, near-zero for set_04.** The gap (mean reconstruction loss on val_attack minus val_benign) is the unambiguous signal: set_01 holds 0.20 (28% relative); set_04 holds 0.02 (2.5% relative — essentially noise). The discrimination ratio reformulation amplifies this in the wrong direction (small denominator inflates ratio for tiny gaps), making the metric look more healthy than it is.

### Q2: Is the model actually training?

**Yes.** Every reconstruction-side component drops monotonically: `train_recon` -73% (set_01), `train_canid` -50%, `train_nbr` -55%, `val_loss` -40%. Optimization is not stuck. **Learning rate is fine** (default 1e-3 Adam) — losses descend cleanly without the oscillation that high-LR causes or the plateau that low-LR causes. This is not an optimizer problem.

### Q3: Then why is `val_discrimination_ratio` not separating?

The model learns to **reconstruct everything in the data distribution**, not just benign. set_04 val_attack vs val_benign reconstruction loss is 0.77 vs 0.75 — VGAE has learned a generic CAN-frame autoencoder that handles attacks nearly as well as benign because:

1. **kl_weight=0.01 is too low.** `train_kl` *increases* by 60-90% over training (1.08 → 2.04 on set_03) — the encoder is pulling the latent distribution further from N(0,I) to maximize reconstruction quality. With `kl_weight=0.01`, the prior penalty is dominated by reconstruction loss; the encoder uses high-variance latents as needed and pays almost nothing for it. Result: the latent space has **no structural prior**, so attack samples land in similarly-dense regions as benign and reconstruct comparably well.

2. **The objective doesn't match the detection task.** Reconstruction-MSE is symmetric in features — it minimizes error on whatever distribution the data has. If attack-frame features are in-distribution (timing perturbations on legitimate CAN IDs, as in set_04's `interval`/`rpm`/`speed`/`standstill`), reconstruction error is just as low for attacks as benign. The CAN-ID prediction head (canid_classifier) *is* discriminative when attacks inject novel IDs (set_01's DoS/fuzzing) — but with `canid_weight=0.1` it's a minority contributor.

3. **`label_filter='benign'` filtering may be inert.** Despite the filter, the model is still optimizing recon over all benign training graphs, so it learns the dominant CAN-frame distribution. Attack frames sit on the same low-dimensional manifold for set_04. This is the fundamental limitation of reconstruction-based anomaly detection on this data: when attacks share the input distribution with benign, the autoencoder generalizes to them.

### Why set_01 separates and set_04 doesn't

set_01's vocabulary is **53 IDs** vs set_04's **2049**. set_01 attacks include DoS and fuzzing — both inject CAN IDs not present in train benign, getting mapped to UNK (index 0) by the cache vocab. The encoder's `id_encoder` embedding for UNK plus the masked-`mask_id` slot are different from in-vocabulary embeddings → real per-graph distribution shift → recon error differential. set_04 attacks are stealth/timing perturbations on **in-vocabulary** IDs → no embedding shift → recon error is essentially the same as benign.

This explains the dataset ordering exactly: set_01 (53 IDs, OOD-by-vocab attacks) >> set_03 (1791 IDs, mixed) ≈ set_02 (2049 IDs, mixed) > set_04 (2049 IDs, all in-vocab attacks).

### Two downstream questions, answered

**Q4: What about increasing `kl_weight` to force a tighter latent prior?**
**A:** Marginal benefit, possibly negative. Tightening KL would push z toward N(0,I), which collapses representational capacity (toward "posterior collapse"). The encoder would reconstruct everything *worse* — not just attacks. Better to keep kl_weight low and either (a) shrink `latent_dim` (currently 64, almost no compression for 35-dim input × small graphs — try 8 or 16) which forces information bottleneck without distorting the loss, or (b) raise `canid_weight` from 0.1 → 1.0 so the discriminative head matters. (b) is the higher-leverage change for set_01-class attacks; both fail for set_04 because no objective re-weighting fixes a model that genuinely can't see attack-vs-benign in the data.

**Q5: Why does `val_discrimination_ratio` (=mean_attack/mean_benign) keep monotonically improving on set_01 (1.20 → 1.33) but flatten on set_03 (1.11 → 1.16)?**
**A:** set_01 has a much smaller benign val pool (`num_arb_ids=53` ⇒ low CAN-frame diversity) so `mean_benign` reconstruction loss drops faster as the model overfits the limited benign manifold; meanwhile `mean_attack` (vocab-shifted) stays higher because OOD CAN-IDs aren't in the training distribution at all. The ratio numerator stays high while the denominator drops → the ratio grows. set_03 attacks include a larger fraction of in-vocabulary ones, so `mean_attack` drops alongside `mean_benign` — both move together, ratio is roughly stationary near 1.16. **Caution:** set_01's 1.33 isn't necessarily a "better model" — it's partly a more-OOD attack distribution. Test-phase AUROC is the comparable metric across datasets; ratio is dataset-dependent.

## Update — 2026-05-06 ~03:10 UTC — what's actually getting reconstructed, and where the signal lives

### Q6: Training improves — what specifically is it getting better at?

Three reconstruction-side targets, in order of relative drop on set_01:

| component | first | last | drop | what it scores |
|---|---|---|---|---|
| `train_recon` | 0.73 | 0.20 | **−73%** | per-feature MSE on continuous `x` |
| `train_nbr` | 0.57 | 0.26 | **−55%** | masked-node neighbor BCE |
| `train_canid` | 2.95 | 1.47 | **−50%** | masked-node CAN-ID cross-entropy |
| `train_kl` | 0.92 | **1.74** | **+90%** | KL of `q(z\|x)` from `N(0,I)` |

The model gets steadily better at reconstructing continuous features, predicting masked CAN IDs, and predicting which nodes are neighbors of a masked node. **It pays for that by letting the latent distribution drift away from its prior** (`train_kl` doubles), because `kl_weight=0.01` makes the prior penalty negligible vs the recon terms. So the encoder uses high-variance latents — fine for reconstruction, useless for OOD detection (which assumes normal-density priors).

### Q7: How distinct are attack vs benign windows? Graph stats first.

Per cache_metadata.json:

| dataset | train N (mean / max) | test_05 (suppress) | test_06 (masquerade) | other test sets |
|---|---|---|---|---|
| set_01 | 24.7 / 40 | 30.3 / 49 | 28.5 / 46 | 27-28 / 32-45 |
| set_02 | 33.2 / 48 | 36.1 / 51 | 33.8 / 50 | 31-35 / 43-52 |
| set_03 | 37.6 / 57 | **48.6 / 68** | **48.6 / 74** | 31-43 / 42-71 |
| set_04 | 31.9 / 46 | 35.8 / 43 | 34.3 / 44 | 32-36 / 43-52 |

**Raw edge count is exactly 98** because window_size=100 → 98 inter-frame chain edges; this is a windowing artifact, not a property of the data. **Unique (src, dst) transition pairs per graph DO vary** — that's the real edge-side signal:

| dataset | train benign μ ± σ | train attack μ ± σ | gap |
|---|---|---|---|
| set_01 | 65.9 ± 5.1 | 71.6 ± 5.0 | +5.7 |
| set_02 | 78.1 ± 6.9 | 78.3 ± 4.3 | +0.3 |
| set_03 | 75.5 ± 4.7 | 78.7 ± 5.4 | +3.2 |
| set_04 | 79.0 ± 4.8 | 82.4 ± 4.6 | +3.4 |

For test_01 (known_vehicle_known_attack), set_01 and set_04 **flip sign** — attack windows have fewer unique transitions than benign (DoS-class attacks flood 1-2 IDs → many duplicate transitions in the chain → fewer distinct edges). The signal exists but sits within ~1σ of class variance, so it's a weak per-graph discriminator. Not zero, as I claimed earlier.

Implications:
- set_04 attack windows have node-count `33-36` vs train benign `31.9` — **3-13% relative shift**, well within natural benign variance. Naive reconstruction can't separate.
- set_03 attack types `suppress`/`masquerade` push node count to 48.6 (29% above train) — those are the windows where set_03 will get the most signal at test time. Per-attack AUROC (already logged at test phase) should rank these high.
- set_01's tiny vocab (53 IDs) means **attack windows that introduce DoS/fuzzing inject brand-new IDs** which inflate node count modestly AND tag the new IDs as UNK (vocab index 0). That's why set_01 separates: not because of node count alone, but because of the embedding-table OOD signal layered on top.

### Q8: Is reconstruction a per-node mean, a per-graph aggregate, or per-feature distribution?

Found out by reading `losses/autoencoder.py:104` and `models/autoencoder/vgae.py:289-296,344-363`:

| where | formula | reduction |
|---|---|---|
| **train loss** (`train_recon`) | `F.mse_loss(cont_out, batch.x)` | **global mean** over nodes × features. Per-node and per-graph signal both **collapsed**. |
| **val loss** (`val_loss`, `val_loss_benign/attack`) | `_per_graph_masked_recon` — `(cont - x).pow(2).mean(dim=-1)` per node, masked-sum per graph, divided by mask count | **per-graph mean** over masked nodes. Class means via `recon[mask_y].mean()`. |
| **test score** (`score`) | `max-σ(recon, mahal, kl)` in z-norm space | **per-graph** scalar. The per-node `recon_per_node` is computed but **summed away** before the score. |

Critical findings:
1. **There is no edge-attribute reconstruction loss.** `nbr_logits` is a *neighbor identity* prediction (BCE over candidate next-IDs), not edge feature reconstruction. Edge features (`batch.edge_attr`) only feed the encoder via GAT attention — the model is never penalized for failing to reconstruct timing/payload patterns encoded on edges.
2. **Per-node spike vs spread is invisible.** `recon_per_node` exists in `_per_graph_masked_recon`'s tensor — but it's immediately summed/meaned per graph. A window with one nuclear-anomalous node produces the same `recon` score as a window with all-slightly-anomalous nodes if their per-node sums match.
3. We have **no logged statistic** that would tell us whether VGAE spikes on attack-injected nodes specifically or smears the error across the window. Need per-node error histograms or per-graph max/range of `recon_per_node` to know.

### Two downstream questions, answered

**Q9: If per-node error is reduced to a per-graph mean, are we losing the spike-vs-spread signal? Should we add a per-node max as a score component?**
**A:** Yes, real signal loss. CAN-bus attacks (especially DoS, fuzzing, masquerade) typically inject 1-N malicious frames into an otherwise-normal 100-frame window — the textbook spike pattern. The per-graph mean `recon` is dominated by the 95+ benign frames; the few attack frames' high error is averaged away. Adding `recon_max = scatter(recon_per_node, batch.batch, reduce='max')` as a fourth dimension in `score = max-σ(recon, mahal, kl, recon_max)` (one-line change in `vgae._score`) would capture spike anomalies the current score discards. Likely the single highest-leverage code change for set_04 if attacks-as-spikes is the right model. For "spread" attacks (suppress: missing frames change distribution evenly) the mean is the right reduction; per-node max stays low. So adding it doesn't hurt the spread case — it strictly augments.

**Q10: What does the constant `edge_count=98` invariant imply about VGAE's ability to use edge information?**
**A:** Less than I claimed. The 98-edge figure is a *raw chain count* fixed by windowing (window=100 → 98 inter-frame edges), not a property of the data. **Distinct (src, dst) transitions per graph vary 65-82** across datasets and *do* differ by class (~3-6 edge gap, ~1σ relative to class variance) — see the table above. So topology is not signal-free; it's a weak per-graph discriminator that the GAT encoder can in principle exploit via attention over real (vs duplicate) transitions.

What's still true: **there is no loss term that supervises edge feature reconstruction.** VGAE consumes `edge_attr` only via GAT attention; if timing-perturbation attacks (set_04 `interval`/`rpm`/`speed`/`standstill`) live in edge_attr deltas, the model has no objective to learn that. The `nbr_logits` head predicts node-identity neighbor classes — *related* to edge-pattern signal but not edge_attr reconstruction. `kl_weight=0.01` further means the latent isn't constrained. So the diagnosis stands directionally: edge_attr signal is under-supervised, just not "invisible". An edge-attr reconstruction head (Linear `[2*latent_dim → edge_feat_dim]` over endpoint pairs, MSE'd against `batch.edge_attr`) would directly target the timing signal — ~30 LOC.

## Next steps — VGAE on can-train-and-test-v1.5 set_01..04

Drawn from the Q1-Q10 diagnosis above. Ranked by expected per-set_04-gap leverage (set_04 is the hardest case and the one with no current signal). All are config or local code changes; none require touching the chassis.

### Code changes (model/loss)

1. **[x] Add `recon_max` to the test score** — `_score` in `vgae.py` currently aggregates per-node reconstruction error to a per-graph mean, throwing away the spike pattern that DoS / fuzzing / masquerade attacks produce (1-N malicious frames in a 100-frame window). One line: `recon_max = scatter(recon_per_node, batch.batch, reduce='max')`; include alongside `recon`/`mahal`/`kl` in the `max-σ` score. Strictly augments — never hurts spread attacks (where per-node max stays low and the mean dominates), recovers spike attacks the current score discards. **Highest expected leverage for set_04 if attacks-as-spikes is the right model.** **Shipped (Q12, item 1).**

2. **[x] Add an edge-attribute reconstruction head** — currently zero loss term supervises edge-feature recovery, even though edge_attr carries the only timing signal (set_04 attack types are timing perturbations on in-vocab IDs). `nn.Linear(2*latent_dim → edge_feat_dim)` over endpoint pairs `(z[edge_index[0]], z[edge_index[1]])`, MSE'd against `batch.edge_attr`, ~30 LOC in `vgae._build()` + `_forward_tensors()` + a new `edge_weight` term in `VGAETaskLoss`. Directly targets set_04's "VGAE has nothing to fit on" diagnosis. Worth one ablation row. **Shipped (Q12, item 3).**

3. **Bottleneck `latent_dim` 64 → 8 or 16** — train graphs have V≈30 nodes and 35-dim continuous features, so `latent_dim=64` is no compression at all. Forces the encoder to discard non-discriminative information. Single hparam override in `ablations.supervised`'s vgae spec. No model code change. Cheapest to try first. **Pending — render as an ablation row alongside the new VGAE arch.**

4. ~~**Raise `canid_weight` 0.1 → 1.0 for vocab-OOD-attack datasets (set_01)** — the canid head IS discriminative when attacks inject novel CAN IDs (DoS / fuzzing). Currently weighted at 10% of recon, so its gradient signal is buried. Test on set_01 first since that's where it's actually doing work; expect set_04 to be unaffected (no vocab-OOD attacks).~~ **Decided against — see "canid_weight decision" below.** Per-dataset weight tuning conflates model identity with dataset identity (every other model hparam — `latent_dim`, `kl_weight`, `nbr_weight` — stays fixed across datasets for the same reason). Component-magnitude arithmetic also says 0.1 is already in-balance: `train_canid` ranges 1.47–4.16 across set_01..04, so at weight 0.1 it contributes 0.15–0.42 vs `train_recon`'s 0.19–0.78 — already comparable. Raising to 1.0 makes canid 3–10× recon and swamps every other gradient. Set_04's gap=0.019 is not a canid_weight problem; it's an objective-mismatch problem (Q3, in-vocab attacks invisible to the canid head regardless of its weight) — addressed by item #2 (edge-attr head, now shipped).

### Eval / monitoring changes

5. **[x] Drop `val_discrimination_ratio` from EarlyStopping monitor.** It's a derived metric (mean_attack / mean_benign) that inflates wildly on small-vocab datasets where the denominator collapses (set_01: 1.33 looks great but is 0.20 gap with 0.70 benign baseline). Switch to `val_discrimination_gap` (additive, dataset-comparable) or to a proper test-time AUROC computed on a held-out dev split. **Decided: ship Candidate A (`val_discrimination_gap`, patience=30, mode='max'). See "EarlyStopping decision" below.**

6. **[x] Tighten `EarlyStopping.patience` 100 → 30** — set_01 fit 351 epochs to plateau at gap=0.20; the gap maxed around epoch 80 and never recovered. Saves ~70 epochs of compute per ablation row globally. **Decided alongside #5: patience=30 ships with the new monitor.**

7. **[x] Log per-node reconstruction-error histograms during val** — answers "spike vs spread" empirically per dataset. ~10 LOC in `validation_step`: log min/max/p50/p95/p99 of `recon_per_node` for benign and attack subsets. Cheap, lets us decide whether action #1 is the right fix from data instead of priors. **Shipped (Q12, item 2 — `val_node_recon_{p50,p95,p99,max}_{benign,attack}` per epoch).**

### Reporting decisions

8. **Per-attack AUROC is the comparable metric across datasets, not `val_discrimination_ratio`.** It's already logged at test phase via `_log_per_attack_auroc`. Build the empirical results table from `test/{set}/auroc_per_attack/{attack}` once tests finish — that's what goes in the paper.

9. **Set_04 should not anchor the VGAE narrative.** The diagnosis (in-vocab attacks + no edge-attr recon loss + no spike-aware score) explains why VGAE produces gap=0.02 on set_04. If actions #1-#2 don't move it, the honest paper reporting is: VGAE works on vocab-OOD attack regimes (set_01-class), is borderline on mixed regimes (set_02/03), and fails on pure timing-perturbation regimes (set_04). The supervised GAT path is the right tool for the latter; report it there.

## Update — 2026-05-06 ~04:00 UTC — compression audit, scoring extended, edge head added

### Q11: Is the encoder actually compressing? Walk through graph size vs model size.

**Per-graph data volume.** From `cache_metadata.json` and a direct read of one cached `.pt` for set_03:

| dim                                             | value | source |
|---|---|---|
| `node_count.mean` (train)                       | 37.6  | `cache_metadata.json::splits.train.graph_stats.node_count.mean` |
| `node_count.max` (train)                        | 57    | same |
| continuous feature dim per node (`in_channels`) | 35    | `g.x.shape[1]` from `data_test_test_02_*.pt` for set_03 |
| edge feature dim per edge (`edge_dim`)          | 11    | `g.edge_attr.shape[1]` from same file |
| id-embedding output dim                         | 32    | `embedding_dim=32` default in `vgae.py:58` |
| edge count (chain windowing)                    | 98    | windowing artifact, all sets |

**Encoder input width per node.** `InputEncoder.out_dim = id_encoder.out_dim + cont_dim` (`_conv.py:51`).
With `proj_dim=0` (default), `cont_dim = in_channels = 35`. So `gat_in_dim = 32 + 35 = 67`.

**Latent width.** `_SCALES["small"] = {"latent_dim": 64, "hidden_dims": [64]}` (`vgae.py:45`).
`z_mean = nn.Linear(latent_in_dim, 64)`, `z_logvar = nn.Linear(latent_in_dim, 64)`.

**The compression ratio is 64/67 ≈ 0.955.** The latent vector per node is essentially the same size as the input vector per node. There is no information bottleneck — by capacity, the encoder can preserve the input verbatim.

By comparison, principled bottleneck choices for this regime:

| latent_dim | input/latent ratio | rationale |
|---|---|---|
| 64 (current) | 1.05× | no compression — autoencoder has no reason to discard non-discriminative signal |
| 16 | 4.2× | matches typical AE bottleneck depth; aggressive |
| 8 | 8.4× | hard bottleneck — forces the encoder to keep only the most predictive directions |

The `train_kl` rise documented in Q6 (1.08 → 2.04 on set_03) is consistent with this: the encoder is using the available 64 dims of latent variance to carry information that doesn't need to be carried, because the recon objective is dominant and the prior penalty (`kl_weight=0.01`) is essentially free.

**Implication.** Bottlenecking `latent_dim` to 8 or 16 is the cheapest change to test (single hparam override), and it directly attacks the diagnosis without distorting the loss. It does NOT replace the edge-recon head (Q10) or the spike-aware score (Q9) — those address different signal-loss paths. They compose.

### Q12: What was actually shipped this turn?

Three model-side changes in vgae.py / losses/autoencoder.py:

1. **`recon_max` added to scoring.** `_per_graph_masked_recon` now optionally returns `(mean, max, per_node)` instead of just mean. `_score()` returns `(recon, recon_max, mahal, kl, z)` 5-tuple. `score()` does max-σ over all four scalar components. New calibration buffers `score_recon_max_{mean,std}` registered alongside `score_{recon,mahal,kl}_*`; `_fit_score_norm` now drives all four uniformly via the `_SCORE_COMPONENTS` constant. Strictly augments — for spread attacks `recon_max ~ recon` and the existing path drives the score; for spike attacks (one nuclear-bad node in a 100-frame window) `recon_max` is large while `recon` is smeared by the surrounding benign frames.

2. **Per-node MSE tensor exposed.** `_per_graph_masked_recon(..., return_components=True)` returns the raw `recon_per_node` tensor before reduction. `validation_step` now logs `val_node_recon_{p50,p95,p99,max}_{benign,attack}` per validation epoch — answers the spike-vs-spread question empirically per dataset (action #7 from the prior next-steps list). Cost: 5 percentile scalars per batch per class, no extra forward.

3. **Edge-attribute reconstruction head.** New `self.edge_decoder = nn.Sequential(Linear(2*latent, latent), ReLU, Dropout, Linear(latent, edge_dim))` over endpoint pairs `[z[edge_index[0]] || z[edge_index[1]]]`. `VGAETaskLoss` gains an `edge_weight: float = 0.1` hparam and an `edge_recon = F.mse_loss(edge_logits, batch.edge_attr)` term. Built only when `_uses_edge_attr=True` and `_edge_dim>0` — otherwise the head would learn from random noise (edge_attr never enters the latent for conv types that ignore it). Loss handles both cases uniformly: when the head is absent, `edge_logits=None` and the term is a zero scalar.

The forward signature is now a 6-tuple `(cont_out, canid_logits, nbr_logits, z, kl_per_node, edge_logits)`. Six unpack sites updated:
- `losses/autoencoder.py:102` (`VGAETaskLoss.forward`)
- `losses/distillation.py:181, 184` (`FeatureDistillation.forward` + teacher unpack)
- `core/curriculum.py:66`, `core/data/preprocessing/curriculum.py:43` (curriculum scoring)
- `models/autoencoder/vgae.py:321, 359` (`validation_step`, `_score`)
- `tests/core/models/test_vgae.py:54-93` (forward-shape + gradient-flow contracts)

`extract_features` errors stack frozen at `[N, 3]` to keep the fusion-reward 3-vector weight (`reward.py:55`) compatible; a new `spike` key (per-graph `recon_max`) is added additively for fusion to opt into without breaking the existing tuned weight.

Old VGAE checkpoints will not load — new `score_recon_max_*` buffers + new `edge_decoder` weights aren't in the saved state_dict. Per the no-backward-compat rule (`feedback_no_backward_compat_wrappers.md`), retrain from scratch. The cuBLAS strict-reductions fix and the new architecture compose; no other code path needs adjustment.

## canid_weight decision — kept static at 0.1

**Position.** `canid_weight=0.1` stays static across all datasets. No per-dataset override.

**Justification (component-magnitude argument).** `train_canid` ranges 1.47–4.16 across set_01..04 (table in §"Per-dataset training trajectory"). `train_recon` ranges 0.19–0.78. With `canid_weight=0.1` the canid term contributes 0.15–0.42 to total loss vs recon's 0.19–0.78 — already comparable in magnitude. Raising to 1.0 makes canid 3–10× larger than recon, swamping every other gradient. This is loss-balancing arithmetic, not a dataset-specific decision.

**Why per-dataset weighting fails the principle.** A loss-component weight is a model hyperparameter. Tuning it per dataset means selecting (model, ds) jointly on signal that ought to come from ds alone. Set_01's separation ratio of 1.33 (§Q1) and gap of 0.20 (§"Per-dataset training trajectory") would be partially explained by tuning that doesn't transfer — the published ablation table loses meaning. Same reason `latent_dim`, `kl_weight`, and `nbr_weight` stay fixed: model identity is one decision, dataset is another.

**What the data actually says.** Set_04's gap=0.019 is not a canid_weight problem. It's diagnosed (§Q3, §"Why set_01 separates and set_04 doesn't") as: all set_04 attacks are in-vocab, so the canid head has no OOD signal to extract regardless of its weight. A 10× weight on a head that sees identical input distribution for benign and attack windows produces 10× the same uninformative gradient. The fix for set_04 is the edge-attr reconstruction head (now shipped) — that targets the timing-perturbation signal the canid head can't see. Action #4 in the prior list was misframed as a config knob; it's actually two separate questions (model balance — static — and which signal to supervise — architecture).

**If calibration is later warranted**, the principled path is: log per-component gradient norm during training, set the static weight to equalize gradient contribution across (recon, canid, nbr, kl, edge) at a chosen reference point, ship one value for all datasets. Not selected per-dataset on val gap.

## EarlyStopping decision — Candidate A (ship), Candidate B (secondary log)

Current: `monitor='val_discrimination_ratio', patience=100, mode='max'`. Both flaws documented: ratio inflates on small-vocab sets where `mean_benign` collapses (§Q5), and patience=100 burned ~270 epochs on set_01 after the gap maxed near epoch 80 (§"Tighten EarlyStopping").

### Candidate A — ship: `monitor='val_discrimination_gap', patience=30, mode='max'`

**Justification.** The gap is additive (`mean_attack − mean_benign`), so it's dataset-comparable in a way the ratio isn't. From §"Per-dataset training trajectory" the gaps cluster (0.20, 0.08, 0.13, 0.02) on a single scale; same monitor key at the same threshold means the same thing across datasets. Patience=30 is calibrated against §"Tighten EarlyStopping": set_01's gap maxed near epoch 80 of a 351-epoch fit and never recovered, so 30 covers the realistic noise-vs-real-improvement window with ~10× margin saved over the prior 100. The new `val_recon_max_gap` (logged from this turn's changes) gives a parallel signal once a few runs are in; if it correlates with `val_discrimination_gap`, this monitor is fine on its own.

**Risk to flag.** The gap can be negative early in training (untrained encoder may reconstruct attacks marginally better than benign on small batches) — `mode='max'` handles this correctly but the first few epochs may show monitor=−0.05; that's not a failure mode, just startup.

### Candidate B — secondary unmonitored log: `val_loss_benign, patience=30, mode='min'`

**Justification.** Optimization quality on benign val is a monotone proxy for "the autoencoder is fitting the benign manifold." Unlike gap or ratio, it does not depend on attack composition at all — pure fit signal. From §"Per-dataset training trajectory" (val_loss_benign column: 0.704 / 0.735 / 0.868 / 0.754) — consistent low-variance across datasets, same scale.

**Trade-off vs A.** Decoupled from the detection task — stops when the AE fits benign, not when discrimination peaks. For datasets where discrimination peaks before fit converges (plausible for set_01 where the gap maxed at epoch ~80 while `val_loss_benign` may still have been descending), B keeps training past the discrimination optimum. A captures the actual research target; B is more stable.

### Recommendation

**Ship A.** Keep B as a secondary unmonitored log to detect the case where A's gap monitor is overfitting on a dataset-specific quirk (e.g., set_01's tiny benign val pool collapsing the gap denominator under small-batch noise) — divergence between A's "best" and B's "best" is a flag to inspect that fit by hand.

## Handoff — next row to render and submit

Untested on real data — login node only. Next: render an ablation row pairing the new VGAE arch (recon_max + per-node histograms + edge-attr head + EarlyStopping Candidate A) against the old config under the same plan, submit on Pitzer, compare `val_recon_max_gap` and `val_discrimination_gap` across set_01..04.

## Q13: Literature alignment — what GAD research already names our problems

Reading: `thoughts.md` (browser research session). Mapping our Q1-Q12 diagnoses to published graph-anomaly-detection work.

### Q13a — MuSE (NeurIPS 2024) names "reconstruction flip"

Kim et al., *Rethinking Reconstruction-based Graph-Level Anomaly Detection*. Theoretically + empirically (10 datasets): reconstruction-error mean is **not monotonic** in anomaly status — anomalous graphs with structural patterns distinct from training can exhibit *equal or lower* mean recon error than normal. **This is the §Q3 set_04 gap=0.019 result with a name.**

MuSE's remedy is a multifaceted summary `phi(G) = [mean, std, max, skew, p95, top-k]` over per-node recon errors, fed to a one-class scorer (isolation forest / OC-SVM trained on benign train phi only). Beats 14 methods. Direct alignment with Q12 shipped:
- `recon_max` in `_score` ← one component of phi(G)
- Per-node histograms in `validation_step` ← phi(G) feature material

The principled MuSE form is the full vector + a non-learned downstream scorer. We currently fold `recon_max` into max-σ; full phi(G) is a candidate next-step refactor.

**Affirms two prior decisions:**
- `canid_weight=0.1` static — MuSE: "Tuning alpha/beta loss weights … will not fix this."
- Drop `val_discrimination_ratio` from EarlyStopping — ratio inflation on small-vocab sets is the reconstruction-flip *symptom*, not model-quality signal.

### Q13b — TAM (NeurIPS 2023) closed-form answer to "who are my neighbors?"

Qiao & Pang. One-class homophily: normal nodes have strong neighbor affinity in latent space; abnormal weak. Per-node score:

```python
score_v = 1 - mean_{u in N(v)} cos(z_v, z_u)
```

Three lines on the `z` the encoder already produces. **This is what `neighborhood_decoder` (1791-dim BCE classifier predicting neighbor IDs) is approximating with a vocab-dependent lookup-table objective.** Empirically dominates reconstruction-based GAD by >10 AUROC across 10 datasets.

`neighborhood_decoder` was added as a workaround when classical Kipf adjacency reconstruction was random (chain-windowed graphs have constant 98-edge topology — nothing to predict). The workaround inherited vocab dependency (UNK cliff on OOD), and was the **other** 1791-dim matmul saturated in the V100 fp32-overflow at the top of this doc. **Action implied: drop the head, replace with TAM affinity.** Executed in tomorrow's log (2026-05-06).

### Q13c — RQGNN (ICLR 2024) Rayleigh quotient as closed-form spectral feature

Dong et al. `x^T L x / x^T x` of node features w.r.t. graph Laplacian discriminates anomalous graphs without learning. CAN-bus mapping: normal windows have correlated ECU signals (low quotient); attack windows inject foreign values (high quotient). Add as a phi(G) feature, zero training. Lower priority than Q13b.

### Q13d — UniGAD (NeurIPS 2024) k-hop subgraph reframing

Zhang et al. Score k-hop ego-networks rather than full graphs; aggregate. Directly addresses §Q3 dilution (one anomalous frame in 100-frame window averaged away). Stretch — graph-construction rework, not near-term.

### Q13e — GGAD (NeurIPS 2024) pseudo-anomaly generation for GAT

Qiao et al. Synthesize pseudo-anomalies by perturbing benign features into tail of marginal distribution; use as labeled negatives. Different track — relevant when revisiting GAT-stage class imbalance.

### Drops (from thoughts.md "What to Drop")

- Normalizing flows on latent — requires clean latent that VGAE-with-aggregate-loss won't produce.
- Temporal delta on `z` — confounded with vehicle state absent conditioning.
- Graph SMOTE — no rigorous version; GGAD replaces.

### Action queue (deferred to 2026-05-06 log)

| Action | Source | Status |
|---|---|---|
| Drop `neighborhood_decoder`, swap in TAM affinity | Q13b | → 2026-05-06 |
| Drop `mahal` + `kl` from `score` | Q13a + own analysis | → 2026-05-06 |
| phi(G) full vector + OC-SVM downstream | Q13a | pending after TAM |
| Rayleigh quotient feature | Q13c | low priority |
| UniGAD subgraph reframing | Q13d | stretch |
| GGAD on GAT stage | Q13e | separate track |

### Q13: literature alignment

Direct overlap with what we already shipped:
  - MuSE (NeurIPS 2024) is the formal name for the "reconstruction flip" problem we diagnosed in §Q3 (set_04 attacks reconstruct as well as benign — and §Q8 on per-graph mean collapsing the spike signal). The MuSE remedy is
  exactly action #1 generalized: phi(G) = [mean, std, max, skew, p95, top-k] over per-node reconstruction errors. We've shipped recon_max (one component) and per-node p50/p95/p99/max histogram logging — those tensors are the raw
   material for phi(G). Next-step extension would be feeding phi(G) into a one-class scorer (isolation forest / OC-SVM) rather than max-σ.
  - MuSE's framing reinforces the canid_weight = 0.1 decision: "Tuning alpha/beta loss weights … will not fix this. The root is in how reconstruction errors are summarized." Same conclusion the component-magnitude argument
  reached.

  Highest-leverage new candidates not in the current next-steps list:
  1. MuSE-style phi(G) + downstream one-class scorer — embarrassingly simple on top of what's already shipped; SOTA across 10 datasets per the paper. Strict augmentation of recon_max.
  2. Rayleigh quotient x^T L x / x^T x — closed-form, zero training, drop in alongside phi(G). Targets exactly the "in-vocab timing perturbation" set_04 case where the attacker disrupts feature smoothness over the graph.
  3. TAM-style local-affinity score — per-node 1 − mean cosine(h_v, h_u in N(v)), gives node-level interpretability for free. Plausibly the right objective for CAN where injected ECUs disrupt neighbor consistency.
  4. GGAD pseudo-anomaly generation — addresses the GAT class-imbalance problem directly (synthesize attack-like nodes from tail distributions). Different track from the VGAE path.
  5. UniGAD k-hop subgraph reframing — reduces the 1-in-100 dilution to scoring 5–10-node subgraphs. Stretch goal; would involve real architectural rework.

  What thoughts.md says to drop (worth flagging because earlier diagnosis hinted at them): normalizing flows on latent (premature without clean latent space), temporal delta on z (confounded with vehicle state), graph-domain
  SMOTE (no rigorous version — GGAD replaces it).
