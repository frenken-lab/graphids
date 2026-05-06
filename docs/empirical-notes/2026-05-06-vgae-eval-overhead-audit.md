# 2026-05-06 — VGAE evaluation overhead audit (new metrics post-TAM/RQ wiring)

**Prior log:** `docs/empirical-notes/2026-05-06-drop-neighborhood-adopt-tam.md`

## Trigger

After wiring TAM affinity, Rayleigh quotient, per-node percentile histograms, and
`recon_max` into `validation_step` and `_score`, a cost audit was requested before
running on large datasets (`set_02+`).

---

## Cost inventory

### 1. `recon_max` scatter-max — negligible

`_per_graph_masked_recon(..., return_components=True)` adds one extra
`scatter(masked_err, batch_idx, reduce="max")` over the existing sum scatter.
O(N) on-device, fused with the already-present sum scatter. No concern at any scale.

### 2. `tam_affinity` + `rayleigh_quotient` in `validation_step` — negligible

Both are O(E) or O(E × d) scatter kernels:
- `tam_affinity`: `cosine_similarity(z[src], z[dst])` + `scatter(sim, src, reduce="mean")` — one fused kernel per edge.
- `rayleigh_quotient`: `(x[src] - x[dst]).pow(2).sum(dim=-1)` + two scatters (numerator, denominator) — operates on raw `batch.x`, no encode pass.

Not a concern at any realistic batch size.

### 3. `torch.quantile` × 4-8 per validation batch — low-medium on `set_02+`

`validation_step` (`vgae.py:427-443`) calls `torch.quantile` up to 8 times per batch:
- p50/p95/p99 over masked benign nodes → recon histogram (3 calls)
- p50/p95/p99 over masked attack nodes → recon histogram (3 calls, if attacks present)
- Same pattern over all-node affinity per class (2 more)

`torch.quantile` internally sorts — O(N log N). For `hcrl_sa` validation batches
(a few thousand masked nodes per batch) this is sub-millisecond. For `set_02+` val
batches with >100k nodes, each sort over the all-node affinity subset could contribute
a few ms. Total per batch: probably <20ms on GPU even at large scale. The `mask.any()`
guard (`vgae.py:419`) short-circuits the recon histogram when no masked nodes fall in
a batch. Verdict: acceptable for now; revisit if profiling shows >5% val overhead.

### 4. Double encode in `_score` — medium, hits every test step

`_score` (`vgae.py:458-480`) performs two encoder forward passes per call:

```python
# Pass 1: unmasked encode → z (for TAM affinity), mu (for Mahalanobis)
z, kl_per_node, mu = self.encode(batch.x, ...)

# Pass 2: masked encode + decode → cont (for recon error), z_masked, kl
(cont, ...), mask = self._masked_forward(batch)
```

`_score` is called from:
- `score()` — every `test_step` batch
- `extract_features()` — fusion feature extraction
- `_fit_score_norm`'s second val_loader pass (calibration)

The double-encode adds roughly one full encoder forward pass per test batch versus a
hypothetical single-pass design. For a VGAE with `hidden_dims=[128]`, `latent_dim=128`,
4 heads — the encoder is the dominant cost and running it twice at inference doubles
the inference compute.

**Why it's structurally necessary as written:** The unmasked z is needed for TAM
affinity because the calibration buffers in `_fit_score_norm` are fitted from unmasked
z (via `_score` in its second loop). Using masked z in `score()` while calibrating on
unmasked z would mis-apply the z-norm and corrupt the anomaly score. The masking path
is needed for recon error — these cannot be the same pass.

**What could reduce it:** if TAM affinity calibration were also done on masked z,
the unmasked encode in `score()` could be dropped and replaced with the masked z.
This would save one encoder pass per test step. The tradeoff is masked z has 15% of
nodes perturbed, which could slightly degrade the structural consistency signal.
Not attempted yet — leave as a future optimization if test throughput becomes the
bottleneck on large datasets.

**`_fit_score_norm` three-pass cost:** The first val_loader pass (mu collection)
runs an unmasked encode. The second pass calls `_score` which runs two more encodes
(unmasked + masked forward). So calibration runs 3 encoder passes per batch total.
Could unify to 2 by collecting mu and score components in one pass, but the second
pass requires `_score`'s structure to stay stable for calibration consistency. Low
priority since calibration runs once at test start, not per test batch.

---

## Correctness notes

### Z inconsistency between `validation_step` and `_score`

`validation_step` (`vgae.py:381`) computes TAM affinity from `z` returned by
`_masked_forward` — the reparameterized latent of the **masked** input. `_score`
(`vgae.py:469`) computes TAM affinity from `z` returned by the **unmasked** encode.

Consequence: the `val_node_affinity_*` MLflow metrics logged during training use masked
z; the test-time anomaly score uses unmasked z. These measure slightly different
distributions. The val affinity histograms are not directly comparable to the
calibrated score ranges logged at test time. Not a scoring bug (test scoring is
self-consistent via calibration), but the monitoring signal is potentially misleading
— e.g., `val_node_affinity_p99_attack` values may not match the affinity component's
contribution to the final `score()`.

Not fixed here; documenting so future dashboard readers don't compare val affinity
percentiles directly against test score distributions.

### `_fit_score_norm` doesn't restore `model.training` on exception

`vgae.py:582`: `if was_training: self.train()` is reached only if the method
completes without an exception. If the `n_total < 100` guard at line 587 raises,
or if the second loop raises, the model stays in eval mode. A future caller that
runs training after a failed calibration will silently get eval-mode behavior
(dropout off, batchnorm frozen). Fix is a `try/finally`:

```python
was_training = self.training
self.eval()
try:
    ...  # mu collection + score collection
    self.score_norm_fitted.fill_(True)
finally:
    if was_training:
        self.train()
```

Not fixed in this session — the calibration raise is typically fatal to the test run
anyway, so the model state mismatch is unlikely to matter in practice. Flagged for
when `_fit_score_norm` gets hardened.

---

## Test coverage gaps

`tests/core/models/test_vgae.py` has no tests for:

| Path | What's missing |
|---|---|
| `validation_step` | No invariant test for the per-class histogram or discrimination-gap logging |
| `_score` | No test for the 7-tuple return or that recon/affinity/rq are finite and shaped `[G]` |
| `score()` | No test — the `score_norm_fitted` guard and the max-σ aggregation are untested |
| `on_test_setup` / `_fit_score_norm` | No test for calibration with benign-only batches |

Existing tests (`test_tam_affinity_shape`, `test_rayleigh_quotient_per_graph`) cover
the primitives correctly. The integration path through `validation_step` and `score()`
is unvalidated — a future refactor of the benign/attack scatter or the z-norm
aggregation would not be caught by the test suite.

Suggested additions (deferred, not authored here):

```python
# INVARIANT: _score returns finite, per-graph tensors
def test_score_output_shapes(model_and_conv):
    model, _ = model_and_conv
    batch = make_batch(3)
    with torch.no_grad():
        recon, recon_max, affinity, rq, mahal, kl, z = model._score(batch)
    for name, t in [("recon", recon), ("recon_max", recon_max),
                    ("affinity", affinity), ("rq", rq)]:
        assert t.shape == (3,), f"{name} wrong shape"
        assert torch.isfinite(t).all(), f"{name} non-finite"

# INVARIANT: score() requires fitted norm; raises otherwise
def test_score_requires_fitted_norm():
    model = _make_vgae()
    batch = make_batch(2)
    with pytest.raises(RuntimeError, match="on_test_setup"):
        model.score(batch)
```

---

## Summary

| Issue | Severity | Datasets affected | Status |
|---|---|---|---|
| Double encode in `_score` (2× encoder per test step) | Medium | All, worst on `set_04` | Known; structurally necessary for calibration consistency |
| `torch.quantile` × 4-8 per val batch | Low-medium | `set_02+` only | Acceptable; revisit if profiling shows >5% val overhead |
| `tam_affinity` / `rq` / `recon_max` | Negligible | — | No action needed |
| Z inconsistency: val monitoring (masked) vs. test scoring (unmasked) | Low | Dashboard only | Documented; not a scoring bug |
| `_fit_score_norm` no exception-path training-state restore | Low | Calibration failures | Fix with `try/finally` when hardening |
| Zero tests for `validation_step`, `_score`, `score()`, calibration | Medium | Regression safety | Deferred; suggested stubs above |

No blocking issues. Current overhead is acceptable for `hcrl_sa`. The double-encode
in `_score` is the one to watch if `set_04` test throughput is slow — it's a known
2× encoder cost, not a bug.
