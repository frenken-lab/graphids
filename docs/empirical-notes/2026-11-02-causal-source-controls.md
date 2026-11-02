# Causal source controls — November 2, 2026

## Decision

Keep the original per-ID-gap candidate and its preset. Two small causal controls
expose different failures without new neural training: training-normal ID novelty
recovers most fuzzing positives, and a vehicle-local same-ID-gap score recovers
half the double positives at an empirical normal-frame budget. Neither repairs
force-neutral. These findings do not establish an independently confirmed model
improvement or an authorized normal profile for a new vehicle.

The next bounded implementation is a smoothed vehicle-scoped same-ID byte
transition control, with backoff and explicit support, on known-ID payload
failures. A per-ID GRU remains conditional on that diagnosis. Compare timing
scope/clipping on identical rows before attributing double's timing advantage to
missing learned history or changing the model's normalization.

## Protocol and correctness

CPU Slurm job **56946541** completed with exit code 0 in **2m01s**, requesting
eight CPUs and 32 GB; no GPU, neural forward pass or optimizer was used.
The frozen source and dependency versions verified before execution. All 13
control/failure-report tests passed in the allocation. The full local CPU suite
passed **146 tests**, with 24 compute/CUDA cases deselected. Ruff, documentation
lint and diff checks pass.

Both controls fit only verified-normal `train_query` recordings: 1,265,599 Impala
frames and 1,711,471 Silverado frames. Their respective ID sets contain 51 and
98 IDs; 50 and 80 IDs have supported timing tables. Normal traffic from attack
training captures, validation rows and reserved target captures cannot expand
these profiles.

- **ID novelty:** score one when the observed ID is absent from the vehicle's
  frozen normal-training union, zero otherwise. The fixed alarm is score >= 0.5.
  First occurrences of training-known IDs are legitimate startup observations.
- **Gap:** absolute robust normalized preceding same-ID gap. The existing fitter
  samples at most 65,536 deterministic rows per training capture, uses median
  positive-gap scale, log transform, median/IQR and a frozen training 99.9th
  percentile clip. An ID needs 32 sampled predecessor pairs. First occurrences
  and unsupported IDs abstain; there is no pooled-ID fallback. All query rows,
  including unknown labels and attacks, still advance causal observation history.
- **Vehicle scope:** equal numeric IDs on different vehicles are independent.
  A vehicle without an authorized normal-training profile abstains. This is
  source diagnosis, separate from the model's pooled checkpoint timing tables.

Every one of **12,758,965 physical validation rows across eleven captures**
matches reviewed source-cache timestamps, IDs, labels, payload support and source
digests. The three saved pilot classifier arms remain aligned with identical
payload scores. All **112 original classifier/payload metric values** reproduced
exactly, including aggregate classification AP **0.9256707652862181** and payload
AP **0.7344246230828388**. No selected checkpoint was replaced.

Threshold diagnostics use only the **3,248,139 verified-normal validation
frames**, pooling source vehicles. A >= threshold with conservative ties must
respect the requested empirical false-positive count. These recordings have
already participated in selection. This is neither independent calibration nor
an alerts/hour evaluation. Unsupported positives remain in recall denominators;
AP excludes unknown labels and uses each detector's declared support.

## Fixed novelty detects fuzzing without normal-ID false positives here

Novelty detects **163/172 fuzzing attack frames (94.77%)**, including **159 first
occurrences**, with **zero false-positive frames across all 12,722,481 validation
negatives**. Both normal captures and all attack-capture backgrounds have no
unseen normal IDs. All remaining attack positives use IDs in normal training;
novelty gives them no detections.

This completes the cross-capture check requested by the November 1 failure
report. The observed absence of false positives does not guarantee that another
driving mode, new benign ECU or another vehicle has the same normal-ID coverage.
Only two trusted training captures supply the profiles.

Novelty cannot detect the **nine fuzzing positives on training-known IDs**.
Original AP on those nine positives and 795,285 negatives is **0.034936**, versus
whole-fuzzing original AP **0.002617** on 172 positives. This conditional score
does not imply strong known-ID performance. Seven known-ID positives have local
timing support; gap AP on that support is 0.148524 versus original 0.052400.
The support denominator and positive count must accompany these small scores.

## Timing helps double, not every weak capture

At a **0.01% empirical normal-frame FPR budget**, original classifier, payload
and gap each spend **324 false-positive frames** on the normal validation
captures. Their thresholds are 0.0084531978, 10.11167145 and 32.58222246,
respectively. Novelty spends zero; its discrete score cannot use a fractional
tie group. Capture false positives below are measured among negatives inside
each attack recording after applying the common-normal-population thresholds.

| Capture | Original TP / total positives (FP) | Payload TP / total positives (FP) | Gap TP / total positives (FP) | Gap-unsupported positives |
| --- | ---: | ---: | ---: | ---: |
| Silverado double-4 | 7 / 356 (414) | 0 / 356 (416) | 177 / 356 (545) | 178 |
| Silverado force-neutral-4 | 26 / 56 (478) | 8 / 56 (455) | 4 / 56 (530) | 0 |
| Silverado fuzzing-4 | 16 / 172 (3,779) | 6 / 172 (213) | 3 / 172 (518) | 165 |
| Impala rpm-2 | 201 / 374 (260) | 361 / 374 (619) | 139 / 374 (189) | 0 |

Gap recovers **49.72% of all double positives**, versus original **1.97%**,
while retaining the unsupported half as misses. On the **identical 1,635,511
gap-supported double rows**, containing 178 positives, gap AP is **0.627940**
versus original **0.012035**. The whole-capture classifier AP 0.006380 and
payload AP 0.026109 include all 356 positives and use a different population.

All 56 force-neutral positives have gap support. On identical gap-supported
rows, gap AP **0.005348** remains below original **0.035646**. More timing is not
a uniform repair. Vehicle-scoped byte novelty in the preceding failure report
and complementary payload performance on RPM justify investigating conditional
payload learning before a larger recurrent architecture.

The neural classifier is also much stronger than gap on Silverado DoS at this
budget: **5,924/5,924** detections versus **578/5,924**, with capture FP 1,049
versus 364. Gap should not replace the classifier based on double alone.

Aggregate gap AP is **0.781547** on **12,713,206 supported known rows**, including
36,141 positives. Original AP is **0.925671** on all 12,758,965 known rows,
including 36,484 positives. Their recording macro APs are 0.501282 and 0.504552,
respectively, over nine mixed-class captures per detector, but the underlying
rows differ. These values are not matched-support model rankings. The report
contains matching-row comparisons within every capture's gap-support stratum.

## Artifacts and reproduction

Maintained implementation:
[event_causal_controls.py](../../scripts/event_causal_controls.py).
Tests: [test_event_causal_controls.py](../../tests/test_event_causal_controls.py).

```bash
# Inside a CPU Slurm allocation. OUTPUT must not exist.
python -m scripts.event_causal_controls --pilot PILOT_ROOT --output OUTPUT
```

Evidence root:
`/fs/ess/PAS1266/graphids/dev/rf15/experimentruns/event-causal-controls-20261102T173642Z/`.

`analysis/report.json` contains frozen training profiles, all support/unknown
counts, per-capture and aggregate metrics, known-ID/novel-ID/startup/gap-support
strata, budget thresholds and input/source/dependency hashes. `report.md`
summarizes the comparison. Eleven `controls-*.npz` files retain all control scores
and masks in physical row order, bound to each source digest. The source snapshot,
source hashes, dependency versions, preparation/submission receipts and
`report-56946541.log` preserve execution evidence.
`artifact-audit.json` verifies source and control-artifact hashes, complete
saved-score coverage, metric reproduction, novelty totals, budget bounds and
successful Slurm accounting.

No target files were opened. No frames were relabeled. The earlier independent
label audit, reviewed-family/non-DoS report, additional seeds and independent
calibration/confirmation grouping remain unfinished. Do not install a hybrid
threshold or claim production gains from this reused selection cohort.
