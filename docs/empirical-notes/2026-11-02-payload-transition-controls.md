# Payload-transition controls — November 2, 2026

## Decision

The frozen vehicle-scoped payload controls are implemented and evaluated. Adding
the preceding same-ID byte improves ranking on several captures, especially
Impala RPM and force-neutral, but does not repair Silverado force-neutral or
double. Retain the selected neural checkpoint. Do not install a fused detector
or launch a per-ID GRU based on these controls alone.

Next, diagnose the pooled normal-score tail ties and per-ID score scales, and
isolate the timing scope/clipping difference seen in the earlier double control.
Independent confirmation/calibration grouping and the remaining reviewed-family
and non-DoS reports remain open. This conditional-byte baseline cannot resolve
whether richer multibyte or longer local history would improve the neural model.

## Frozen recipe

Only verified-normal `train_query` captures with verified payload lengths fit
counts. The two profiles are separate by vehicle, numeric CAN ID and byte
position. Every training byte and eligible preceding same-ID byte pair is
counted; no sampling or query adaptation occurs. Recording boundaries reset
predecessors, while execution-block boundaries preserve them.

For current byte `b`, preceding same-position byte `a`, training marginal count
`n(b)`, position total `N`, pair count `n(a,b)` and context total `n(a)`:

```text
P_marginal(b)     = (n(b) + 1/256) / (N + 1)
P_transition(b|a) = (n(a,b) + 32 * P_marginal(b)) / (n(a) + 32)
frame_score      = mean(-log(P(current_byte))) over valid current bytes
```

Both distributions normalize across all 256 values and assign finite surprise
to unseen values. Prior masses **1 and 32 were fixed before the evaluation**.
There was no validation sweep. Sparse contexts blend toward the marginal; first
occurrences, unavailable preceding byte positions and unseen contexts use it
directly. Unknown labels and attacks still contribute preceding observations,
but cannot update counts. Current observed bytes index frozen predictions.

Both controls abstain for the whole frame if any current valid byte position
lacks a training profile. Missing/unverified payloads and unprofiled vehicles
also abstain. No cross-ID or cross-vehicle fallback invents semantic matches.
The two controls share exactly the same supported rows. Unsupported positives
remain misses in budgeted recall; AP excludes unknown labels and unsupported
frames, with counts retained separately.

| Training vehicle | Frames | Valid bytes | Eligible byte pairs | Sparse transition states |
| --- | ---: | ---: | ---: | ---: |
| Impala | 1,265,599 | 8,644,637 | 8,644,295 | 346,466 |
| Silverado | 1,711,471 | 11,121,317 | 11,120,700 | 512,459 |
| Total | 2,977,070 | 19,765,954 | 19,764,995 | 858,925 |

These are correlated observations from only two normal captures, not millions
of independent training examples.

## Source validation results

Every one of **12,758,965 physical validation frames across eleven captures**
matches the saved pilot arms and reviewed source caches. All earlier classifier,
payload, ID-novelty and timing-control metrics and thresholds reproduce exactly.
The original classifier/payload parity check covers **112 metric values**.
Reserved target files remain unopened.

The new controls support **12,758,798 frames**, containing 36,317 positives and
all 12,722,481 negatives. The 167 unsupported frames are all fuzzing positives.
All attack captures except fuzzing have complete current-payload/control support.

| Capture | Marginal AP | Transition AP | Original classifier AP | Neural payload AP |
| --- | ---: | ---: | ---: | ---: |
| Impala DoS-2 | 0.207026 | 0.259395 | 0.979127 | 0.914843 |
| Silverado DoS-4 | 0.718084 | 0.712633 | 0.980346 | 0.831343 |
| Impala force-neutral-2 | 0.722467 | 0.921582 | 0.778327 | 0.729704 |
| Impala rpm-2 | 0.072947 | 0.515462 | 0.309721 | 0.455335 |
| Impala standstill-1 | 0.493099 | 0.778172 | 0.855627 | 0.831365 |
| Silverado force-neutral-4 | 0.000345 | 0.003559 | 0.035643 | 0.007340 |
| Silverado double-4 | 0.006186 | 0.007894 | 0.006380 | 0.026109 |
| Silverado triple-4 | 0.336733 | 0.355966 | 0.593182 | 0.513292 |
| Silverado fuzzing-4, common byte support only | 0.000298 | 0.320196 | 0.062864 | 0.091614 |

Each row compares identical frames across its four columns. The fuzzing row
contains **five positives** and 795,285 negatives. It is not comparable to the
whole-fuzzing classifier AP 0.002617 on 172 positives, or neural payload AP
0.007263 on 160 positives. Of nine fuzzing positives on training-known IDs,
eight have neural payload support and five have complete trained-position
support. ID novelty already detects 163/172 fuzzing positives separately.

On exactly matched new-control support, aggregate AP is **0.019949 marginal**
versus **0.026972 transition**. Recording macro AP is **0.284132** versus
**0.430540**, averaging the nine mixed-class recordings on that support.
Both normal captures have undefined AP and remain represented by exposure and
false-positive counts. The unchanged classifier aggregate AP 0.925671 uses a
different positive denominator, so it is not a matched aggregate ranking column.

These results support a limited conclusion: preceding byte history is informative
beyond marginal frequency on some captures. They neither establish a general
transfer gain nor show that the weakest captures need a particular neural cell.

## AP gains do not give a usable pooled operating point

Thresholds use only the same **3,248,139 verified-normal selection-validation
frames** as earlier diagnostics. They are not independently calibrated thresholds
or false-alert rates per hour. Conservative >= thresholding never splits tied
scores or randomizes decisions to spend an allowance.

| Allowed empirical normal FPR | Allowed normal FP | Marginal achieved FP | Transition achieved FP |
| --- | ---: | ---: | ---: |
| 0 | 0 | 0 | 0 |
| 0.001% | 32 | 0 | 2 |
| 0.01% | 324 | 0 | 2 |
| 0.1% | 3,248 | 3,066 | 3,088 |

At the 0.01% allowance, both new thresholds are **15.734820905471427**.
Large tied score groups prevent spending most of the allowance. The neural
classifier, neural payload and gap control each attain 324 normal FP at that
allowance. Comparing their recall as if the achieved FP counts matched would
hide a material difference.

At this allowance, both byte controls detect **zero** force-neutral-4 or double-4
positives; marginal incurs zero capture FP and transition two in each capture.
Transition detects three fuzzing positives with three capture FP, while its 167
unsupported positives remain misses. It detects 44/941 triple positives with
zero capture FP. It detects no positives on the other attack captures.

The looser 0.1% allowance does not repair the Silverado pair: both still have
zero detections, with 2,710 marginal capture FP and 11,425 transition capture FP
in each capture. Transition detects 242/492 Impala force-neutral positives with
23 capture FP and 516/2,215 standstill positives with five capture FP, but still
zero RPM positives with three capture FP. Better RPM AP does not translate into
recall at these pooled thresholds. Inspect normal-score tails and scale
differences before treating this as a missing-capacity result; do not fit profiles
or tune new transformations on these validation negatives and call them test data.

## Implementation, validation and artifacts

Core implementation:
[event_payload_control.py](../../scripts/event_payload_control.py).
Report integration:
[event_causal_controls.py](../../scripts/event_causal_controls.py).
Tests: [test_event_payload_control.py](../../tests/test_event_payload_control.py).

Six new tests include an independent serial counting/scoring oracle, 256-value
probability normalization, vehicle isolation, capture resets, unknown-label
history, variable lengths, full-frame abstention, block invariance, future
perturbations, immutable fitted arrays, 29-bit IDs and serialized count reloads.
All 19 focused control/report tests pass. The frozen allocation passed **152 CPU
tests**, with 24 compute/CUDA cases deselected. Ruff, formatting, diff checks,
documentation lint and strict MkDocs build pass.

CPU job **56946799** completed with exit code 0 in **4m33s**, including the full
CPU suite, source comparison and artifact audit. It requested eight CPUs and
32 GB and used no GPU, neural forward pass or optimizer.

```bash
# Inside a CPU Slurm allocation; OUTPUT must not exist.
python -m scripts.event_causal_controls --payload-transition \
  --pilot PILOT_ROOT --output OUTPUT
```

Evidence root:
`/fs/ess/PAS1266/graphids/dev/rf15/experimentruns/event-payload-controls-20261102T175023Z/`.

`analysis/report.json` retains the recipe, fit exposure, complete metrics,
matched-support/backoff strata, per-frame byte totals, thresholds and provenance.
`analysis/report.md` summarizes results. Eleven `controls-*.npz` archives preserve
scores and masks in physical row order. Two `payload-profile-*.npz` archives
preserve every sparse integer count table. Source/dependency hashes verified
before execution; preparation/submission receipts and `report-56946799.log`
retain the execution evidence.

`artifact-audit.json` verifies exact reproduction of earlier results and budgets,
112 original/payload metric values, matching marginal/transition populations,
score coverage/finiteness/abstention, support/backoff partitions, byte totals,
serialized count totals, every score/profile hash and empirical budget bounds.
No new threshold, neural feature or classifier artifact was promoted.
