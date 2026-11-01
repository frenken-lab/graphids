# Weak-capture diagnosis — 2026-11-01 15:17:25 EST

Recorded: **2026-11-01T15:17:25-05:00**, America/New_York.

UTC: **2026-11-01T20:17:25+00:00**.

## Decision

Keep the original per-ID-gap candidate. **Test a training-normal ID-novelty
control before adding a per-ID GRU.** The three weak captures have different
failure modes. In fuzzing, most attacks arrive on IDs with no preceding local
history. In force-neutral, timing support is available and the classifier ranks
some attacks above ordinary traffic, but useful recall still incurs substantial
false positives. Double has mixed timing support and some complementary payload
signal at a looser budget.

The current evidence does not establish that recurrent memory is the missing
mechanism. It also does not establish that the labels are wrong or that a simple
novelty alarm will generalize to other captures or unseen vehicles.

## Completed analysis and exposure

CPU job **56924611** completed with exit code 0 in **1m20s**, requesting no GPU.
The report verifies every one of **12,758,965 validation rows across 11 captures**
against source-cache timestamps, IDs, labels, recording digests and payload support.
All three saved classifier arms have aligned rows and identical payload scores.
It exactly reproduces their saved metrics on the three focused captures.

| Silverado validation capture | Explicitly matched training capture | Training positives | Validation positives |
| --- | --- | ---: | ---: |
| double-4 | double-3 | 296 | 356 |
| force-neutral-4 | force-neutral-3 | 32 | 56 |
| fuzzing-4 | None | 0 | 172 |

Capture matching is an explicit reporting annotation checked against reviewed
source assignments; no family feature is inferred inside model execution.
The focused captures contain 4,088,233 physical rows and zero unknown labels.
All unknown-label paths remain tested. Fuzzing has **12 positive frames without
payload support**; payload AP excludes these, while budgeted payload recall keeps
all 172 positives in its denominator and unsupported frames cannot alarm.

Byte-value and same-ID byte-transition profiles use only the two verified-normal
`train_query` captures. Validation never updates these profiles. Primary profiles
are vehicle-scoped; pooling numeric IDs across vehicles is a sensitivity check.
Neither profiles nor budget thresholds change the production model.

## What the failures contain

### Fuzzing: unseen IDs and little local history

- **163/172 attack frames (94.77%)** use IDs absent from Silverado's normal
  training capture; all 795,285 labeled-negative frames in this fuzzing capture
  use IDs present in that training capture. This is a single-capture observation,
  not an evaluated ID-novelty detector across the full validation cohort.
- **159/172 positives (92.44%)** are first occurrences of their IDs in this
  recording. Only 12 have an eligible preceding same-ID payload byte pair;
  13 have a preceding same-ID event, and eight have fitted timing support.
- The classifier misses 170/172 positives at 0.5. Their median original score is
  approximately 0.0000657, below the negative-frame median of 0.0000758.
- Nine positives have IDs present in vehicle-scoped normal training. Low
  known-position byte-novelty counts must not be read as normal payloads:
  most attack IDs lack a normal payload profile altogether.

A per-ID GRU would have no preceding local sequence for most of these positives.
The immediate question is whether a training-normal ID-support control detects
them without unacceptable false positives on other captures and startup traffic.
Such a control would be vehicle-scoped source diagnosis, not zero-shot transfer
to a new vehicle whose legitimate IDs are unknown.

### Force-neutral: available timing and source-dependent payload support

All 56 positives have preceding history, valid payloads and fitted same-ID timing
support. The median gap is 13.15 ms; the reconstructed normalized gap median is
-29.04. Zero recall at 0.5 cannot be attributed to missing timing support here.
The original attack-score median is 0.006586, versus 0.0000786 over all negative
frames in the capture. These comparisons mix different negative IDs; they do not
establish an ID-conditioned false-positive rate.

Every positive has a byte value absent at that ID/position in Silverado normal
training, but **none** has marginal byte-value novelty after pooling normal
training across both vehicles. All 56 also contain an unseen same-ID transition
between individually known byte values. A frame can have a novel value at one
byte position and a novel transition between known values at another position.

Normal variation remains substantial: 131,592/1,646,182 negative frames have
vehicle-scoped byte-value novelty, and 246,337 have novel transitions between
known values. A raw novelty flag is therefore not an adequate alarm rule.
Only 32 matching training positives and cross-vehicle support differences deserve
attention before increasing architectural complexity.

### Double: two attacked IDs, mixed support

The 356 positives split equally between IDs `0x140` and `0x34A`. Exactly 178
positives have fitted timing support; the other 178 use the fallback distribution.
All have preceding history and valid payloads. There are 178 positives with novel
byte values and 46 with novel transitions between individually known values.
The original classifier detects none at 0.5.

Payload AP is 0.026109, versus classifier AP 0.006380 on the same rows. This is
complementarity worth measuring, but the budget comparison below shows it helps
only at the looser of the highlighted operating points.

## Recall at matched empirical normal-frame budgets

Each detector's threshold uses only the **3,248,139 known-negative frames from
the two verified-normal validation captures**. These are selection-validation
diagnostics, not independent calibration. Conservative tie handling achieves
324 false-positive frames at the 0.01% budget and 3,248 at 0.1%, for each detector.

False positives in parentheses below occur among labeled-negative frames
**inside the attack capture**, after applying that global normal-budget threshold.
They expose differences between normal-capture and attack-capture backgrounds.

| Normal-frame FPR budget | Capture | Original classifier recall (FP) | Payload recall (FP) |
| --- | --- | ---: | ---: |
| 0.01% | double-4 | 1.97% (414) | 0% (416) |
| 0.01% | force-neutral-4 | 46.43% (478) | 14.29% (455) |
| 0.01% | fuzzing-4 | 9.30% (3,779) | 3.49% (213) |
| 0.1% | double-4 | 10.39% (4,457) | 44.10% (2,011) |
| 0.1% | force-neutral-4 | 60.71% (5,318) | 25.00% (2,018) |
| 0.1% | fuzzing-4 | 9.88% (8,900) | 12.79% (13,028) |

Lowering the threshold recovers some force-neutral recall, but does not resolve
its false-positive trade-off. On double at 0.1%, the payload head detects 157
positives, including 121 missed by the classifier; at 0.01% it detects none.
On force-neutral, payload adds no positives beyond the classifier at any tested
budget. On fuzzing at 0.01%, it adds only one positive and 187 additional
false-positive frames if combined by OR. There is no uniform case for automatic
head fusion from these results.

The machine-readable report also includes ordinary/balanced classifier arms,
zero and 0.001% budgets, achieved normal rates per capture, common-support AP,
score/gap quantiles, disagreement counts and physical-row examples. None of these
thresholds was installed as a production operating point. Frame false positives
are not merged alerts/hour.

## Labels, clocks and limits

The analysis checks reviewed file hashes and source identity, unchanged saved
labels, finite monotone cached clocks, and complete physical-row coverage.
Existing upstream clock-audit evidence records eight corrected rows in
force-neutral-4 (maximum 58 microseconds), four in double-4 (60.8 microseconds),
and none in fuzzing-4. It is retained as `reviewed-clock-evidence.json` under the
analysis root. This does not independently establish attacker origin, trusted
incident intervals, or new ground-truth labels. No frames were relabeled.

The checkpoint and this validation cohort have already selected features and
epochs. These findings are exploratory. Two normal training captures provide
limited coverage of ordinary driving; byte novelty also occurs on many legitimate
frames. Pooled versus vehicle-scoped differences are sensitivity evidence,
not proof that cross-vehicle pooling caused the neural model's failures.

## Next bounded step

Evaluate a **fixed, training-normal ID-novelty control on every source validation
capture**, including normal traffic and recording starts, with explicit vehicle
scope and all unknown/missing-payload rows retained. Follow with a simple causal
same-ID-gap control on known IDs. These controls should establish what the neural
candidate adds and whether weak known-ID attacks need more learned payload history.
The present report does not justify starting a per-ID recurrent training run yet.

## Reproduction and validation

Maintained script: [event_failure_report.py](../../scripts/event_failure_report.py).

Tests: [test_event_failure_report.py](../../tests/test_event_failure_report.py).

```bash
# In a CPU Slurm allocation. FOCUS.json explicitly maps validation keys to
# matched training keys; use an empty list when no matched capture exists.
python -m scripts.event_failure_report --pilot PILOT_ROOT \
  --focus FOCUS.json --output NEW_DIRECTORY
```

Evidence root:
`/fs/ess/PAS1266/graphids/dev/rf15/experimentruns/event-failure-report-20261101T201147Z/`.
`analysis/report.md` and `analysis/report.json` contain the generated results;
`artifact-audit.json`, `source-sha256.json`, `submission.json`, `focus.json` and
`report-56924611.log` retain the audit, source snapshot and execution evidence.
Input score/config/checkpoint hashes, dependency versions and the reviewed source
manifest are preserved. The report reads checkpoint timing buffers on CPU and
executes no neural forward pass or optimizer update. No target files are opened.

Seven focused tests pass locally and in the CPU allocation. The full CPU suite
passes **140 tests** (24 compute/CUDA cases deselected); Ruff and diff checks pass.
The artifact audit verifies exact saved-metric reproduction, coverage, timing
partitions, disagreement/decision counts, unsupported-positive denominators,
all empirical normal budgets and frozen source hashes.
