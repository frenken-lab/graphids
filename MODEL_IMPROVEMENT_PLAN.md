# CAN event model: evidence and implementation plan

Updated: 2026-11-02. Research reviewed through October 31.

The active candidate is **raw CAN frames plus normalized per-ID timing gap**, with
one stream, ordinary BCE and the joint normal-payload objective. Keep this path
small. The October 30 comparison is complete; weaker feature branches and their
generators were retired on October 31. The paired frozen-classifier pilot,
source-only failure report and causal diagnostic controls below are implemented
separately; other future improvements remain proposals. A pilot implementation
is not evidence of a model-quality gain.

## 1. Current implementation and operating constraints

The active preset is
[`configs/experiments/graph_mamba_events_source.yml`](configs/experiments/graph_mamba_events_source.yml).
Its model and representation use schema 2: 51 observation fields, a 35-field
forecast header, width 128, two Mamba layers, dropout 0.1, one A100, 25 epochs,
16,384-event chunks, a 65,536-event optimizer window and 8,388,608 graph tokens.
The classifier retains raw elapsed/bus timing; normalized same-ID gap, normalizer
support and first-seen indicators supplement the raw observations.

- [x] Fit timing statistics only on reviewed, verified-normal `train_query` data.
- [x] Keep fitted timing tables in checkpoints; evaluation never refits them.
- [x] Preserve every physical row, all available ID bits, bytes, lengths and timestamps.
- [x] Preserve chronological stream state, exact tails and unknown-label masks.
- [x] Forecast current payload from preceding states and current header only.
- [x] Report full-epoch loss totals with separate classification/normal-byte counts.
- [x] Save aggregate and recording-level metrics and complete frame predictions.
- [x] Refresh `last.ckpt` independently of best-score improvement, with SHA sidecars.
- [x] Retain loading of the selected schema-2 and explicit schema-1 baseline checkpoints.
- [x] Remove no-elapsed, normalized-bus, payload delta/XOR and graph-age branches,
  preceding-payload copies and graph-age timestamps from active execution.
- [x] Replace the control/context generators with one candidate workflow.
- [x] Complete A100 verification of the October 31 cleanup; job 8238137 passed.

Each newly prepared candidate uses **three jobs**: one CUDA gate, one fit after a
successful gate, and one completeness report after the fit terminates. Nothing is
submitted by preparation alone:

```bash
python -m scripts.event_quality_suite prepare OUTPUT --seed 43
python -m scripts.event_quality_suite verify OUTPUT
python -m scripts.event_quality_suite submit OUTPUT
python -m scripts.event_quality_suite status OUTPUT
```

Use distinct output directories for repeat seeds. Source snapshots, expanded
configs, reviewed manifests, dependency versions, hashes and submission receipts
are frozen. Existing/partially submitted suites are not silently overwritten or
resubmitted. The report is `candidate-results.json`;
[`scripts/event_suite_report.py`](scripts/event_suite_report.py) also reads the
historical controls/context suite without constructing retired model variants.

The raw-message and causal-state contract is described in
[the event-model reference](docs/reference/graph-mamba-events.md). Numeric CAN IDs
are not assumed to have common semantics across vehicles. Reserved target files
remain excluded from development. Source-only transfer and target-normal
adaptation are separate protocols. Missing-message detection needs timer-driven
outputs; an observed-frame classifier cannot emit a score during complete silence.

## 2. What was tested and what worked

### Matched October 30 suite

Evidence root:
`/fs/ess/PAS1266/graphids/dev/rf15/experimentruns/event-quality-context-20261030T185508Z/`.

All **15 jobs completed with exit code 0**: two CUDA gates, eleven fresh fits and
two reports. Each fit completed 25 validation epochs on exactly 12,758,965 frames:
36,484 positive and 12,722,481 negative. Training contained 8,526,701 labeled
frames, including 178,904 positives, across ten captures; only two captures were
verified normal. DoS supplied 166,791 training positives (93.23%).

The arms share seed 42, chunk length, optimizer-window size, graph capacity and
training exposure. The context baseline reuses the ordinary-BCE one-stream fit.
Best epochs maximize aggregate validation classification AP, choosing the earliest
on ties. Epoch numbers below are one-based. These are exploratory source-validation
results, selected over epochs and arms, not independent-test estimates.

| Tested configuration | Ascend job | Best classification AP | Best epoch | Final AP | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| One stream, ordinary BCE, joint objective | 8217868 | 0.047757 | 10 | 0.018840 | Matched historical baseline |
| One stream, positive-weighted BCE | 8217869 | 0.070588 | 15 | 0.010163 | Retire generator arm |
| Four streams, ordinary BCE | 8217870 | 0.433955 | 23 | 0.406115 | Retire generator arm |
| Four streams, positive-weighted BCE | 8217871 | 0.179478 | 22 | 0.164425 | Retire generator arm |
| One stream, classification only | 8217872 | 0.065888 | 9 | 0.001629 | Retire generator arm |
| Remove elapsed recording time | 8217874 | 0.744381 | 25 | 0.744381 | Retire feature branch |
| Normalize bus interarrival time | 8217875 | 0.805402 | 25 | 0.805402 | Retire feature branch |
| **Normalized per-ID gap** | **8217876** | **0.925671** | **25** | **0.925671** | **Keep as the candidate** |
| Payload delta | 8217877 | 0.148489 | 24 | 0.133621 | Retire feature branch |
| Payload XOR | 8217878 | 0.785281 | 25 | 0.785281 | Retire feature branch |
| Graph-age bias | 8217879 | 0.089398 | 16 | 0.025462 | Retire feature branch |

The positive weight was the training-only negative/positive ratio, 46.6607622.
Weighting and classification-only training were tested on the baseline, **not**
on the winning per-ID model. Their results do not establish that every balanced
objective or payload-loss change would fail on the winner. Context features were
individual interventions; no winning-feature combination was evaluated. Retired
means removed from the maintained path, not scientifically disproven.

The earlier October 16 fit, job 7894453, reached best AP 0.1436 at epoch 20 and
final AP 0.0737. It used a 16,384-event optimizer window and 13,150 total steps.
It is useful historical context, but is not the matched comparator above.

### Winning checkpoint: strengths and remaining failures

| Metric at epoch 25 | Value |
| --- | ---: |
| Classification AP / ROC-AUC | 0.925671 / 0.994256 |
| Precision / recall / F1 at diagnostic threshold 0.5 | 0.898060 / 0.893186 / 0.895616 |
| TP / FP / FN | 32,587 / 3,699 / 3,897 |
| False-positive frame rate at 0.5 | 0.000290745 |
| Payload anomaly AP / ROC-AUC | 0.734425 / 0.995319 |
| Positive-capture macro AP, nine attack recordings | 0.504552 |

The macro AP is the unweighted mean of nine recording APs, excluding the two
zero-positive normal recordings. It is **not** pooled non-DoS AP or a macro score
across seven attack families. DoS accounts for 31,878 / 36,484 validation positives
(87.38%), so aggregate AP can hide weak small groups.

| Validation recording | Positives | Classification AP | Recall at 0.5 | False-positive frames |
| --- | ---: | ---: | ---: | ---: |
| Impala DoS-2 | 25,954 | 0.979127 | 89.20% | 2,158 |
| Silverado DoS-4 | 5,924 | 0.980346 | 99.98% | 403 |
| Impala force-neutral-2 | 492 | 0.778327 | 99.80% | 243 |
| Impala rpm-2 | 374 | 0.309721 | 48.13% | 238 |
| Impala standstill-1 | 2,215 | 0.855627 | 99.95% | 516 |
| Silverado force-neutral-4 | 56 | 0.035643 | 0% | 4 |
| Silverado double-4 | 356 | 0.006380 | 0% | 1 |
| Silverado triple-4 | 941 | 0.593182 | 66.63% | 101 |
| Silverado fuzzing-4 | 172 | 0.002617 | 1.16% | 32 |
| Impala attack-free-1 | 0 | Undefined | Undefined | 0 |
| Silverado attack-free-4 | 0 | Undefined | Undefined | 3 |

All recording metrics use the **same aggregate-AP-selected epoch**, rather than
choosing a separate favorable epoch per recording. AP is undefined on all-normal
captures. False-positive frames are not merged alerts/hour; normal exposure and
an alert rule have not been calibrated.

What the evidence supports:

- Same-ID timing is the strongest tested individual addition on this cohort.
- The payload head is now informative in aggregate. The old baseline's
  below-chance payload AUC no longer describes the selected model.
- Weak Silverado families and RPM need focused work; timing-only success is not
  enough to establish payload-semantic or stealth-attack detection.
- Training is not demonstrably converged: winner AP at epochs 23/24/25 was
  0.9041 / 0.7261 / 0.9257. More epochs are a hypothesis, not an established fix.
- One seed and very few independent captures do not establish stable transfer.

### Correctness, infrastructure and checkpoint audit

| Check | Result and scope |
| --- | --- |
| Original control CUDA gate, 8217866 | Passed; 126 tests, weighted one/four-stream checks and public launcher smoke |
| Original context CUDA gate, 8217867 | Passed; 126 tests, combined-context checks and public launcher smoke |
| Training/report jobs | All eleven fits have 25 epochs and matched coverage; reports 8217873/8217880 complete |
| October 30 experiment-package fix | 125 CPU tests on Pitzer; fit 56868128 and separate checkpoint test 56868130 completed |
| Final-checkpoint regression | Synthetic two-epoch fit/test predictions bitwise equal; `last.ckpt` epoch 1, step 4, despite tied best AP |
| October 31 cleanup | 123 CPU tests pass; 24 compute/CUDA cases deselected; historical schema-1/schema-2 Lightning loading covered |
| Static/package checks after cleanup | Ruff, diff checks, documentation symbols, strict MkDocs and wheel build pass |
| Frozen historical suite | Source/config/dependency hashes verify; current report tooling still reads all arms |
| Cleanup CUDA gate, 8238137 | **Passed**, exit 0 in 3 minutes: 147 tests, bitwise checkpoint parity, bounded source benchmark and public launcher smoke |
| Recording-report follow-ups | 125 CPU tests pass (24 compute/CUDA cases deselected), including 10 focused report/suite tests; saved historical reports regenerate; lint and strict docs pass |

The historical suite used the older frozen callback. Seven `last.ckpt` files are
stale (baseline, both weighted controls, four-stream unweighted, classification
only, delta and graph age). All eleven best-AP checkpoints and their SHA sidecars
verified. The selected file is `checkpoints/epoch=24-step=3300.ckpt`, genuinely
containing the final epoch. Use verified epoch/global-step and explicit best files
when loading historical runs; filename `last` alone is insufficient evidence.

Evidence to preserve under the suite root: `suite.json`, `controls-results.json`,
`context-results.json`, `training-cohort.json`, `training-manifest.json`,
`source-sha256.json`, `artifact-sha256.json`, `dependency-versions.json`,
`working-tree.diff`, gate receipts and complete per-epoch prediction indices.
Do not edit the completed source snapshot to match current code.

Infrastructure receipts:

- `/fs/ess/PAS1266/graphids/dev/rf15/experimentruns/exp-refactor-validation-fixed-20261030T192540Z/validation.json`
  records the successful Pitzer fit/test/checkpoint audit. The preceding failed
  smoke is retained separately as evidence of the original callback defect.
- `/fs/ess/PAS1266/graphids/dev/rf15/experimentruns/id-gap-focus-validation-20261031T190830Z/`
  contains the cleanup snapshot and completed job 8238137. `checkpoint-parity.json`
  verifies bitwise equality for the actual baseline and winner checkpoints against
  frozen execution on A100: fields, headers, both output heads, graph/bus state,
  temporal caches and winner timing buffers. This used six synthetic frames per
  checkpoint across two chunks, not the full validation cohort.
  `gate/correctness.log` records 147 passing tests; `gate/gate.json` is passed,
  and the public smoke manifest is finished. The bounded source benchmark scored
  320 rows; it is correctness evidence, not a production-throughput or quality fit.

## 3. Research findings and what they imply here

These are primary papers or author/publisher material reviewed on October 31.
External benchmark numbers are not comparisons against this project. The proposed
adaptations are engineering hypotheses, with the protocol differences stated.

| Research | Relevant result or mechanism | Application and limit |
| --- | --- | --- |
| [CANet, 2019](https://arxiv.org/html/1906.02492v1) | Separate per-ID recurrent subnetworks combine into a joint traffic state. | Supports learning local histories; its decoded signals, ID-specific weights and reconstruction objective differ from our raw-byte shared-cell proposal. |
| [Temporal Graph Networks, 2020](https://arxiv.org/abs/2006.10637) and [author's architecture description](https://emanuelerossi.co.uk/tgn/) | Event-driven node memory combines with graph operators. | Supports a causal local-memory experiment; generic dynamic-graph results do not establish CAN detection gains. |
| [Decoupling Representation and Classifier, ICLR 2020](https://arxiv.org/abs/1910.09217) | Classifier rebalancing can help long-tailed visual recognition with a frozen representation. | Motivates a cheap frozen-encoder diagnostic before new recurrent architecture; binary CAN attribution and correlated frames differ from multiclass images. |
| [Class-Balanced Loss, CVPR 2019](https://arxiv.org/abs/1901.05555) | Weights use effective sample counts rather than raw inverse counts. | Motivates capped training-group weighting; millions of correlated CAN frames are not independent samples. A single attack/nonattack weight cannot rebalance attack families. |
| [Group DRO, ICLR 2020](https://arxiv.org/abs/1911.08731) | Worst-group optimization needs regularization to generalize in overparameterized models. | Consider only after reviewed groups and simple balance; tiny rare groups and label noise can dominate an unregularized worst-group objective. |
| [Gradient Surgery / PCGrad, NeurIPS 2020](https://papers.nips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html) | Addresses conflicting shared-parameter task gradients. | Measure classifier/payload gradient norms and cosine first; different scalar loss magnitudes do not establish interference. |
| [On Calibration of Modern Neural Networks, ICML 2017](https://proceedings.mlr.press/v70/guo17a.html) | Temperature scaling is a simple calibration method. | Calibration is a separate task from learning ranking; it requires separate calibration data here. |
| [ROAD dataset guide, 2024](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0296879) and [online masquerade benchmark, journal version 2026](https://arxiv.org/abs/2406.13778) | Advanced attacks can preserve nominal timing; attack types, labels and streaming protocols matter. | Keep timing as a strong cue, then test payload/context behavior. Do not relabel existing weak captures as masquerade or infer trusted incident intervals. |
| [MIDS, June 2026 preprint](https://arxiv.org/html/2606.18599v1) | Dual ID/payload streams and bidirectional Mamba target synthesized timing-preserving attacks. | Explore causal payload/context learning; its 100-frame bidirectional windows and evaluation protocol are not immediate per-frame causal inference or zero-shot transfer. |
| [Lightweight temporal and spatial fusion, June 2026](https://www.sciencedirect.com/science/article/pii/S1570870526000909) | Publisher abstract describes per-ID timing plus payload likelihood and precision-oriented AND fusion. | Head complementarity is worth measuring. Requiring a timing alarm would suppress timing-normal payload attacks, so do not adopt AND fusion by default. Abstract reviewed; full text unavailable here. |
| [MSC-IDS, IEEE IoT Journal early access, October 2026](https://ieeexplore.ieee.org/abstract/document/11719585/) | Publisher abstract describes masked payload-image pretraining with limited labels. | Supports investigating label-efficient payload representation learning; image reconstruction/windows do not establish causal forecasting or held-out-vehicle gains. Abstract reviewed; full text unavailable here. |

## 4. Next implementation work, in priority order

### P0. Verify cleanup, diagnose the saved winner and confirm repeatability

- [x] Finish job 8238137 and inspect checkpoint-parity, test and public-smoke artifacts.
- [x] Add per-epoch recording macro AP and a weakest-first recording ranking to
  suite reports. Average only mixed-class recordings with defined finite AP;
  retain all-normal recordings in the underlying metrics. Best/final summaries
  use the existing aggregate-AP epoch selection. This is a reporting change,
  not the full diagnosis below or a model-quality improvement.
  Nine focused report/suite tests passed. Regenerating all eleven historical
  summaries from saved indices reproduced winner macro AP 0.504552; the output
  is `recording-ap-results.json` under the October 30 suite root.
- [x] Compare classifier and payload AP on the three weakest recordings at the
  same selected epoch, with each head's eligible known-row count. Mark payload
  scores as untrained for classification-only arms; different support masks mean
  the AP values are not necessarily measured on identical rows. This does not
  implement score fusion or frame-level disagreement analysis.
  Ten focused report/suite tests passed. `recording-head-results.json` under the
  October 30 suite root records the context comparison: winner payload AP is
  0.007263 on fuzzing, 0.026109 on double and 0.007340 on force-neutral, versus
  classifier AP 0.002617, 0.006380 and 0.035643 respectively. These are diagnostic
  head comparisons, not measured improvements from fusion or retraining.
- [ ] Produce a source-only missed-attack report from the saved epoch-25 scores:
  per-recording/per-reviewed-family AP, pooled non-DoS AP, score distributions,
  support/first-seen rates and classifier/payload disagreement.
- [x] Include unknown labels, missing payload support and all evaluated rows in
  coverage accounting; unsupported rows must not disappear from denominators.
- [ ] Audit weak captures' reviewed labels, timestamps and timing support before
  inferring a model defect. Do not change labels based on the model's mistakes.
- [ ] Freeze a repeat-run recipe, then evaluate seeds 43 and 44 with the same
  source exposure/budget after the cleanup gate passes. No new full fits yet.

#### November 1 focused failure report

[`scripts/event_failure_report.py`](scripts/event_failure_report.py) completes
the diagnosis for Silverado double-4, force-neutral-4 and fuzzing-4, using the
saved pilot predictions and original checkpoint timing buffers. CPU job
**56924611** passed in **1m20s**, with no GPU or neural training. All 12,758,965
validation rows across eleven captures match reviewed source-cache timestamps,
IDs, labels and payload support; all paired arms preserve the payload scores.
The three captures have zero unknown labels; twelve fuzzing positives lack
payload support and remain in budgeted-recall denominators. Seven focused tests
and the **140-test CPU suite** pass. The complete family/non-DoS report and an
independent ground-truth label audit remain unfinished.

Findings change the immediate priority:

- Fuzzing has no matching training capture. **163/172 attack frames use IDs
  absent from that vehicle's normal training capture; 159 are first occurrences.**
  Only twelve have eligible preceding same-ID byte pairs. Per-ID recurrent memory
  has little history to exploit for most of this capture. All 795,285 negative
  frames here have training-normal IDs; an ID-novelty control still needs checking
  on the remaining validation/normal captures before claiming a useful detector.
- Force-neutral has only 32 matching training positives, but **all 56 validation
  positives have timing support and valid payloads**. Vehicle-scoped byte novelty
  occurs on all 56 and disappears under cross-vehicle pooling. This is sensitivity
  evidence, not proof that pooling causes the model failure.
- Double has 296 matching training positives. Its validation positives split
  equally between two IDs; only half have fitted timing support. Payload AP is
  0.026109 versus classifier AP 0.006380, on identical rows.
- At a **0.01% empirical normal-frame FPR budget**, original classifier recall
  is 1.97% / 46.43% / 9.30% on double / force-neutral / fuzzing, with
  414 / 478 / 3,779 false-positive frames inside the corresponding attack captures.
  Payload recall is 0% / 14.29% / 3.49%. At 0.1%, payload improves double recall
  to 44.10% with 2,011 capture false positives, versus classifier 10.39% with
  4,457. There is no uniform fusion benefit. These thresholds use the same
  3,248,139 normal selection-validation frames and are not production calibration.

The November 2 follow-up below evaluates a vehicle-scoped training-normal
ID-novelty control across all source validation captures and a causal same-ID-gap
control on supported known IDs. Continue deferring the per-ID GRU until the
remaining known-ID failure analysis justifies learned local history. Keep the
original candidate and the current preset.

The [dated evidence note](docs/empirical-notes/2026-11-01-151725-EST-weak-capture-failures.md)
records interpretation, matched-budget tables, source/clock limits and reproduction.
Machine-readable and generated Markdown reports, source/config/score hashes,
reviewed clock metadata and `artifact-audit.json` are under
`/fs/ess/PAS1266/graphids/dev/rf15/experimentruns/event-failure-report-20261101T201147Z/`.
All original saved metrics on the focused captures reproduce exactly; the audit
checks coverage, support partitions, disagreement counts and empirical budgets.

#### November 2 causal controls: two distinct, measurable blind spots

[`scripts/event_causal_controls.py`](scripts/event_causal_controls.py) fits
vehicle-scoped ID sets and robust timing tables only on the two verified-normal
training queries. CPU job **56946541** completed in **2m01s**, with no GPU or
neural execution. All **12,758,965 validation rows** align with all three saved
pilot arms; **112 original classifier/payload metric values reproduce exactly**.
Six new tests and the **146-test CPU suite** pass; the cluster gate passed all
13 control/failure-report tests. Per-frame control scores, support/startup masks,
frozen profiles and provenance are saved separately from the selected model.

- The fixed ID-novelty alarm detects **163/172 fuzzing positives (94.77%)**,
  including 159 first occurrences. It produces **zero false-positive frames on
  all 12,722,481 validation negatives**, including both normal captures and
  recording starts. Other attacks use training-normal IDs and receive no novelty
  detections. This establishes a source-cohort blind spot, not future normal-ID
  coverage, a zero-shot detector, or a validated production fusion policy.
- At the **0.01% empirical normal-frame FPR budget**, gap detects **177/356
  double positives (49.72%)**, versus original **7/356 (1.97%)**. Capture false
  positives are 545 versus 414. Gap abstains on 178 positives with insufficient
  sampled timing support; these remain misses. On identical gap-supported rows,
  gap AP is **0.627940**, versus original **0.012035**. The whole-capture
  classifier AP 0.006380 has a different denominator.
- Force-neutral remains weak: gap detects **4/56**, versus classifier **26/56**,
  with 530 versus 478 capture false positives at that budget. All 56 positives
  have local timing support; novelty detects none. More timing alone is not a
  uniform remedy. RPM also retains complementary payload signal: 361/374
  payload detections versus classifier 201/374 and gap 139/374, with capture
  false positives 619/260/189 respectively.
- Gap AP is 0.781547 on 12,713,206 supported known rows (36,141 positives).
  Original AP remains 0.925671 on all known rows (36,484 positives). Support
  differs, so these are not a matched aggregate ranking comparison. The report
  supplies identical-row comparisons within each capture's gap-support stratum.
  At 0.01%, gap and neural heads each use 324 normal-validation false positives;
  the binary novelty score uses none because ties cannot spend a continuous budget.

The payload follow-up below completes the smoothed vehicle-scoped byte-transition
control with sparse-history backoff. Compare timing scope and clipping on
identical supported rows before attributing double's gap advantage to new
recurrent capacity or proposing a normalization change. Local control tables and
the model's pooled checkpoint tables differ; these experiments do not isolate
that cause. Complete reviewed-family/non-DoS reporting and independent
confirmation/calibration grouping before promoting a hybrid or another neural
architecture.

Evidence root:
`/fs/ess/PAS1266/graphids/dev/rf15/experimentruns/event-causal-controls-20261102T173642Z/`.
`analysis/report.json`, `analysis/report.md` and `analysis/controls-*.npz`
retain results and all frame scores/masks. The
[dated note](docs/empirical-notes/2026-11-02-causal-source-controls.md) records
matched-denominator comparisons, protocol limits and reproduction.

#### November 2 payload controls: history helps ranking, weak captures remain

[`scripts/event_payload_control.py`](scripts/event_payload_control.py) adds
vehicle/ID/position-scoped marginal and preceding-byte conditional distributions,
enabled by `scripts.event_causal_controls --payload-transition`. Marginal counts
have uniform prior mass one; transition counts blend toward the marginal with
prior mass 32. These values were fixed before evaluation. Both controls require
normal-training support for every valid current byte position, share the same
scored rows and report mean byte NLL. Missing preceding positions and unseen
contexts use the marginal. Query observations advance history but never counts.

CPU job **56946799** completed in **4m33s**, including **152 passing CPU tests**
(24 compute/CUDA cases deselected), the full validation comparison and the
artifact audit. It fits 19,765,954 normal training bytes and 19,764,995 preceding
byte pairs, preserving 858,925 sparse transition states. All 12,758,965 physical
validation rows align; all earlier classifier/payload/ID/gap metrics and budgets
reproduce exactly, including the 112 original classifier/payload metric values.
There is no GPU or neural execution.

| Validation capture | Marginal AP | Transition AP | Original classifier AP | Neural payload AP |
| --- | ---: | ---: | ---: | ---: |
| Impala force-neutral-2 | 0.722467 | 0.921582 | 0.778327 | 0.729704 |
| Impala rpm-2 | 0.072947 | 0.515462 | 0.309721 | 0.455335 |
| Silverado force-neutral-4 | 0.000345 | 0.003559 | 0.035643 | 0.007340 |
| Silverado double-4 | 0.006186 | 0.007894 | 0.006380 | 0.026109 |

These four captures have identical support for all four columns. Transition
history adds useful ranking beyond frequencies on some captures but does not
repair the weak Silverado pair. On exactly matched full control support,
aggregate AP increases from 0.019949 to 0.026972 and recording macro AP from
0.284132 to 0.430540. These controls remain poor aggregate detectors; their
12,758,798 supported rows exclude 167 fuzzing positives. Do not compare those
aggregate APs directly with the full-support classifier as if coverage matched.

At the **0.01% allowed empirical normal-frame FPR**, tied scores leave most of
the 324-frame allowance unused: marginal achieves **zero** normal FP and
transition **two**, versus 324 for the neural heads. Neither byte control detects
any force-neutral-4 or double-4 positive at this allowance. At 0.1%, achieved
normal FP are 3,066 and 3,088 (allowance 3,248), but those captures still have zero
detections and 2,710/11,425 capture FP. Improved AP does not establish useful
recall at the current pooled operating points.

The byte controls support only **5/172 fuzzing positives**, versus 9 on known
IDs and 160 with neural payload support. Transition AP 0.320196 versus marginal
0.000298 on these five positives is not a whole-fuzzing result. Original and
neural payload AP on exactly those rows are 0.062864 and 0.091614. The existing
ID-novelty control remains the relevant diagnosis for most fuzzing positives.

**Next: inspect normal-score tail ties and per-ID score scales, and isolate the
timing scope/clipping difference on identical rows.** Keep calibration/confirmation
grouping separate from this reused selection cohort. These simple one-byte
conditional controls do not establish that a GRU will fix force-neutral or
double, nor rule out richer multibyte/longer-history learning. Retain the selected
model and avoid automatic fusion or a larger fit from this result alone.

Evidence root:
`/fs/ess/PAS1266/graphids/dev/rf15/experimentruns/event-payload-controls-20261102T175023Z/`.
`analysis/report.json`, `analysis/report.md`, eleven control-score archives,
two sparse count archives and `artifact-audit.json` retain the complete evidence.
The [dated note](docs/empirical-notes/2026-11-02-payload-transition-controls.md)
records formulas, coverage, all capture APs, operating-point limits and reproduction.

Use `graphids.evaluation.iter_score_chunks` and the packed indices for diagnosis;
never read reserved target recordings. Any thresholds swept on the existing
selection scores are diagnostics. Do not use them to certify final operating
points. Report variability by independent capture/session and by seed; an IID
frame bootstrap would exaggerate certainty. Two source vehicles cannot support a
strong population-wide cross-vehicle claim.

The current validation cohort has already selected architecture and epochs.
Reserve new independent confirmation captures before reporting a final improvement.
Add positive-capture macro AP and reviewed-family results to reporting first,
without silently changing the historical aggregate-AP checkpoint selection rule.

### P1. Establish how much timing and payload scores explain without new architecture

- [x] Fit a lightweight same-ID-gap detector on verified-normal training data,
  with declared support and fallback; compare on identical saved-score cohorts.
  November 2 control uses vehicle-local tables, 32 sampled pairs per ID, frozen
  training clipping and explicit abstention for first/unsupported occurrences.
  All frames remain in budget denominators; matched-support neural AP is reported.
- [x] Fit a smoothed per-ID/byte-position categorical transition baseline with
  backoff for sparse transitions. Preserve byte-validity and preceding-only access.
  November 2 paired marginal/transition controls share full-byte support, retain
  unsupported frames in budget denominators and preserve all sparse counts.
- [ ] Evaluate the existing classifier and payload head separately, including
  non-DoS and weak groups, before fitting a combination.

These are diagnostic controls, not a restoration of retired neural feature flags.
A neural architecture should demonstrate incremental value over simple causal
controls. Keep vehicle-scoped and source-transferable ID assumptions explicit;
identical numeric IDs across vehicles are not ground-truth semantic matches.

### P2. Reduce recording/family dominance before enlarging the model

The **paired frozen-encoder head experiment** is implemented in
[`scripts/event_classifier_pilot.py`](scripts/event_classifier_pilot.py), using only causal training
representations: natural objective versus capped recording/group-balanced
objective with the same head initialization, update count and exposure. This
separates a classifier problem from a representation problem. Do not train or
select the new head on the validation predictions used to assess it.

Pilot recipe: load the verified epoch-25 per-ID-gap checkpoint, freeze its entire
state and disable dropout, then initialize two copies of its 129-parameter linear
classifier. Train for **five head epochs**, AdamW LR 0.001 and weight decay 0.01,
using the original chronological loader and optimizer windows. Both heads consume
the same detached representation from each forward pass. Only positive frames
receive recording weights; negatives retain weight one and unknowns are masked.
Training-only positive counts determine capped equal-mass weights (maximum 10),
with the total positive weight preserved. This balances recordings, not reviewed
attack families, and does not use the historical global positive weight 46.66.

Evaluate the original and both final adapted heads on the full existing validation
cohort, using one common frozen forward pass and identical payload support/scores.
Report aggregate and recording AP, recording macro AP and diagnostic confusion
metrics at 0.5. Select no epoch or threshold on this cohort. Save full predictions,
training coverage/weights, source/config/dependency provenance and head-only
artifacts bound to the base checkpoint SHA. These artifacts represent a separate
adaptation stage, not a resumable full-model checkpoint. The production preset
and model implementation remain unchanged.

```bash
# Inside one single-A100 Slurm allocation; OUTPUT must not exist.
python -m scripts.event_classifier_pilot --config WINNER.yml \
  --checkpoint WINNER.ckpt --output OUTPUT
```

Eight synthetic CPU tests cover capped positive mass, training-only weights,
unknown-label masking, window normalization, paired cap-one identity, causal
readout parity, frozen parameters/buffers, exact row/tail coverage and hook cleanup.
The full CPU suite passed **133 tests** (24 compute/CUDA cases deselected).
Ruff, documentation lint and the strict documentation build passed.

#### November 1 pilot result: retain the original candidate

One A100 job, **8251729**, completed successfully in **2m34s**, including the eight
focused tests. Each head received 660 updates across five complete exposures of
8,526,701 training frames. Evaluation covered all 12,758,965 validation frames.

| Final head | Aggregate AP | Recording macro AP | Recall at 0.5 | False-positive frames at 0.5 |
| --- | ---: | ---: | ---: | ---: |
| Original epoch-25 classifier | 0.925671 | 0.504552 | 0.893186 | 3,699 |
| Ordinary BCE adaptation | 0.933404 | 0.489161 | 0.969493 | 3,744 |
| Capped recording-positive adaptation | 0.927782 | 0.505310 | 0.961956 | 3,741 |

The balanced head barely changes macro AP relative to the original (+0.000757).
It does not fix the weakest Silverado captures: double AP falls from 0.006380 to
0.003352 and force-neutral from 0.035643 to 0.018150; both still have zero recall
at 0.5. Fuzzing AP changes from 0.002617 to 0.002665, with unchanged 1.16% recall.
The all-normal Silverado capture has five false-positive frames versus three
originally; Impala remains zero. Ordinary adaptation improves aggregate ranking
but reduces recording macro AP. **Neither adapted head replaces the original.**
Do not launch an end-to-end weighted fit on the strength of this result. This
single fixed-budget/cap pilot does not establish that all head adaptation or
balancing fails, or prove a representation defect.

The artifact audit exactly reproduced **all original aggregate and per-recording
metrics**, verified identical payload metrics and validation coverage across
heads, preserved positive mass at 178,904 with final weights capped at ten,
confirmed unchanged frozen parameters/buffers and base checkpoint bytes, and
reloaded both head artifacts successfully. This also extends the cleanup's
checkpoint evidence to full-cohort metric reproduction, not full-frame bitwise
prediction equality or another end-to-end training run.

Evidence root:
`/fs/ess/PAS1266/graphids/dev/rf15/experimentruns/event-classifier-pilot-20261101T194504Z/`.
`pilot/pilot.json` contains the complete results, counts, weights and provenance;
`pilot/scores/{original,ordinary,balanced}/` holds full predictions;
`pilot/classifier-heads.pt`, `artifact-audit.json`, `submission.json` and
`pilot-8251729.log` retain the artifacts and verification evidence. Runtime source,
manifest, config and checkpoint hashes passed before execution. No target files
were opened and no threshold was fitted.

If a future balanced-head pilot improves weak groups convincingly, test that
single change in end-to-end training.
Implement weights in `EventTrainingWindow` metadata and `_loss_sums` /
`training_step`, not by dropping events or breaking chronological streams.
Freeze capped group weights from training data and normalize their mean over
known training rows to one. For an optimizer window, use
`sum(weight_i * BCE_i) / known_target_count`, with the denominator covering the
whole window and its actual tail. Do **not** divide by the sum of weights inside
each window: with one stream and mostly single-recording windows, that would
cancel a constant recording weight and defeat the intervention. Keep a separate
eligible normal-byte denominator for the payload loss. Verify the objective
against an independent serial calculation, including single-group windows.

Recording identity is already explicit. Attack-family labels need a reviewed
training metadata contract; `Recording` currently has no family field. Do not
infer families from filenames inside model execution. Version manifest/cache
identity deliberately if metadata changes. Count independent captures as well as
frames, cap rare-group weights, and keep normal exposure represented. Group DRO
and focal/effective-count loss are follow-ups only if simple balance warrants them.

A shuffled recording schedule is another separate hypothesis. If tested, permute
recording assignment deterministically from `(seed, epoch)`, preserving every
recording's chronology, independent state and exact tail coverage. Persistent
workers must receive the actual epoch schedule; update checkpoint data identity
and verify resumed versus uninterrupted optimizer/RNG equivalence.

### P3. Learn local per-ID payload history alongside the retained global model

This is the next architecture experiment if diagnostics show missing temporal
payload/context information. The current graph stores preceding **global-encoder
states** at each ID; it is not a recurrent encoder over that ID's own sequence.
The gap result motivates testing local history but does not prove recurrence will
help or that the gain is more than a timing shortcut.

Extend `EventGraphMambaState` with a per-recording bank updated by one shared small
GRU or state-space cell. Retain raw observations and the selected timing input.
Use shared parameters across IDs, zero initialization and a declared unseen-ID
behavior; avoid a fixed learned vocabulary that assumes cross-vehicle ID semantics.

For each event:

1. Read preceding local and global states.
2. Forecast the current payload from those states and the current header.
3. Encode the current raw observation and update only its ID's local state.
4. Update global context and classify using current local/global representations
   and the existing causal graph aggregate.
5. Commit state for subsequent events.

Repeated same-ID occurrences inside a chunk must update sequentially. Forecasts
must not see the byte they predict. Keep recording resets, FP64 time checks,
chunk-boundary detachment, explicit memory limits and no silent history eviction.
Use a new checkpoint/state identity and explicit weight initialization; never
relax strict resume checks to accept a changed architecture.

Begin with an independent serial output/state/gradient oracle. Compare this one
mechanism against the retained candidate at matched exposure, seeds and optimizer
budget. Add a parameter-matched candidate control if capacity grows materially.
Measure preprocessing, state memory and latency before optimizing segmented scans.
Do not reintroduce the retired graph-age bias as part of this change.

### P4. Improve payload learning only where diagnostics justify it

The selected payload head already has aggregate AP 0.7344 / AUC 0.9953; do not
assume it is broken because the old baseline was poor. Inspect byte/ID/vehicle
normal-score distributions, counters, high-entropy bytes and weak-family overlap.
The classification-only baseline control did not isolate payload utility on the
selected model.

Log separate classifier/payload gradient norms and cosine on shared parameters
at representative training windows. If sustained conflict appears, test one
loss-schedule or gradient-balancing intervention; PCGrad is a conditional candidate,
not an automatic addition. If normal coverage limits representation learning,
compare normal-only causal pretraining followed by supervised fine-tuning with
an equal-exposure joint-training control. Only two training captures are currently
verified normal, so collecting more independently reviewed normal driving modes
may matter more than adding layers.

Pretraining may use only explicitly approved training roles. Save its stage,
objective and manifest identity; fine-tuning loads compatible encoder weights into
a fresh optimizer. It is not ordinary resume with a changed training population.
Additional normal reference captures need a reviewed protocol/manifest revision.
A later autoregressive byte head may model same-payload dependencies, but must
mask the target and later bytes. Offline masked reconstruction and causal
forecasting are different tasks and must be evaluated as such.

For optimization stability, inspect full loss/gradient curves before extending
training. A conservative learning-rate decay pilot is separate from family balance
or architecture changes and needs equal-step controls. The epoch-24 AP dip does
not by itself identify learning rate, overfitting or a numerical defect as its cause.

### P5. Calibrate operating points and test complementary heads independently

Reserve source calibration and independent assessment captures, grouped by
recording/session and content identity. The current role enum has source
`train_*` / `val_*` and target roles, but no explicit source calibration/final-test
roles. Add a reviewed evaluation-role contract and evaluator without expanding the
fit loader's training population. `target_calibration` is not a substitute for a
source development split. Do not repurpose already-used validation as untouched test.

Temperature scaling can improve probability calibration, but a positive global
rescaling is monotone: it preserves ranking/AP and the 0.5 decision boundary.
It cannot repair poor weak-family ranking. Choose thresholds on calibration data
under the declared operating requirement, then freeze them for assessment.

Compare classifier and payload scores alone before any fixed or learned fusion.
Fit a combiner only on separate development or recording-grouped out-of-fold
predictions, with source-only normal-score fallback for unsupported IDs. A union
changes the false-alert rate; an AND rule may veto payload-only attacks. Either
requires fresh calibration using the actual fused scores and alert logic.

Declare alert windows, persistence/merging and refractory intervals. Report
false-positive frames, merged false alerts/hour and the measured normal hours.
No alert budget has yet been chosen; example rates are not existing acceptance
criteria. Quantile normalization does not guarantee future error rates under shift.
Trusted incident intervals are required for incident recall and detection delay;
frame-origin labels do not automatically supply them. No point adjustment that
credits an entire attack interval after one detected frame.

## 5. Acceptance and the next small experiment batch

Before another full fit:

- [x] Complete cleanup CUDA verification, including historical checkpoint parity.
- [ ] Complete the saved-winner diagnostic report.
- [ ] Freeze selection/calibration/confirmation grouping and the metric rule.
- [ ] Reuse the verified winner as the reference; confirm seeds before promoting
  a new mechanism. Keep each mechanism a separate, reviewable experiment.

For every proposed change, preserve source/config/dependency hashes, manifest
identity, seed, prediction coverage and support flags. Test changed causality,
normal-only fitting, loss denominators, stream isolation, tails, fitted-buffer
loading and exact epoch-boundary resume. Report per-capture/per-family AP,
weak-group recall at a frozen operating point, aggregate metrics and runtime.
Only claim improved recall at a common false-alert budget after calibration and
independent assessment exist. Do not invent a universal improvement percentage.

The next order is: **complete known-ID payload/timing controls and saved-score
diagnosis; establish independent confirmation grouping and repeat the retained
candidate; then select one neural intervention if the evidence warrants it.**
The completed frozen-head balance pilot has not earned an end-to-end weighted
fit. Calibration design and normal-data collection can proceed alongside
diagnosis. Keep larger models,
more epochs, bidirectional windows, synthetic attacks, target adaptation and
suppression timers deferred until their specific data/protocol requirements are
met. Do not recreate an eleven-arm standing suite to pursue these ideas.
