# CAN event model: improvement checklist and implementation plan

Updated: 2026-10-27. Scope: improve detection quality in the active
`can_event_dm` / `event_graph_mamba` pipeline. This document records the research
recommendations and proposed implementations; unchecked items are not implemented.
It does not authorize a claim of improved accuracy before the experiments finish.

## 1. Evidence and constraints

The latest full fit was Ascend job **7894453**, completed on October 16 after
25 epochs and 13,150 optimizer steps. Its source validation cohort contains
12,758,965 frames, including 36,484 attack frames (0.286%).

| Validation metric | Best AP checkpoint, epoch 20 | Final checkpoint, epoch 25 |
| --- | ---: | ---: |
| Classification AP | 0.1436 | 0.0737 |
| Classification ROC-AUC | 0.9160 | 0.9353 |
| Recall at threshold 0.5 | 3.43% | 0% |
| F1 at threshold 0.5 | 0.0664 | 0 |
| True positives / false positives at 0.5 | 1,252 / 0 | 0 / 0 |
| Payload anomaly ROC-AUC | 0.2754 | 0.2067 |

The best validation-F1 threshold on epoch 20's own predictions is approximately
0.00104: precision 72.2%, recall 13.3%, F1 0.2246. This is a diagnostic selected
on the same validation data, not an independently evaluated operating point.
Accuracy from predicting every frame normal is already 99.714%.

Training has 8,526,701 labeled frames and 178,904 positives. About 93.2% of
positives are DoS. Ten query recordings include only two verified-normal
recordings. Frame count therefore overstates independent driving-state coverage.

The previous `train/payload=0` reports were final-step values: that step used an
attack recording where the payload objective was masked. They do not show that
the payload branch received no training.

Preserve the established [raw-message input requirement](docs/reference/graph-mamba-events.md):
every observed ID, payload, length and timestamp must reach the model, in order.
Derived timing/difference features below are optional ablations alongside raw
inputs, not replacements for them. Learned per-ID context is the main architectural
direction. Keep unknown labels unknown and all physical rows intact.

### Current implementation status

- [x] Add full-epoch classification/payload loss totals with separate target counts.
- [x] Publish those totals before Ray's epoch-end reporting callback.
- [x] Add configurable `positive_weight` (default 1) to frame BCE.
- [x] Add aggregate ROC-AUC, confusion counts, precision, recall, F1 and FPR.
- [x] Save per-recording metrics in validation `index.json`.
- [x] Generate four matched weighting/interleaving configurations.
- [x] Validate CPU behavior: 81 tests passed, 15 CUDA cases skipped; final focused
  regression run passed 12 tests. These checks do not establish detection gains.
- [x] Complete CUDA checks for the new loss/reporting changes (October 23 gate 8019984).
- [x] Add an optional matched classification-only control (`payload_weight=0`).
- [ ] Run the four training comparisons; submitted behind the CUDA gate on October 23.
- [ ] Complete the model-quality experiments below.

## 2. Define the prediction task before optimizing it

- [ ] Keep attacker-frame attribution and contextual anomaly detection as separate tasks.
- [ ] Define an alert aggregation rule and false-alert budget before tuning thresholds.
- [ ] Obtain independently trusted attack intervals before claiming incident recall
  or detection delay.
- [ ] Keep unseen-vehicle and unseen-attack evaluations separate.

An injected frame can trigger anomalous legitimate responses. Detecting those
responses may be useful operationally while counting as a false positive under
attacker-origin labels. Do not silently relabel every frame in an attack capture
as malicious, infer trusted intervals from sparse origin labels, or reinterpret a
positive time bin as proof of an unsafe vehicle state.

Maintain three explicit evaluation products:

1. **Frame attribution:** existing known origin labels; one score per observed row.
2. **Time-window detection:** a declared aggregation of frame scores and a declared
   window label rule. Report it as that proxy unless independently annotated.
3. **Incident detection:** trusted intervals, matching rules and detection delay;
   unavailable when interval ground truth is unavailable.

The same observable ID/payload/history can sometimes arise from legitimate and
malicious senders. If the available observations are identical, exact sender
authenticity cannot be inferred from them alone. This is an information limit,
not evidence that every attack in this cohort is indistinguishable.

## 3. Finish the training controls

- [ ] Run ordinary BCE versus weighted BCE, each with one and four streams.
- [ ] Compare joint training with a matched one-stream, unweighted
  classification-only control (`payload_weight=0`); this tests the effect of the
  auxiliary objective, not the value of a separately pretrained encoder.
- [ ] Add recording/family-balanced objective contributions as a separate ablation.
- [ ] Randomize recording assignment per epoch while preserving each recording's chronology.
- [ ] Repeat promising comparisons with seeds 42, 43 and 44.

### Existing four-way comparison

[`scripts/event_ablation.py`](scripts/event_ablation.py) generates four 25-epoch
experiments. All use 16,384-event chunks, a 65,536-event effective optimizer
window, seed 42, explicit checkpointing and 8,388,608 graph tokens. Only stream
count and positive weighting differ. One stream accumulates chronological chunks;
four streams interleave independent recordings. Neither drops or duplicates rows.

```bash
.venv/bin/python scripts/event_ablation.py /path/to/new/comparison \
  --positive-weight 46.660762196485265
gx exp submit /path/to/new/comparison/event_ablation_streams1_weighted0_seed42.yml
```

Submit each generated YAML separately. The script refuses to overwrite existing
files. Give repeated suites distinct experiment names as well as output folders.
Prepared configs are under
`/fs/ess/PAS1266/graphids/dev/rf15/experimentruns/event-quality-controls-20261023T174455Z/comparison/`.

Use `--classification-only` to generate an additional one-stream, unweighted arm
with `payload_weight=0`; the original four arms remain unchanged. Its payload
scores are untrained diagnostics and must not enter detector fusion or checkpoint
selection.

### October 23 execution

Suite artifacts are under
`/fs/ess/PAS1266/graphids/dev/rf15/experimentruns/event-quality-controls-20261023T174455Z/`.
The suite preserves a working-source snapshot, file hashes, the tracked diff,
dependency versions, expanded configurations and the reviewed training manifest.
Recounting the ten current training caches confirms 178,904 positive and 8,347,797
negative frames, giving `positive_weight=46.660762196485265`.

| Arm | Ascend job |
| --- | --- |
| CUDA correctness, bounded one/four-stream weighted parity, public Ray smoke | 8019984 |
| One stream, ordinary BCE, joint payload objective | 8020093 |
| One stream, ordinary BCE, classification only | 8020097 |
| One stream, weighted BCE, joint payload objective | 8020100 |
| Four streams, ordinary BCE, joint payload objective | 8020104 |
| Four streams, weighted BCE, joint payload objective | 8020107 |

All five fits require successful completion of job 8019984 (`afterok`); an invalid
dependency cancels them. Each fit has a unique experiment name, 25 epochs and the
same chunk/optimizer-window controls described above. `suite.json` records job
identities and paths. Submission is not completion or evidence of improved model
quality. CPU regression job 56603752 on Pitzer passed 65 tests with 18 CUDA skips.
The first CUDA attempt (8019139) passed 125 tests but failed two new assertions
comparing CPU-reported Lightning metrics with CUDA expectations. The assertions
now compare on CPU; dependent fits were automatically cancelled and resubmitted.
The failed attempt is retained under `gate-attempt1/`. Corrected gate 8019984
passed all 127 tests, independent weighted output/state/gradient parity for one
and four streams, and the public Ray/Lightning smoke fit. The bounded benchmark
used one repetition and 163,840 scored events per arm; it is a correctness/runtime
check, not a model-quality comparison or a new production-throughput selection.

[`scripts/event_ablation_report.py`](scripts/event_ablation_report.py) reads
`suite.json` and complete epoch indices to generate `comparison-results.json`.
It reports best-AP and final epochs, the learning curve, and per-recording metrics
at the single selected epoch. A complete comparison requires finished fits with
all expected epochs and matching physical-row/known-label populations. It does
not fit thresholds or establish alert-budget performance.
Report job 8020403 runs after all five fits terminate (`afterany`), including
failure cases; the report remains incomplete unless every fit finishes all
25 epochs with comparable evaluation populations. Four synthetic report checks
passed for checkpoint selection, population mismatches and incomplete fits.

**Terminal status audit:** all five fits failed. Baseline 8020093 completed nine
validation epochs, then failed while writing epoch-10 score shards with
`OSError: [Errno 122] Disk quota exceeded`. ESS reports 100,000/100,000 inodes
used despite approximately 426 GiB of available space. Its last checkpoint
survives and matches its saved SHA256. The other four fits and report job failed
within one to three seconds with exit `0:53`, without application logs; the exact
startup errors are unavailable. Scheduler failure also prevented the baseline's
application manifest from being updated, leaving a stale `running` status.
`suite.json` and `comparison-results.json` now include scheduler accounting and
mark the comparison incomplete. The partial baseline's best AP is 0.03494 at
epoch 6, with zero recall at the diagnostic 0.5 threshold. This is not a completed
25-epoch comparison. Resolve output file-count pressure before resuming the
baseline or rerunning the other arms; no jobs were resubmitted by this audit.

### Output layout correction

Artifact schema 2 now appends each epoch's complete frame predictions to one
`scores.bin`, with recording/source/chunk offsets, coverage and metrics in
`index.json`. FP64 values, physical row order and metric definitions are
unchanged. There are two persistent prediction files per epoch, regardless of
execution chunk count: 250 for the five 25-epoch fits instead of 98,125 NPZ shards
plus indices. Metric scratch mappings use local `TMPDIR`.

`scripts/pack_event_scores.py` migrates complete schema-1 epochs and optionally
removes the old shards only after checking source identity, exact coverage and
the packed-file SHA256. `iter_score_chunks` reads both formats. All 34 retained
epochs (25 from the October 16 fit and nine from the interrupted baseline) were
consolidated, preserving 433,804,810 score rows and their existing metrics.
Best/last checkpoint checksums and all 30 reviewed source-file paths were
verified afterward. No model checkpoint/input schema change is required.

CPU integration passed 73 tests with 18 CUDA skips. Gate 8021017 passed 137 tests,
weighted one/four-stream parity and the public Ray/Lightning smoke fit. These
checks validate the storage change; the five-arm quality comparison remains
incomplete. Cleanup inventories and verification receipts are under
`/users/PAS2022/rf15/graphids-cleanup-audits/20261023T182040Z/`.

Cleanup completed after byte-for-byte archive comparisons and archive SHA256
checks. Historical runs and stale Slurm/default-run outputs were archived outside
ESS before removal. Project loose-file count fell from 77,158 to 1,929, releasing
about 83.4 GiB on ESS; filesystem inode use fell from 100% to 20% (80,503 free).
`final-summary.json`, archive receipts and the original inventory record exact
paths and counts. Raw/shared cached data, databases, reviewed inputs, the active
event cache, the October 16 event baseline and the interrupted current suite were
retained. The interrupted suite's original source snapshot still documents its
schema-1 writer; retries must use a fresh source snapshot with the schema-2 writer.

The weight is negative/positive frame count from this training cohort only.
Recompute it for a different training manifest. The 65,536-event optimizer window
differs from the original run's 16,384; compare these four arms against each other
before attributing differences from the original run to weighting or interleaving.
The shared loss helper also applies the weight to validation BCE; compare AP and
operating-point metrics across arms rather than treating weighted/unweighted BCE
as identical objectives.

### Family balance and ordering

Implement optional recording/family weights in `EventTrainingWindow` metadata and
`EventGraphMamba.training_step`. Assign families from reviewed training metadata,
not a filename heuristic in model execution. Derive weights on training data only.
Use a declared recording-mean or family-mean loss, with denominators covering the
entire optimizer window; retain the existing normal-byte objective denominator.
Cap extreme weights and include an unweighted control. A global positive weight
does not change the relative balance among positive attack families.

For shuffled recording assignment, generate a deterministic permutation from
`(seed, epoch)` before scheduling streams. Keep rows chronological, process every
recording exactly once, and preserve exact tail coverage. Update persistent-worker
scheduling, checkpoint data identity and resume behavior together; changing a
parent dataset field alone does not update already-running worker copies.

**Checks:** exact row coverage, independent stream states, true objective totals,
reproducible schedules, and resumed-versus-uninterrupted optimizer/RNG equivalence.

## 4. First representation experiment: explicit local context

- [x] Implement the elapsed-recording-time ablation while preserving artifact timestamps.
- [x] Implement normalized bus timing alongside the existing raw timing input.
- [x] Add optional per-ID gaps and previous-payload differences.
- [x] Add explicit age to each graph memory entry.
- [ ] Evaluate individual changes before combining them.

### October 27 implementation

The features are opt-in through `EventGraphMambaConfig`: `use_elapsed_time`,
`timing_mode`, `use_id_gap`, `payload_context` (`none`, `delta`, `xor`, `both`),
and `graph_age`. Changed feature/state semantics require model schema 2; defaults
retain schema-1 behavior and checkpoint loading. Per-ID observation features
additionally require `can_event` representation schema 2, which prepares raw
predecessor metadata without changing the lossless on-disk cache schema.

Timing normalization fits only reviewed, verified-normal `train_query` recordings.
It uses at most 65,536 deterministic evenly spaced pairs per recording: median
positive gap as the `log1p` scale, log-space median/IQR, and a training-only 99.9th
percentile clip (at least the scale). Near-zero IQR falls back to unit spread.
Per-ID distributions require `timing_min_support` sampled pairs (default 32);
unsupported IDs use a pooled same-ID-gap distribution and an explicit support
indicator. First-seen rows have zero gap features and a separate first-seen mask.
Fitted tables are checkpoint buffers; evaluation never refits them. The numeric
ID tables pool reviewed source vehicles and do not assert semantic alignment.

Delta/XOR inputs compare only bytes present in both observations. They feed the
current observation encoder; the same-frame payload forecast consumes preceding
encoder states and current header/timing only. Graph attention subtracts
`softplus(age_decay) * log1p(age_seconds / graph_age_scale)` from valid preceding
node logits; physical timestamp subtraction remains FP64. All nodes and existing
capacity guards are retained.

`scripts/event_context_ablation.py OUTPUT --seed 42` generates seven configs:
baseline, no elapsed time, normalized bus gap, per-ID gap, payload delta,
payload XOR, and graph age. It preserves the base run's data, training budget,
and objective settings, and does not submit jobs. Repeat with seeds 43 and 44 in
separate directories. Correctness/smoke validation is separate from the pending
full held-out quality comparisons; no detection gain is claimed.

Validation on October 27 passed 120 full-suite tests on A100 and 98 CPU-selected
tests with CUDA hidden. The focused checks cover physical predecessor oracles,
tied clocks, variable byte validity, unseen-ID fallback, training-only fitting,
future/current-payload causality, independent CPU/CUDA output/state/gradient
parity, corrupted/incompatible checkpoints, and exact optimizer/RNG resume.
A combined-feature public Ray/Lightning GPU fit passed. A schema-1 checkpoint
created by the pre-refactor source produced identical outputs through the current
default model when checked on the same hardware. A bounded synthetic gate passed
one/two-stream parity and exact coverage of 32 rows per arm with identical initial
weights; its timings are not production-throughput evidence. Receipts are under
`/fs/ess/PAS1266/graphids/dev/rf15/experimentruns/event-context-validation-nwg6n_ik/`.
Ruff, documentation symbol checks, strict MkDocs, wheel building, and lock checks
also passed.

**Hypothesis:** the current global encoder spends capacity reconstructing local
timing and value relationships; elapsed time may also provide an attack-schedule
shortcut. Neither explanation is established by the current metrics.

Implementation locations:

- `graphids/data/events.py`: construct causal metadata and valid masks.
- `graphids/models/event_graph_mamba.py`: encode optional context and graph age.
- `graphids/data/config.py` / `graphids/models/config.py`: expose typed options.
- `graphids/data/representation.py`: version changed input semantics.

For event `t` with ID `i`, find the latest preceding observation of `i`, even across
a chunk boundary. Candidate auxiliary inputs are `t - last_time[i]`, byte-wise
payload differences and XOR bits, previous-byte validity, and a first-seen mask.
Compare bytes only where both observations contain a valid byte; missing/padded
bytes must not become observed zeros. Update the previous-observation state only
after constructing the current event's context.

Apply `log1p(gap / scale)` and robust scaling fitted on eligible training data.
Select/freeze clipping thresholds without looking at validation or target scores.
Use a global fallback and a support indicator for unseen IDs; do not fit an ID
normalizer from evaluation traffic in a source-only experiment. Keep FP64 physical
timestamps for subtraction and row-order checks before converting model features.

For graph node `j`, retain its last-update timestamp. Add a learned age embedding
or nonnegative learned decay bias to attention over preceding states. At event
`t`, age is `t - last_time[j]`; equal timestamps still follow physical row order.
Keep all valid nodes represented and retain the existing capacity checks.

Preserve the forecasting rule: the current payload cannot enter any path used to
predict that same payload. Current-payload differences are permissible for the
classifier only; forecasting can use prior differences and the current header.

**Checks:** chunked/full-stream feature equality, tied timestamps, first-seen IDs,
changing payload lengths, interleaved recordings, future-perturbation causality,
and no current-payload leakage into forecasts. Version checkpoint identity when
input dimensions or state semantics change.

**Decision:** promote a feature only if it improves held-out per-family recall at
the same false-alert budget across seeds. Record latency and memory changes too.

## 5. Establish simple causal baselines

- [ ] Fit a per-ID timing detector on verified-normal training recordings.
- [ ] Fit a per-ID/byte-position categorical transition detector.
- [ ] Compare optional shallow classifiers on the same causal observations.
- [ ] Keep all baseline outputs compatible with the frame artifact contract.

For timing, estimate an empirical gap distribution per supported ID, with a
declared minimum support and pooled fallback. Score unusually short/long gaps
against that training distribution. Keep unknown-ID detection as a distinct
signal rather than equating every unfamiliar ID with a confirmed attack.

For payloads, estimate smoothed distributions such as
`P(byte_t | ID, byte_position, previous_byte)` on verified-normal training rows.
Compare transition-conditioned and marginal distributions, and use backoff when
transition support is sparse. Do not treat zero origin labels inside attack
recordings as verified-normal training evidence.

Never pool identical numeric IDs across vehicles as though they necessarily have
the same meaning. Evaluate vehicle-scoped tables where vehicle identity is
explicitly available, and a separate source-only transferable baseline with a
declared unknown-ID fallback. A target-normal adaptation arm is a different
protocol with its own data allowance and untouched target test captures.

Persist counts, smoothing/backoff settings, fitted normalizers, manifest hashes
and prediction coverage. Evaluate every eligible row, including unsupported rows
with explicit support flags. Do not improve reported metrics by silently excluding
hard or unseen IDs.

**Decision:** require the neural model to demonstrate incremental value over these
baselines under the same splits and calibration, not just a higher training score.

## 6. Repair and diagnose the normal-payload objective

- [ ] Plot normal/attack payload scores by vehicle, ID and byte position.
- [ ] Compare normal-only pretraining with training both objectives jointly.
- [ ] Compare frozen, partially unfrozen and jointly optimized encoders.
- [ ] Compare mean-byte surprise against calibrated top-k/maximum surprise.
- [ ] Test a joint-byte or signal-aware head only after the simpler controls.

First inspect whether counters, high-entropy normal bytes, operating-mode changes,
or weak normal-state coverage dominate the score. Below-chance ROC-AUC does not
by itself justify reversing the score: such a reversal must be selected on
development data and evaluated independently.

Create a separate pretraining configuration using only reviewed, verified-normal
training captures. Additional normal reference captures may be included only with
an explicit training-role/provenance revision; validation/reference and target
captures do not become training data implicitly. Predict each payload from causal
preceding context and the current header, using valid-byte target counts.

Save a pretrained checkpoint with its training-manifest digest, stage and objective
identity. Add a deliberate pretrained-weight initialization path for supervised
fine-tuning: validate architecture, load encoder weights, and initialize a fresh
optimizer. Do not disguise a changed training manifest or objective as an ordinary
resume. Normal epoch-boundary resume should remain strict.

For the fine-tuning experiment, hold training data, supervised budget and evaluation
protocol fixed. Compare random initialization, pretrained/frozen encoder,
pretrained/trainable encoder, and existing joint training. Log per-objective
gradient norms on the shared input projection and temporal layers; scalar loss
magnitudes alone do not establish gradient interference.

The existing eight categorical byte distributions share preceding context but
do not condition on other current bytes. A later autoregressive byte head can
score `P(byte_b | prior_context, header, byte_0...byte_(b-1))`; its masks must
prevent access to the byte being predicted or later bytes. Preserve all raw bytes
and test short payloads. Signal-aware variants require reviewed decoding or
learned boundaries, with no assumed cross-vehicle semantic alignment.

**Checks:** no training gradients from attack/unknown-normality payload targets,
positive normal-byte target counts, forecast causality, masked-byte correctness,
and separate pretraining/fine-tuning optimizer identities.

## 7. Main architecture experiment: local recurrent state plus global context

- [ ] Add a shared per-ID recurrent encoder alongside the bus-level Mamba.
- [ ] Fuse local, global and age-aware graph context for classification.
- [ ] Forecast payloads from preceding local/global state and the current header.
- [ ] Compare local-only, global-only and combined models at comparable capacity.

Extend `EventGraphMambaState` with a per-recording bank of local states and
last-observed timestamps. Use one shared GRU or small state-space cell across IDs
initially; separate learned weights per numeric ID would complicate unseen-ID
behavior and could memorize vehicle identity.

For each arriving frame:

1. Read the preceding local state of its ID and preceding global state.
2. Produce the causal payload forecast from those states and the current header.
3. Encode the raw current observation and update the local state for that ID.
4. Update the global sequence encoder and read the preceding graph memories.
5. Classify using current local/global context and the graph aggregate.
6. Commit updated states and timestamps for the next frame.

Preserve independent banks across recordings, zero-initialize unseen IDs, detach
carried state at the existing chunk boundary, and reset at recording boundaries.
Repeated occurrences of the same ID inside a chunk must advance sequentially;
updating every occurrence from the same incoming state is incorrect.

Start with a serial correctness implementation and an independent test oracle.
Optimize later with segmented scans/grouped operations only if state, output and
gradient parity hold. Include state memory and sparse-ID behavior in the capacity
budget; do not silently evict history. This state/architecture change requires a
new checkpoint identity, not relaxed loading of incompatible checkpoints.

**Decision:** compare results at matched training exposure and report model size,
steps, throughput and latency. A larger combined model needs a parameter-matched
global-only control before attributing a gain to per-ID structure.

## 8. Calibrate and combine complementary detectors

- [ ] Reserve explicit calibration captures separate from model-selection captures.
- [ ] Calibrate each head's score against supported normal data.
- [ ] Compare each head alone, a fixed combination and a small learned combiner.
- [ ] Select thresholds against an alert-level budget, not frame F1 alone.

Persist per-head normal empirical score distributions, optionally stratified by
supported ID/vehicle context with a documented fallback. Convert raw scores to
comparable tail scores, for example `-log(max(1 - F_normal(score), epsilon))`,
with an explicit tie convention. Quantile normalization does not guarantee a
future false-positive rate under distribution shift.

Compare the classifier, timing and payload detectors individually before trying
a fixed maximum/weighted combination or a logistic combiner. Fit any combiner on
training/development predictions separated from its assessment set, using
recording-grouped out-of-fold predictions when data are scarce. Do not train a
combiner on the same held-out predictions used to report its final quality.

Define how threshold crossings become alerts: window width, persistence,
hysteresis, merging and refractory interval. Fit the final threshold against the
normal exposure using that exact alert rule. Maximum/union fusion changes the
false-alert rate and requires its own calibration.

Report both false-positive frames and merged false alerts per hour. Include normal
hours and counts; short exposure cannot certify a low future alert rate. Any
example budgets such as 1 or 10 alerts/hour are development choices to declare,
not acceptance requirements already established for this project.

## 9. Missing-message detection is a separate extension

- [ ] Define timer-driven predictions and trusted suppression evaluation data.
- [ ] Add expected-arrival state and a timer/heartbeat scoring interface.
- [ ] Test complete silence and individual-ID disappearance separately.

The current model scores arriving frames. Other traffic may expose a stale ID,
but no score is emitted during complete silence without an independent timer.
Maintain expected next-arrival distributions from normal training, evaluate
overdue IDs on timer ticks, and avoid declaring every nonperiodic ID overdue.
Preserve tick predictions separately from observed-frame artifacts: a timer tick
is not a fabricated CAN message and has no attacker-frame origin label.
Measure timer latency independently of GPU chunk throughput.

## 10. Evaluation and experiment acceptance

- [ ] Freeze reviewed capture groups and separate training, selection, calibration
  and final test roles before fitting.
- [ ] Keep source-only transfer and target-normal adaptation as distinct protocols.
- [ ] Run existing coverage, causality and resume checks for every changed state path.
- [ ] Report per-vehicle/per-family results and worst-group regressions.
- [ ] Use recording-level variability/uncertainty, not IID frame bootstrap claims.
- [ ] Archive reproducible configs, source hashes and complete prediction coverage.

Use the same untouched assessment cohort, threshold protocol and native prediction
unit for every comparison. If current validation has already guided several
architecture choices, report further gains there as exploratory; use new held-out
captures for confirmation when available. Large numbers of correlated frames do
not compensate for very few independent recordings.

For each run save:

| Artifact | Required contents |
| --- | --- |
| Run identity | Source revision plus dirty-tree hash/snapshot, config, seed, manifest digest, dependency versions |
| Coverage | Physical rows and eligible targets by recording, support flags, unknown-label counts |
| Learning curve | Epoch losses, target counts, per-family AP, checkpoint selection rule |
| Detection metrics | AP/AUC, TP/FP/FN/TN, precision/recall/F1 at frozen thresholds, per-group results |
| Alert metrics | Aggregation rule, threshold, false alerts, normal hours, supported incident metrics |
| Runtime | Peak memory, streaming latency, preprocessing/state-update cost and events/second |

Promote a change when held-out recall improves at the same declared false-alert
budget, results are consistent across seeds, and important families do not suffer
an unexplained regression. Report the actual tradeoff rather than inventing a
universal minimum improvement percentage. Check parameter/runtime controls when
the change adds capacity. Freeze final test access until development choices are
fixed.

## 11. Recommended execution order

1. Finish CUDA verification and the four existing weighting/interleaving controls.
2. Establish timing/byte baselines; ablate elapsed time and test explicit local context.
3. Diagnose/pretrain the payload branch and assess per-ID normal score calibration.
4. Implement shared per-ID recurrence with global-only and capacity-matched controls.
5. Calibrate/fuse complementary heads on separate development data.
6. Confirm on held-out recordings, then evaluate separately approved transfer tasks.
7. Add timer-driven suppression detection when its target and ground truth are ready.

Change one mechanism at a time before combining winners. Increasing width, depth
or epochs is a later capacity experiment, not the first response to the current
failure pattern.

## 12. Research basis and limitations

- [CANet: An Unsupervised Intrusion Detection System for High Dimensional CAN Bus Data](https://arxiv.org/html/1906.02492v1)
  motivates per-ID recurrent state and combined context. Its decoded-signal setup
  differs from this raw-message model; it does not establish a gain here.
- [CANShield: Signal-based Intrusion Detection for Controller Area Networks](https://arxiv.org/html/2205.01306v3)
  motivates signal relationships, multiple time scales and component-specific
  thresholds. Its signal-selection/decoding assumptions require separate review.
- [Time-Based CAN Intrusion Detection Benchmark](https://arxiv.org/abs/2101.05781)
  supports including lightweight timing controls for injection attacks. It does
  not establish that timing alone detects payload-preserving masquerade.
- [A comprehensive guide to CAN IDS data and introduction of the ROAD dataset](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0296879)
  explains attack/label distinctions and dataset limitations, including the need
  to assess realistic normal exposure and stealthier attacks.
- [can-train-and-test: A Curated CAN Dataset for Automotive Intrusion Detection](https://arxiv.org/html/2308.04972v1)
  distinguishes known/unknown vehicles and attacks. Its benchmark scores use a
  different evaluation setup and are not directly comparable with this run.
- [MIDS: Detecting Stealthy Masquerade and Tampering Attacks on CAN Bus via Bidirectional Mamba](https://arxiv.org/html/2606.18599v1)
  is a June 2026 preprint with bidirectional window processing and synthesized
  attacks. Its results do not demonstrate immediate causal frame attribution or
  zero-shot cross-vehicle transfer under this project's protocol. Account for
  window accumulation/lookahead if reproducing it.
- [Earlier local gear/RPM diagnosis](docs/empirical-notes/2026-10-16-missed-attack-diagnostics.md)
  demonstrates attribution/context differences and heterogeneous transfer on the
  earlier snapshot architecture. Do not present those probe gains as measurements
  from the current event model.

The architectural explanations above are testable hypotheses. The October 16 run
shows weak detection and unstable recording-level performance; it does not isolate
a single cause or guarantee that any proposed change will improve results.
