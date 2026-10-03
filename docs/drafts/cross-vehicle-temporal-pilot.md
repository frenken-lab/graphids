# Cross-vehicle temporal pilot

This pilot tests whether one supervised temporal detector learns behavior shared
across vehicle models. It does not claim that CAN IDs or payload byte positions
have common semantics. The source is `can-train-and-test-v1.5`; vehicle names
follow its README, not directory names. The published sets provide three usable
leave-one-vehicle-out folds: Silverado, Forester, and Traverse. Impala has no
corresponding unknown-vehicle attack test set, so it cannot be a fourth fold.
The Forester fold also holds out the Subaru manufacturer; the other folds hold
out a Chevrolet vehicle while another Chevrolet remains in training.

## Data contract

- Each fold trains on the `train_01` and `train_02` recordings of the other three
  vehicles. Whole files are assigned to source training or validation by a
  versioned recording manifest. Exact duplicate files are grouped by hash.
- A target vehicle's `train_01_attack_free` captures are split by file into
  normal calibration and normal test. Its `test_02` and `test_04` captures are
  attack tests. No target capture enters training or source validation. Both
  evaluation regimes permit the model's ordinary causal stream state and 64
  unscored warmup messages; neither fits parameters on the target. The
  zero-shot regime uses no target calibration statistics or thresholds.
- The builder rejects exact file-content overlap between source, calibration,
  normal test, and attack test captures. Partial overlaps or shared driving
  sessions cannot be ruled out from the published metadata; inspect these
  before a publication claim.
- Raw mode uses `(vehicle, CAN ID)` vocabulary entries fitted on source training
  only; target IDs map to unknown. Behavioral mode assigns local ID indices in
  first-seen order within each recording, with a fixed 8,192-ID capacity.
  The assignment is causal and collision-free. Neither mode performs semantic
  signal alignment.
- Behavioral inputs are log interarrival time, same-ID interarrival time,
  deviation from earlier same-ID periods, same-ID payload byte change fraction,
  and first-seen flag. They are computed within each recording. Absolute
  timestamps are converted to relative recording time before float32 packing.

## Experiment matrix

| Fold | Raw-ID baseline | Behavioral shared model |
| --- | --- | --- |
| Silverado held out | `silverado_raw_fit.yml` | `silverado_behavioral_fit.yml` |
| Forester held out | `forester_raw_fit.yml` | `forester_behavioral_fit.yml` |
| Traverse held out | `traverse_raw_fit.yml` | `traverse_behavioral_fit.yml` |

The fit YAMLs are in `configs/experiments/cross_vehicle/`. Both models use the
same supervised temporal hybrid backbone and per-ID memory. The behavioral
variant disables learned ID embeddings. These six runs test an ID-dependence
hypothesis. `silverado_impala_only_behavioral_fit.yml` is a single-source
comparison. The `source_vehicle` field can be set to each other training vehicle
to compare pooling with all three single-source alternatives without choosing
a source based on target test labels. Adapter comparisons come after this gate.

Validate a config with `gx exp config configs/experiments/cross_vehicle/silverado_behavioral_fit.yml`.
Launch a fit on a compute node with `gx exp submit <fit-yaml> -C pitzer`.
After it completes, run:

```bash
python -m scripts.cross_vehicle_evaluate \
  configs/experiments/cross_vehicle/silverado_behavioral_fit.yml \
  /path/to/checkpoints/best_model.ckpt \
  /path/to/cross_vehicle_report.json
```

The evaluator freezes a threshold from benign source validation scores for
zero-shot scoring. For normal-target adaptation, it freezes a second threshold
from separate benign target calibration captures. Neither run updates model
weights. It reports per-test AUPRC, recall and false-positive rate at the fixed
threshold, false alerts per hour using a one-second refractory interval, median
latency among detected attack episodes, and per-attack AUPRC/recall. An attack
episode groups positive frames within one second of the previous positive frame,
allowing benign frames between injections; the report includes detected and
total episode counts.
The threshold-based metrics in Lightning's ordinary test logs still choose
thresholds from test labels and are descriptive only; use the evaluator JSON
for deployment-style comparisons. Suppression remains unmeasurable with the
provided frame labels and is not included in this pilot.

## Training budget and baseline interpretation

### Revised pilot (v2; fits running)

The revised cross-vehicle source split uses whole content-grouped recordings and
targets 10% of source events per vehicle. It requires scored benign and attack
events in each vehicle's validation partition and attacked training data in
each vehicle. The versioned split manifest records hashes, class counts,
represented attack families, and actual proportions. At seed 42, the achieved
pooled validation fractions are 9.9936% (Silverado holdout), 9.9681%
(Forester holdout), and 9.9532% (Traverse holdout). The held-out vehicle's
calibration and test captures remain separate. Training now requests 1,024
segmented lanes and 32-event chunks. This changes optimizer-step count, so
results must be labeled as a new pilot. Phase metrics are written to
`.graphids/phase_metrics.json` for fits and alongside evaluator reports for
evaluation. A missing GPU sample appears as `null` with sample count zero.

CUDA graph replay exhausted a 16 GiB V100 during the first behavioral fit.
The failed attempts were archived under `runs/cross_vehicle_v2/failed_cuda_graph`.
The matched six-run v2 configuration disables CUDA graphs. A 40-batch V100
preflight completed for both variants with 1,024 lanes: raw peak allocated
memory was 5.44 GiB and behavioral peak was 10.38 GiB. These are preflight
measurements, not full-fit results.

Submitted v2 fit jobs: Silverado raw `55349133`, behavioral `55349139`;
Forester raw `55349140`, behavioral `55349141`; Traverse raw `55349144`,
behavioral `55349152`. Their paired evaluator jobs are `55349153`–`55349158`,
followed by comparison job `55349159` only after all six evaluations succeed.
The comparison will be written to
`/fs/ess/PAS1266/graphids/runs/cross_vehicle_v2/comparison.json`.
The complete lane plans contain 1,296, 1,368, and 1,152 optimizer steps per
epoch for the Silverado, Forester, and Traverse holdouts respectively; the
expected two-epoch totals are 2,592, 2,736, and 2,304. Full-run coverage and
actual global steps still require confirmation from the running jobs.

The measurements and decisions below describe the completed 256-lane baseline.

An exact section replay prototype is available through the lane loader's
`exact_replay` mode. It reconstructs each section's full causal state by
replaying its recording from the beginning, scores each event once, and matches
continuous scores in the small cross-vehicle regression test. For a recording
with `N` events and section width `W`, it processes approximately
`N²/(2W)` prefix events. This sequential reconstruction cost makes it
unsuitable for checkpoint selection or full target reports; those remain on
continuous evaluation pending a more efficient state snapshot implementation
and V100 lane sweep.

The six pooled fits run for **two complete epochs**, with a 12-hour limit per
job. Depending on the held-out vehicle, each epoch contains about 21–26 million
source training messages and 14–16 million source validation messages. The
shared supervised base config allows 20 epochs; the cross-vehicle config
overrides that limit to two to check the full data, training, and evaluation
pipeline at matched compute cost.

Training uses 256 segmented lanes with 32-event chunks, 2,048-event cores,
up to 256 unscored replay events per section, two ordered loader workers with
two prefetched batches, and CUDA graph capture for eligible batches. Source
validation and target evaluation remain continuous. The segmented training
path changes the number of optimizer updates and approximates long-running
state across section boundaries; interpret this pilot accordingly. The
Validation and target scoring use 32 continuous lanes with 256-event chunks,
matching the much smaller number of independent recordings without changing
the model's causal scoring order. Cached binary class and raw-ID vocabulary
sizes avoid scanning target test tensors during fit setup. The
256-lane training geometry passed the full-epoch throughput and correctness checks on
`set_01`. The 1,024-lane diagnostic improved throughput but reduced optimizer
updates and had worse bounded validation loss, so it is not selected here.

These runs are screening experiments, not fully trained quality baselines.
Two passes may undertrain the classifier, especially on rare attack labels, and
an apparent difference between raw and behavioral inputs may reflect different
learning speeds. The best checkpoint is selected by source validation loss,
but two epochs cannot establish that either model has converged.

Before claiming a strong baseline, inspect training and source validation loss
and per-attack performance across the pilot epochs. Then train raw and
behavioral models for a longer, matched budget with source-validation-based
checkpoint selection or early stopping, repeat with multiple seeds, and compare
against single-vehicle training. Keep the held-out vehicle's attack test labels
out of epoch selection, tuning, and threshold calibration. Report the longer
training cost and whether its conclusions agree with this pilot.

## Decision rule

For this two-epoch pilot, compare the two attack-test AUPRCs by macro average
within each fold. The behavioral variant passes the *pilot gate* in a regime
when its macro AUPRC improves by at least 0.02 in at least two of three folds,
its macro recall at the fixed threshold does not decrease in at least two
folds, and its normal-test false alerts per hour stay below
`max(raw * 1.25, raw + 1)` in every fold. The thresholds and this rule are
fixed before viewing target scores. `scripts/cross_vehicle_compare.py` creates
the six-run comparison and applies the rule. Passing is a reason to run more
seeds and single-vehicle baselines, not a final transfer claim.

A gain only after target calibration supports normal-data adaptation, not
zero-shot transfer. Compare against independently trained single-vehicle
models before claiming that pooling helps. If raw and behavioral variants
fail, inspect recording provenance, per-attack label coverage and feature
distributions before adding adapters or mixture-of-experts.
