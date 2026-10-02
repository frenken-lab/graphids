# Temporal split and evaluation audit

This audit describes the baseline at `0da1dd8`. The CUDA Graph comparison remains valid
as a systems experiment, but the current set_01 split and hybrid anomaly
metrics are not ready for publication experiments. Full validation contains
no attacks, the vocabulary sees validation rows, and successor anomaly scores
are compared with predecessor labels. Longer training should wait for the
evaluation gates below.

The three existing planning drafts were read and preserved. This note adds
current evidence to the September 26 audit; several earlier fixes are already
present, including preprocessing version in the in-process cache key,
explicit mixed versus attack-free training, and shared binary score metrics.

## Audit results

| Area | Result | Evidence and consequence |
| --- | --- | --- |
| Declared source separation | Pass for directory separation | Catalog train and test directories differ. This alone cannot prove independent recording provenance. |
| Exact raw file overlap | Pass for set_01 train versus test; training duplication found | Grouped all 70 set_01 CSVs by byte size and hashed every group with multiple files. Four identical pairs occur within training; none cross training and test directories. Partial recording overlap remains unaudited. |
| Chronological split boundaries | Pass by code inspection | Each stream's last 20% is validation; split starts become self-events with zero deltas/IAT and split ends reset state. Successors in the lane loader stay within the materialized split. |
| Vocabulary | Blocker for strict inductive validation | `CANBusTemporalSource.build` scans every row in training source directories before chronological splitting. `vocab_scope: train` excludes test, but includes validation tails. This exposes validation ID support, not labels. The actual effect on set_01 vocabulary has not been quantified. |
| Causal input features | Pass with naming caveat | Bytes and the feature called entropy are row-local; ID transitions, deltas, and IAT use preceding events within the stream. Entropy is computed from normalized byte magnitudes, not a histogram of byte symbols. No fitted scaler is used in this path. |
| Warmup | Pass for existing masks | First 64 events per validation/test stream advance state without classification loss/metrics; segmented training uses unscored context. Do not move warmup based on attack labels. |
| Validation attack coverage | Fail | All 3,021,438 scored events in the complete set_01 validation cache are benign. Removing the 8,192-event prefix cannot provide AUROC. |
| Test execution | Fixed in this continuation | `test_dataloader` now honors `eval_batch_mode: stream_lanes`, as validation already did. Evaluation lanes use whole recordings continuously, even when training uses segmented execution. |
| Hybrid target semantics | Blocker | Lane IAT targets `msg[i,16]`, already in the input, while event IAT targets the interval to `i+1`. Next-ID and payload targets refer to `i+1`. Event batches also lose next-event targets at every batch end; lanes carry explicit successors across chunks. |
| Hybrid anomaly label alignment | Blocker | Validation/test record `anomaly_score[i]` against `y[i]`, although next-ID/payload components score event `i+1`. True terminals contribute zero anomaly scores to metrics. This is not a correctly aligned next-event detector. |
| Threshold policy and scale | Blocker for deployment metrics | Shared score operating points sweep thresholds on test labels and scan the entire score vector for each unique threshold. These are descriptive test-curve points, not validation-calibrated deployment thresholds; the sweep is potentially quadratic. |
| Checkpoint equality | Pass | Fresh CPU loads confirm equal model tensors and Adam states, equal initial weight digest, and equal section queue digest. |
| Checkpoint provenance | Partial | Reports record config/cache paths, training cache sidecar digest, versions, and initial/queue digests. Checkpoints contain model hyperparameters and optimizer state, but no complete raw/split/vocab manifest or training code revision. |

Code surfaces: `graphids/core/data/datasets/can_bus.py`,
`graphids/core/data/preprocessing/{temporal,vocab}.py`,
`graphids/core/data/datamodule/{temporal,stream_lanes}.py`,
`graphids/core/models/temporal/_hybrid/{targets,runtime,model}.py`,
`graphids/core/models/base.py`, and `graphids/exp/temporal_comparison.py`.

## Verified set_01 cache counts

These are complete cached streams, with 64 warmup events per evaluation
stream. Counts were recomputed from `is_scored` and `y`, not inferred from
filenames. The artifacts are
`experimentruns/cuda_graph_2026_10_01/methodology_audit.json` and
`raw_duplicate_audit.json`.

| Split | Streams | Scored benign | Scored attack | Warmup |
| --- | ---: | ---: | ---: | ---: |
| Train | 18 | 12,033,295 | 57,108 | 0 |
| Validation | 18 | 3,021,438 | 0 | 1,152 |
| Test 01 known vehicle known attack | 8 | 5,638,878 | 63,280 | 512 |
| Test 02 unknown vehicle known attack | 8 | 6,283,238 | 164,167 | 512 |
| Test 03 known vehicle unknown attack | 12 | 8,613,682 | 20,825 | 768 |
| Test 04 unknown vehicle unknown attack | 12 | 13,205,543 | 14,244 | 768 |
| Test 05 suppress | 4 | 3,383,016 | 0 | 256 |
| Test 06 masquerade | 8 | 6,644,152 | 87,882 | 512 |

The duplicate benign training files are `accessory-1.csv`, `accessory-2.csv`,
`attack-free-1.csv`, and `attack-free-2.csv`, each appearing byte-identically
under both `train_01_attack_free` and `train_02_with_attacks`. This duplicates
training weight and validation weight. The existing chronological split applies
the same boundary to each identical copy, so this evidence does not establish
train/validation event overlap. Directory-derived vehicle IDs also cannot
independently verify the catalog's known/unknown vehicle claim.

`attack_type` remains file/path-derived context. Benign rows in attack-named
files can have a nonzero attack type. Suppression files have zero positive
event labels, so the current labels cannot measure suppression detection.
Retain them for false-positive reporting; do not relabel them from filenames.

## Checkpoint identity

The authoritative paired report remains `owned-gradients/pairs.json`; its
selection records median gain `0.28921297976191385`. The complete epoch reports
are `eager_epoch/comparison.json` and `graph_epoch/comparison.json`.

Freshly computed final checkpoint SHA256 values are:

- Eager: `53d542b8e7c3d2b4773ea3310811b0ea9065a3c98c48a0124dd93ce3a16266ba`.
- Graph: `a56c9fdbe7f3ab2e34b53e732f7696a5aa6217b42f40aff5eb1e317b136d2515`.

Whole checkpoint files differ even though model weights and Adam states are
equal. Both saved checkpoints report `epoch: 1` and `global_step: 1720`.
All saved Adam step counters equal 1,712. Actual optimizer updates remain
1,712: Lightning's step count includes skipped
warmup closures and must not replace the measured update count. The cache
digest in the reports is a sidecar value, not a fresh hash verification in this
audit. The baseline commit identifies the audited implementation, not a
proven training-time revision embedded in these checkpoints.

These checkpoints remain historical throughput artifacts. Changing splits,
vocabulary, or target semantics requires a new training lineage and cannot
retroactively turn them into final model quality evidence.

## Proposed continuous evaluation protocol

This protocol defines the intended behavior. Apart from test loader routing,
it is not yet implemented and should not be presented as an operational final
evaluation pipeline.

### Split manifest

Build a versioned manifest from the original training sources only. Group
byte-identical recordings by content hash and assign each group once. Check
recording/session metadata and partial overlap before calling groups independent.
Reserve whole recording groups for validation, targeting 20% of groups within
available vehicle and attack-family strata, ordered by a fixed SHA256 key
derived from split seed 42 and group content hash. Freeze the resulting file
assignments before comparing models or training seeds. Require both scored
classes in validation and attack examples left in supervised training; if the
available independent groups cannot satisfy this, stop and document the missing
development data instead of borrowing any test recording. The exact manifest
and whether grouping permits this allocation remain required deliverables.

This changes the claim from within-recording future-tail validation to held-out
recording validation. Keep the old tails only as a named benign diagnostic.
Never select windows around attacks or retain the test sets as a development
pool. Fit vocabulary on final training assignments only; map unseen validation
and test IDs to zero with the existing unknown-ID features. Any statistics or
score normalization must also use training or validation as declared. Create
a new cache version/digest and verify complete expected test directories;
the source currently silently skips absent test directories.

Mixed supervised/joint training may use attacked training recordings. An
attack-free anomaly experiment trains only on benign development training
sources and uses the same held-out evaluation assignments; attack-inclusive
validation may calibrate its threshold, but that calibration must be disclosed.
Do not describe the current mixed joint run as unsupervised normality training.

### Stream state and coverage

Run complete validation and every declared test recording in time order.
Initialize state independently for each recording, clear all training state,
and reserve exactly the first 64 events as unscored warmup. State continues
across chunks without parameter updates; reset at true recording ends.
Use continuous evaluation lanes, keep partial batches, remove validation
prefix limits, and set evaluation batch limits to 1.0. Evaluation geometry may
change throughput but must preserve scores and coverage within declared
tolerances. Require per-split coverage reports for unique events, both classes,
warmup, padding, and eligible successor targets. The existing coverage callback
audits training only; evaluation needs its own audit.

### Targets and event alignment

Classification at event `i` uses its causal features and compares with `y[i]`.
For next-event scoring, the heads after event `i` predict destination,
`log1p(iat[i+1])`, and payload delta of `i+1`. Obtain all three targets from
explicit successors within the same split and recording, including across
chunk boundaries. Input features must never include successor features.

Attribute that likelihood score to the arriving event `i+1`, with its own
label, attack type, event ID, and scoring mask. A prediction from the last
warmup event can score the first scored event; a terminal event generates no
new successor score. Keep separate classification and anomaly coverage counts
where a scored event lacks an eligible predecessor. Use synthetic onset/offset
and nonuniform-IAT cases to verify alignment, and require chunk-size invariance.
Keep components separate until timing targets are corrected; the current
composite cannot be called a consistent next-event likelihood.

### Metrics and checkpoint selection

Report classification and anomaly results under distinct metric namespaces
so joint runs cannot overwrite one with the other. Primary ranking metrics
are AUROC and average precision, per named test split and pooled across all
declared splits; report prevalence and eligible counts alongside them.
Report macro-F1, MCC, precision, recall, and false-positive rate at a threshold
chosen on validation once per seed, then frozen for all test splits.

For the proposed main supervised/joint table, select the checkpoint with
highest complete-validation classification AP, breaking ties by earliest
epoch. For a separate anomaly table use validation anomaly AP. Calibrate the
main threshold by highest validation MCC, breaking ties by higher threshold;
optional validation operating points target 95% recall and 99% precision.
Report unattainable targets explicitly. Test-curve operating points may be
included only as descriptive curves. Replace the quadratic threshold sweep
with a sorted cumulative calculation before full test runs.

Keep single-class splits in false-positive reporting and pooled results, but
mark their AUROC unavailable. Per-attack results require reliable label and
family provenance and a declared benign reference population. Do not silently
drop suppression files or claim event-level suppression recall.

### Experiment lineage and launch gates

Each fit/test artifact must record the exact checkpoint SHA256, training code
revision and dirty diff digest, resolved configuration, training seed, split
manifest digest, raw recording hashes, vocabulary digest, preprocessing
version, cache hashes, dependency/device versions, scoring alignment version,
and threshold calibration artifact. Test must reload that exact checkpoint
without fitting, checkpoint reselection, or threshold tuning on test labels.

Before longer runs, implement and test the manifest/vocabulary changes,
successor timing and label alignment, evaluation coverage, scalable metrics,
distinct score namespaces, and provenance. Preserve graph replay ownership,
gradient cloning hooks, RNG handling, and eager boundary fallbacks. Reverify
CUDA correctness after changing targets; retain the old throughput artifacts.

After those gates pass, run one checkpointed end-to-end pilot with complete
validation and test coverage. Then freeze a five-seed plan (42 through 46)
using the same split manifest, epoch/update budget, and selection policy across
models. Choose the longer-training budget from pilot convergence and walltime,
and report per-seed results plus mean and dispersion. No longer jobs were
launched during this audit.

## Verification of this continuation

The loader regression exercises two named test datasets with segmented
training settings, continuous evaluation, partial lanes, a validation prefix,
warmup/scoring masks, and successors across chunk boundaries. Both test
datasets retain every event once and remain independent of the validation
prefix.

Validation commands:

```bash
.venv/bin/python -m pytest tests/core/data/test_temporal_datamodule.py tests/core/data/test_temporal_coverage.py tests/core/preprocessing/test_temporal.py -q
.venv/bin/ruff check graphids/core/data/datamodule/temporal.py tests/core/data/test_temporal_datamodule.py
.venv/bin/python scripts/lint_docs.py
```

Results: 53 tests passed; Ruff and documentation checks passed. This
continuation did not modify CUDA Graph training, target construction, cache
contents, or the selected throughput configuration. No fresh CUDA benchmark
was needed for the test-loader routing change.

## Recording validation implementation

The requested validation fix adds `source.val_split: recordings` and
`source.split_seed: 42`. It is enabled in all eight set_01 final training
configs. `configs/experiments/temporal_set_01_recording_holdout.yml` provides
the selected 256-lane training geometry with complete validation every epoch.
Historical comparison configs retain their chronological caches and prefix
diagnostics; use the new config for attack-inclusive validation.

The new policy hashes training-source files, materializes one copy per exact
content group, and assigns complete recordings deterministically within
attack-family strata. It selects no test rows or attack-centered windows.
Validation must contain both scored classes after warmup, and mixed training
must retain attacked recordings. Attack-free experiments use the same held-out
validation recordings while excluding attacked recordings from training.

Vocabulary construction now uses the final training recordings for this
policy. The split seed is independent of model initialization seed. A
`split_manifest.json` records hashes, duplicate aliases, assignments, and
counts; its digest and a separate cache suffix distinguish the new lineage.
Validation prefixes are rejected for recording holdouts. The earlier target
alignment, threshold, and partial-recording-provenance findings still apply.

The set_01 manifest assigns eight recordings to training and six to
validation, after collapsing four duplicate benign files. Each of the five
attack families has two recordings, so holding out one per family plus one
benign recording exceeds the nominal 20% recording target. Raw counts are
6,640,741 training events with 18,494 attacks and 4,819,950 validation events
with 38,614 attacks. All 38,614 validation attacks occur after the 64-event
warmup. The old checkpoints belong to the previous split and require fresh
training to use this holdout correctly.

Cache build and verification artifacts are under
`experimentruns/recording_validation_2026_10_01/`. The fixed manifest digest is
`a26025c6bfb8b79566cfa9fd825dbbe7bc7892d7046c6b4c8b20d40662dfa785`.

The CPU cache build completed on Pitzer in 188.76 seconds. Counts recomputed
from the saved tensors match the manifest: validation has 4,780,952 scored
benign events, 38,614 scored attacks, and 384 unscored warmup events. All six
test sets retain their previous event, label, stream, and warmup counts. The
new vocabulary may change their ID mapping, as intended. No training job was
launched.

Regression coverage includes duplicate grouping, whole-stream boundaries,
train/validation event ID separation, validation-only/test-only unknown IDs,
test-independent deterministic assignments, attack-free training, infeasible
holdouts, attacks confined to warmup, cache identity, complete validation, and
the final experiment configs.

Final checks: 89 focused data, preprocessing, coverage, and configuration
tests passed; Ruff, documentation lint, and `git diff --check` passed.
