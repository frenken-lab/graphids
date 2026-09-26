# 2026-09-26 Temporal Methodology Audit

## Verdict

Publication-gate result: **pass with blockers for final experiment launch**.

The current temporal data pipeline is defensible on the core leakage controls:
catalog train/test directories are disjoint, train/validation splitting is
chronological within streams, validation and test split starts are converted to
self-events with zeroed transition features, test streams are materialized from
separate source directories, and supervised temporal classifiers apply
`is_scored` masks before losses and metrics.

Do not start long checkpointed train/test experiments until the blockers below
are resolved or explicitly scoped out of the publication claim.

## Evidence

### Commands

Focused regression tests:

```bash
.venv/bin/python -m pytest tests/core/preprocessing/test_temporal.py tests/core/data/test_temporal_datamodule.py tests/core/models/test_temporal_gat_vgae.py tests/core/models/test_temporal_rnn_classifier.py tests/exp/test_ray_temporal.py
```

Result: `16 passed, 14 warnings in 11.86s`.

Baseline result-view check:

```bash
.venv/bin/gx exp results --view fit_baselines --format json
```

Result: 20 `FINISHED` fit runs: 4 model variants
(`event_classifier`, `gat`, `rnn`, `vgae`) across 5 datasets
(`hcrl_sa`, `set_01`, `set_02`, `set_03`, `set_04`).

### Code Audit

- `configs/data/datasets.json` declares separate train and test source
  subdirectories for all five publication datasets.
- `graphids/core/data/datasets/can_bus.py` uses filename/path substring
  inference for `attack_type` (`infer_attack_type`, lines 71-77), loads rows
  from declared source directories only (`load_can_rows`, lines 80-111), and
  defaults `vocab_scope` to `train` (`CANBusTemporalSource`, line 124).
- The filesystem cache path includes the preprocessing version via
  `graphids.paths.cache_dir`; the temporal source also includes representation
  digest, vocab scope, validation fraction, and warmup settings in
  `cache_key`/`cache_root_path` (lines 143-168).
- `build_temporal_event_table` preserves source provenance, sorts within
  stream, computes source ID, payload deltas, and inter-arrival time using
  `shift(1).over("stream_id")`, and derives labels from the current row
  `attack` flag (lines 93-168).
- `split_temporal_train_val_tables` partitions by stream and assigns the final
  interval to validation, then calls `mark_split_start_self_events`,
  `mark_terminal_reset`, `add_temporal_split_masks`, and
  `assert_temporal_splits_disjoint` (lines 267-306).
- `prepare_temporal_eval_table` applies the same split-start and warmup mask
  logic to full validation/test-style streams (lines 309-320).
- Supervised classifiers and VGAE mask warmup/non-scored events before
  train/validation/test losses or buffers. Representative evidence:
  `TemporalEventClassifier.training_step`, `validation_step`, and `test_step`
  lines 92-151; `TemporalVGAE.training_step`, `validation_step`, and
  `test_step` lines 103-145.
- Classification test metrics include AUROC, MCC, macro/weighted F1,
  per-class precision/recall/specificity/F1/AUROC/AP, average precision, and
  calibration error in `classification_test_metrics` (lines 20-70). Operating
  points and per-attack AUROC are logged from `_ModelBase` (lines 73-159).

### Cached Data Audit

Read-only cache inspection used existing caches under:

```text
/fs/ess/PAS1266/graphids/cache/v10.0.0/<dataset>/temporal_1e036beae166_voc_train_val_0.2_vw_64_tw_64
```

No cache rebuild was launched. All five caches were complete. All five had
zero train/test source-directory overlap and zero train/validation `event_id`
overlap.

| Dataset | Cache | Train scored labels | Val scored labels | Val warmup | Test scored labels by subdir | Test warmup | Unknown ID rate in scored val/test |
|---|---|---:|---:|---:|---|---:|---|
| `hcrl_sa` | ready | 749,561: 712,108 benign / 37,453 attack | 187,004: 173,225 / 13,779 | 384 | `test_01`: 399,409: 327,175 / 72,234; `test_02`: 186,107: 157,708 / 28,399; `test_03`: 305,959: 282,584 / 23,375; `test_04`: 79,723: 71,676 / 8,047 | 448 | 0.0000% val; 0.0000% all tests |
| `set_01` | ready | 12,090,403: 12,033,295 / 57,108 | 3,021,438: 3,021,438 / 0 | 1,152 | `test_01`: 5,702,158: 5,638,878 / 63,280; `test_02`: 6,447,405: 6,283,238 / 164,167; `test_03`: 8,634,507: 8,613,682 / 20,825; `test_04`: 13,219,787: 13,205,543 / 14,244; `test_05`: 3,383,016: 3,383,016 / 0; `test_06`: 6,732,034: 6,644,152 / 87,882 | 3,328 | 0.0000% val; tests range 0.0000%-59.4901% |
| `set_02` | ready | 16,283,951: 16,048,274 / 235,677 | 4,069,570: 4,066,506 / 3,064 | 1,408 | `test_01`: 13,219,787: 13,205,543 / 14,244; `test_02`: 8,444,659: 8,424,610 / 20,049; `test_03`: 12,149,544: 12,120,753 / 28,791; `test_04`: 4,789,392: 4,771,384 / 18,008; `test_05`: 4,946,201: 4,946,201 / 0; `test_06`: 6,301,420: 6,261,089 / 40,331 | 3,328 | 0.0000% val; 0.0000% all tests |
| `set_03` | ready | 13,291,255: 13,121,331 / 169,924 | 3,321,525: 3,321,525 / 0 | 1,280 | `test_01`: 8,575,151: 8,412,646 / 162,505; `test_02`: 6,853,594: 6,697,405 / 156,189; `test_03`: 9,447,055: 9,428,903 / 18,152; `test_04`: 6,895,123: 6,745,175 / 149,948; `test_05`: 4,250,268: 4,250,268 / 0; `test_06`: 7,490,410: 7,418,117 / 72,293 | 3,840 | 0.0000% val; tests range 0.0000%-9.5458% |
| `set_04` | ready | 9,795,505: 9,766,822 / 28,683 | 2,447,202: 2,447,093 / 109 | 1,664 | `test_01`: 6,895,123: 6,745,175 / 149,948; `test_02`: 17,407,198: 17,349,959 / 57,239; `test_03`: 6,853,594: 6,697,405 / 156,189; `test_04`: 8,182,986: 7,962,015 / 220,971; `test_05`: 2,001,981: 2,001,981 / 0; `test_06`: 8,754,307: 8,601,168 / 153,139 | 3,840 | 0.0000% val; 0.0000% all tests |

The warmup counts match 64 events per cached stream. For example, `set_04`
validation has 26 streams and 1,664 warmup events.

## Findings

### Split and Leakage Audit: pass

- Train/test source directories are disjoint for `hcrl_sa`, `set_01`,
  `set_02`, `set_03`, and `set_04`.
- Train/validation split is chronological within each `stream_id`; validation
  takes the final interval per stream.
- Split starts are rewritten as self-events and transition deltas/IAT are
  zeroed, so no transition feature crosses from train into validation or from
  pre-split context into test.
- Test streams are built from each test source directory independently, not
  sliced out of train/validation streams.
- Baseline configs use the default `vocab_scope: train`; unseen IDs map to
  node id `0`, with `src_is_unknown`, `dst_is_unknown`,
  `src_unknown_bucket`, and `dst_unknown_bucket` present in the temporal
  message features.

### Preprocessing Audit: pass with caveat

- Raw payload parsing, entropy, byte deltas, and IAT are row-local or
  previous-event-within-stream computations. No audited feature uses future
  rows.
- Persistent cache paths include preprocessing version, representation digest,
  vocabulary scope, validation fraction, and warmup settings.
- Caveat: the in-process `CANBusTemporalSource.cache_key` omits
  `PREPROCESSING_VERSION` even though the filesystem path includes it. Fresh
  experiment processes are protected by the versioned path, but long-lived
  Python sessions could reuse a stale in-memory `DatasetState` after a version
  bump.
- Caveat: `attack_type` is inferred from file/path substrings. This is usable
  for coarse per-attack reporting, but it is not an event-level label source.
  The cached audit shows benign rows with nonzero `attack_type` in attack-named
  files, and suppress splits have `attack_type=15` while `y=0` throughout.

### Metric and Evaluation Audit: pass with blockers

- Supervised temporal classifier, GAT, RNN, and VGAE steps use `is_scored`
  masks before losses or test buffers.
- Supervised classifier test metrics include the required publication metrics:
  AUROC, MCC, macro/weighted F1, per-class metrics, operating points, and
  per-attack AUROC where labels contain both classes.
- Missing validation AUROC for `set_01` and `set_03` is expected: their scored
  validation splits are single-class benign after masking. This is a caveat for
  validation diagnostics, not evidence of a pipeline failure.
- Accuracy must not be used as the primary publication metric. The cached
  splits are extremely imbalanced; e.g. `set_04` scored validation has only
  109 attack events out of 2,447,202 scored events.
- Blocker: VGAE does not currently log comparable test-stage AUROC/MCC/F1 or
  operating-point metrics from its anomaly scores. Its `test_step` records
  scores, labels, and attack types, but `_ModelBase` only logs classifier
  metrics when `test_metrics` exists.

### Model Comparability Audit: pass with caveat

- Event classifier: stateless MLP over temporal message features plus source
  and destination ID embeddings.
- GAT baseline: causal transformer-style attention over event batches plus ID
  embeddings.
- RNN classifier: GRU over temporal event features plus ID embeddings, with
  state carried across batches and reset by `reset_after`.
- VGAE: recurrent variational event autoencoder over temporal message features
  plus ID embeddings, scoring reconstruction error plus KL.
- Current 20-run matrix is one-epoch `fit` only. It validates data/model/Ray
  execution, but it is not final test evidence.
- Caveat: the current source builder trains on both `train_01_attack_free` and
  `train_02_with_attacks`. That is valid for supervised IDS claims if stated,
  but it is not valid for an unsupervised anomaly-detection claim unless the
  training source is changed or the claim is narrowed.
- Caveat: model capacity and budget are not yet normalized beyond `scale:
  small` and `max_epochs: 1`. A later model-card pass should document
  parameter counts, epochs, early stopping/checkpoint selection, and train/test
  wall-clock budget.

## Blockers

1. **VGAE has no comparable test-stage metrics.**
   - Surface: `graphids/core/models/temporal/vgae.py` `test_step` and
     `graphids/core/models/base.py` `_log_classifier_metrics`.
   - Evidence: `TemporalVGAE.test_step` records anomaly `scores` and labels,
     but the shared test metric logger only computes metrics when a model owns
     `test_metrics`; VGAE does not.
   - Required fix: add binary anomaly-score test metrics for VGAE or a shared
     score-based evaluator that logs AUROC/AP, MCC/F1 at declared thresholds,
     operating points, per-test-split metrics, and per-attack AUROC where both
     classes exist.

2. **Training-regime claim is not yet fixed for publication.**
   - Surface: `CANBusTemporalSource.build`, lines 233-247, combines
     `train_subdir` and `train_attack_subdir` for train/validation.
   - Evidence: cached train splits contain positive attack labels for every
     audited dataset.
   - Required fix: before final experiment configs, choose and document one of:
     supervised training with attack-bearing train data; or unsupervised
     anomaly detection with attack-free training sources for applicable models.

## Required Fixes Before Final Experiments

1. Implement and test VGAE test-stage score metrics, or remove VGAE from the
   final comparable baseline table.
2. Freeze the final training protocol in config: supervised mixed-train or
   attack-free anomaly training, with names that make the regime explicit.
3. Use checkpointed train/test configs. Do not use the current one-epoch
   `fit` matrix as final evidence.
4. Add a small hardening patch so `CANBusTemporalSource.cache_key` includes
   `PREPROCESSING_VERSION`, matching the filesystem cache path.
5. Expose semantic `attack_type` names through the temporal datamodule/source
   before relying on per-attack metric names in final tables.
6. In final reporting, exclude or separately report test subdirectories with
   single-class scored labels, such as suppress splits where `y=0` throughout.

## Recommended Final Evaluation Metrics

- Primary: AUROC, MCC, macro-F1, weighted-F1, and average precision on held-out
  test splits.
- Required operating points: precision at fixed recall, recall at fixed
  precision, and the thresholds used for those points.
- Required breakdowns: per-test-subdir metrics, pooled test metrics,
  per-class precision/recall/F1/specificity, and per-attack AUROC where both
  benign and attack labels exist.
- Secondary only: accuracy. Accuracy is too sensitive to the observed class
  imbalance to support a publication claim by itself.

