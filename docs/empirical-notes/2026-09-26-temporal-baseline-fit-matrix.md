# 2026-09-26 Temporal Baseline Fit Matrix

## Scope

Completed the one-epoch temporal baseline fit matrix for plan
`temporal_baseline_2026_09_26`.

These are fit-phase, small-scale, checkpoint-light baseline/probe runs over the
real temporal CAN data. They validate the current Ray-on-SLURM execution path
and provide first-pass fit/validation metrics. They are not final test-stage
claims.

Relevant commits:

- `701dd1b` - removed the unsupported Ray client server port flag.
- `e2c3492` - added `fit_baselines` result view.

After this note, `fit_baselines` is scoped to
`plan_id: temporal_baseline_2026_09_26`.

Query:

```bash
gx exp results --view fit_baselines --format json
```

## Coverage

All 20 baseline configs under `configs/experiments/baselines/` have FINISHED
MLflow fit runs:

- event_classifier: `hcrl_sa`, `set_01`, `set_02`, `set_03`, `set_04`
- gat: `hcrl_sa`, `set_01`, `set_02`, `set_03`, `set_04`
- rnn: `hcrl_sa`, `set_01`, `set_02`, `set_03`, `set_04`
- vgae: `hcrl_sa`, `set_01`, `set_02`, `set_03`, `set_04`

## Second Wave Job IDs

These 11 jobs filled the missing coverage after the first 9-run wave. All
completed with SLURM state `COMPLETED` and exit code `0:0`.

| Job ID | Variant | Elapsed |
|---:|---|---:|
| 54818246 | temporal_baseline_event_classifier_hcrl_sa | 00:00:59 |
| 54818248 | temporal_baseline_event_classifier_set_04 | 00:02:47 |
| 54818253 | temporal_baseline_gat_hcrl_sa | 00:01:10 |
| 54818255 | temporal_baseline_gat_set_01 | 00:05:27 |
| 54818259 | temporal_baseline_gat_set_02 | 00:06:46 |
| 54818261 | temporal_baseline_gat_set_03 | 00:05:18 |
| 54818265 | temporal_baseline_gat_set_04 | 00:04:12 |
| 54818266 | temporal_baseline_rnn_hcrl_sa | 00:01:02 |
| 54818269 | temporal_baseline_rnn_set_03 | 00:04:12 |
| 54818310 | temporal_baseline_vgae_set_01 | 00:04:51 |
| 54818311 | temporal_baseline_vgae_set_04 | 00:03:38 |

## MLflow Runs

| Variant | Dataset | MLflow run ID | val_acc | val_auroc | val_loss |
|---|---|---|---:|---:|---:|
| temporal_baseline_event_classifier_hcrl_sa | hcrl_sa | 35ca89f9ee9a4f69af068f4e530e1ac3 | 0.997321 | 0.998452 | 0.010961 |
| temporal_baseline_event_classifier_set_01 | set_01 | 3c929e813d18416aa6c6f5371ac7a289 | 0.999855 | n/a | 0.001068 |
| temporal_baseline_event_classifier_set_02 | set_02 | a35c3b6775d14ddc809480e10e8c8a91 | 0.999005 | 0.998984 | 0.002886 |
| temporal_baseline_event_classifier_set_03 | set_03 | 625e20ecd4c2475c9087361cd368afd3 | 0.999986 | n/a | 0.000378 |
| temporal_baseline_event_classifier_set_04 | set_04 | 59e1ae1f2d454922a3f03a3a303b0045 | 0.999961 | 0.912795 | 0.000905 |
| temporal_baseline_gat_hcrl_sa | hcrl_sa | 656d2955e9734f29b8582a151db04297 | 0.994685 | 0.999108 | 0.020353 |
| temporal_baseline_gat_set_01 | set_01 | 8e90ede92518469f839026dba77a1d41 | 0.999986 | n/a | 0.000159 |
| temporal_baseline_gat_set_02 | set_02 | eb07ff90202c4a0e92d1a6aa4112abbe | 0.999052 | 0.999572 | 0.002467 |
| temporal_baseline_gat_set_03 | set_03 | 113d8064194e44bda3cc9ebfe1384df4 | 0.999966 | n/a | 0.000162 |
| temporal_baseline_gat_set_04 | set_04 | 8102593d58074366983bd56f9dec3461 | 0.999950 | 0.839293 | 0.000442 |
| temporal_baseline_rnn_hcrl_sa | hcrl_sa | a1373cf974334cc6aeb8886d64c7ff67 | 0.996503 | 0.999720 | 0.012392 |
| temporal_baseline_rnn_set_01 | set_01 | f62c2d20992148b1b5d8d951850df4dd | 1.000000 | n/a | 0.000283 |
| temporal_baseline_rnn_set_02 | set_02 | 7674b9e802b84817b8bfee10950a71ac | 0.999220 | 0.985808 | 0.004157 |
| temporal_baseline_rnn_set_03 | set_03 | b611b670f28b43df8676656abb19dae3 | 1.000000 | n/a | 0.000394 |
| temporal_baseline_rnn_set_04 | set_04 | 7cbf5f09aad44fd3a48c1b68e4b10b3d | 0.999955 | 0.918547 | 0.000550 |
| temporal_baseline_vgae_hcrl_sa | hcrl_sa | 2936fdf1fc054fe5988c0e8f20179e69 | n/a | 0.488791 | 1339.430298 |
| temporal_baseline_vgae_set_01 | set_01 | 6398efe3bdb9490e8270c3e671fe0cd2 | n/a | n/a | 26.805429 |
| temporal_baseline_vgae_set_02 | set_02 | 40322d4e72e648a6ac10129486a860ee | n/a | 0.325415 | 19.061075 |
| temporal_baseline_vgae_set_03 | set_03 | fc4a328a9af449ad8be23ea07d08a146 | n/a | n/a | 18.578741 |
| temporal_baseline_vgae_set_04 | set_04 | 1a02073e2ed94c6f9a0b7f74b15472d6 | n/a | 0.673548 | 30.565287 |

## Notes

- The Ray background `ray --block` steps appear as `CANCELLED` in `sacct`
  because the SLURM cleanup trap terminates the background Ray server after the
  driver exits. The top-level jobs and GraphIDS manifests are the success
  signal here.
- VGAE metrics are reconstruction-style fit diagnostics plus sparse AUROC where
  the model logged it. Treat these as probe metrics, not final anomaly
  benchmark numbers.
- Classifier/GAT/RNN validation accuracies are very high after one epoch. These
  should be followed by checkpointed train/test runs before using them as final
  paper/table results.
