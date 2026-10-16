# Gear/RPM diagnosis and prediction target — 2026-10-16

Choose a **hybrid, message-preserving anomaly direction**, with attacker-frame attribution evaluated separately. The source evidence supports retaining individual messages and investigating normal-only payload-value distributions. It does not support replacing the existing graph classifier/forecast branches with the message classifier, or treating its RPM bin detections as successful attacker-frame attribution. Tight-budget RPM detection remains unresolved. No further full encoder fit was launched.

This investigation uses the 25 reviewed Impala/Silverado source recordings (21 capture groups), the existing splits, and the frozen epoch-6 checkpoint `e513006088ffd06a6135a21dcc30fdf208ae1aed17ddbcdaeecb30c6ca86dcc6`. It audits 22,382,542 frames. Frozen extraction covers all 1,145,440 source-query bins on two A100 ranks, with no missing or duplicate bins. Native prediction coverage checks account for 20,776,971 observed ID-bins and all 21,285,666 query frames. Reference recordings supply the remaining frames. Unknown bin labels remain unknown and are excluded from metrics; all observations are retained. Message/transition empty bins score zero and are explicitly unsupported for silence detection. No target recording or unresolved variant was evaluated.

After applying the saved evaluator arithmetic, both replay heads match all 694,916 retained conditioned validation scores exactly. The primary validation comparison uses those retained scores. Calibration, production snapshots, manifests and checkpoints remain unchanged. Existing zero-shot results remain a reference; the new probes are conditioned only.

Validation bus-bin AP, including verified-normal queries from the same vehicles:

| Readout | Gear AP | RPM AP |
|---|---:|---:|
| Existing classifier | 0.001535 | 0.002258 |
| Existing forecasting | 0.003855 | 0.002694 |
| Frozen pooled, ordinary BCE | 0.001846 | 0.001927 |
| Frozen pooled, balanced BCE | 0.002155 | 0.002349 |
| Per-ID max, bin supervision | 0.001805 | 0.001709 |
| Per-ID, observed-ID supervision | 0.005461 | 0.002916 |
| Causal message MLP | 0.468925 | 0.289044 |
| Normal-only byte transitions | 0.038600 | 0.131725 |

Balanced pooled BCE changes little, and localization on frozen states does not resolve the failure. These results do not establish that end-to-end rebalancing could never help; they show that balance or a localized linear readout alone is an insufficient remedy on this encoder. The node sidecars use float16, while pooled inputs retain float32, which is a numerical limitation of this comparison.

The prediction unit changes the interpretation substantially:

| Message probe | Attacker-frame AP | Attacker-frame TP / positives at 0.5 | Observed ID-bin AP | Bus-bin AP |
|---|---:|---:|---:|---:|
| Gear | 0.423263 | 242 / 548 | 0.442268 | 0.468925 |
| RPM | 0.000236 | 0 / 374 | 0.253367 | 0.289044 |

RPM has **zero attacker-frame true positives** at 0.5, but 142 detected positive ID-bins. Those detections therefore arise from legitimate messages in the same ID-bin. This is a contextual anomaly signal, not recovered attacker-frame authenticity. Presence of an attacker frame in a bin is also not an independently verified operational-state label. Frame, observed-ID and bus metrics have different populations and are not interchangeable.

The normal-only byte baseline has RPM attacker-frame AP 0.114938, substantially above the message classifier’s 0.000236. Payload-value predictability therefore contains useful RPM evidence that a classifier fitted to the source injection examples does not transfer successfully. However, the byte baseline has zero gear/RPM bus recall at the 10/hour false-bin budget. Its raw likelihood is not an operational replacement or a new production calibration. The baseline pools source vehicles by CAN ID, so differing normal ID semantics can broaden its distribution.

Message-probe recall under shared pooled empirical normal thresholds:

| Budget (false bins/hour) | Achieved pooled rate/hour | Gear recall | RPM recall | Threshold-boundary ties |
|---|---:|---:|---:|---:|
| 0 | 0.0000 | 0.2865 | 0.0000 | 1 |
| 10 | 9.7082 | 0.2865 | 0.0000 | 1 |
| 100 | 99.0234 | 0.3540 | 0.1925 | 2 |

These thresholds use only 0.51503 hours of verified-normal source validation exposure. Zero observed false bins is an empirical result on that exposure, not a future false-alarm guarantee. At the pooled 10/hour threshold, gear recall is 31.91% on Impala and 0% on Silverado; Silverado reaches 100% gear recall at the separate fixed 0.5 threshold. The vehicle disagreement matters. Fixed thresholds, diagnostic sweeps and the existing model’s calibration remain separate tracks.

Overwrite is real but cannot explain all misses. Impala gear payload survival falls from 1,943/1,973 positive training ID-bins to 386/492 in validation; RPM falls from 851/967 to 207/374. Training has 6,500 injected gear frames in 1,973 bins and 3,338 RPM frames in 967 bins, whereas validation has exactly one injected frame per positive bin (492 gear, 374 RPM). The payload and injection-density regimes change together. Message scoring recovers 90/167 overwritten RPM bins and 13/13 overwritten Silverado gear bins at 0.5, but 0/106 overwritten Impala gear bins. Thus its gains do not isolate overwrite recovery as a universal cause. Representation, supervision and readout capacity all change in the message comparison. Ten epochs with batches of 4,096 native supervised units also give the message/ID heads more optimizer steps than the pooled heads.

Payload identity also shifts: Impala training/validation RPM payloads are `80000000000018` and `80394e19734008`. All 374 validation RPM attacker frames contain a byte value unseen for that ID/position in verified-normal source training. Gear attack bytes all have marginal normal support, which is not joint payload support. Exact `1F5:0d0d000300000800` appears in 524 attacker frames and 792 legitimate source frames, including 771 verified-normal Silverado frames (4 training, 767 validation). This is overlap of current CAN ID/payload across recordings and vehicles; timing, history, reference context and other IDs are outside that comparison. It does not establish whole-model indistinguishability. No same-recording current-ID feature-pattern collisions were found for gear/RPM, which likewise does not prove that origin attribution will generalize.

Attacked gear/RPM IDs have reference coverage and graph neighbors. Gear has eight valid bytes and RPM seven, consistent with their payload lengths. The graph relationships are constructed from count/change behavior, and the four reference profiles describe traffic statistics. The existing forecast heads predict counts, change occurrence and new-ID rate, not payload values. Their supervision is restricted to verified-normal recordings; attacker-origin negatives inside attack recordings do not become normal forecast targets.

All ten reconstructed schedules match saved recording/bin/event work signatures. Each uses 2,550 of 3,523 available windows, with rotating omissions; their union covers all 3,523 windows. RPM positive exposure varies from 149 to 789 bins per schedule, gear 725–1,548, and DoS 12,790–13,302. Exposure imbalance exists, but residual loss tells a different story: at the selected frozen checkpoint, the first original schedule assigns about 21.12% of classifier loss to gear, 1.12% to RPM, and 68.86% to double/triple captures. RPM positive BCE averages about 0.0005 on that training schedule. Training recognition does not transfer to validation. This training calculation uses original finite episodes/context modes, while validation uses continuous streams, so the loss gap alone does not isolate overfitting.

The exact paired-rank reduction audit gives first-schedule component means 0.00410 classification, 0.17248 count, 0.05447 change and 0.00789 new-ID. Loss magnitudes are not gradient magnitudes. The separate deterministic family-stratified gradient audit reports projection/temporal gradients and their scope; normal forecasting can exert substantial pressure, but one sample per family does not identify a sole cause. Saved epoch metrics show weak gear/RPM validation AP throughout the ten epochs, with some fluctuations rather than an earlier strong detector. No alternate checkpoint was selected.

The outcome is **mixed across targets, families and vehicles**. Existing classification has double/triple AP 0.6807/0.8049 versus message-probe 0.1859/0.0046. Existing forecasting retains random-ID-fuzzing AP 0.9463 versus classifier 0.0013. Retain these complementary branches as controls. The next substantive model direction should preserve messages, score local ID context, and test normal-only payload distributions conditioned on previous messages and available bus context. Evaluate an operational anomaly target separately from attacker-frame origin, and investigate per-ID/vehicle normal likelihood calibration and normal-state coverage before another full encoder fit. RPM at stringent false-bin budgets remains a narrower unresolved follow-up.

All gains are exploratory source evidence. The encoder/checkpoint was already selected using source validation, including validation-only fuzzing; these are not independent generalization results. There are no trusted attack intervals supporting incident recall, attack duration or detection-delay claims.

The [CLI protocol](../reference/missed-attack-diagnostics.md) describes inputs, weighting, causal features and artifacts. Full per-family/per-vehicle/native-unit metrics, counts, confusion matrices, threshold ties and overwrite strata are in `results.json`; the complete tables are in `evidence-report.md`. Additional evidence includes `audit/payload-pattern-support.json`, `audit/feature-evidence.json`, `audit/training-exposure.json`, `embeddings/distributed-loss-contributions.json`, `embeddings/gradients.json`, and per-probe `verified-coverage.json`.

Artifact root: `/fs/ess/PAS1266/graphids/dev/rf15/experimentruns/graph-mamba-missed-attacks-v1/`. GPU allocation Ascend 7890431 completed in 50m10s; CPU report 56283156 completed in 18m25s. Sixteen focused checks and the earlier regression batches passed in Slurm, including future perturbations, labels/overwrites/boundaries, training-only fitting, aggregation, coverage rejection, and global DDP loss denominators. The native two-GPU fixture matched serial scores and exact coverage. Production schemas, checkpoints, manifests and calibration artifacts were not changed.
