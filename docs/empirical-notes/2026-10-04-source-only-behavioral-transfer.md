# Source-only behavioral transfer audit, 2026-10-04

## Question and protocol

Does a detector using behavior shared across CAN streams transfer to a vehicle it has never seen? This audit uses the **completed** cross-vehicle v2 runs, rather than repeating an equivalent fit. The raw-ID and behavioral variants use matched leave-one-vehicle-out folds. The behavioral input is causal log bus interarrival, same-ID interarrival, deviation from the preceding eight same-ID intervals, same-ID payload byte-change fraction, and first-seen flag. It carries no target-ID embedding or decoded payload signal. It does not include the later bit-field pilot, whose detection gate failed.

The evaluator takes the `zero_shot` threshold at event-level FPR target `0.001` from **benign source-validation scores only** (`scripts/cross_vehicle_evaluate.py`, lines 163–174). It then scores the held-out vehicle's separate normal-test and attack files without target fitting or target threshold calibration. A one-second refractory period is used for false alerts/hour. The v3 split manifests identify source sets, content-grouped source training/validation recordings, target normal files, and target attack files. The model and threshold are trained on source vehicles; ordinary causal state evolves as target messages arrive.

Source/report artifacts:

- `/fs/ess/PAS1266/graphids/runs/cross_vehicle_v2/comparison.json`
- `/fs/ess/PAS1266/graphids/runs/cross_vehicle_v2/audit.json`
- `/fs/ess/PAS1266/graphids/dev/rf15/cross_vehicle/{fold}_{variant}_fit_v2/{fold}_{variant}_fit_v2/cross_vehicle_report.json`
- `/fs/ess/PAS1266/graphids/cache/v10.0.0/cross_vehicle/v3_events_{vehicle}_from_pooled_*_seed42_val0.1_warm64_64/split_manifest.json`

## Zero-shot results

| Held-out vehicle | Variant | Known/unknown attack AUPRC | Known/unknown attack recall | Normal alerts/hour | Normal exposure |
|---|---|---:|---:|---:|---:|
| Silverado | Raw ID | .025 / .335 | .000 / .243 | 2823.6 | .325 h |
| Silverado | Behavioral | .634 / .340 | .104 / .307 | 550.6 | .325 h |
| Forester | Raw ID | .019 / .157 | .019 / .025 | 1139.7 | .217 h |
| Forester | Behavioral | .854 / .886 | .467 / .626 | 4.6 | .217 h |
| Traverse | Raw ID | .071 / .216 | .004 / .020 | 96.4 | .197 h |
| Traverse | Behavioral | .458 / .897 | .359 / .910 | 15.2 | .197 h |

The behavioral model improves macro AUPRC and thresholded recall in all three folds; the repository's comparison declares the pilot rule passed. This is **evidence of learned behavioral transfer**, not evidence of aligned physical signals. The target-normal alert count is roughly 179 on Silverado, one on Forester, and three on Traverse when multiplying reported rates by exposure. The latter two counts have very wide uncertainty because the recordings are short. Silverado remains far too noisy for a low-alert deployment claim.

## Attack-family limits

The aggregate result is uneven. Behavioral zero-shot recall is only .022 for Silverado speed spoofing and .011 for Silverado gear spoofing. Forester triple spoofing has zero recall, while Forester systematic injection has .656 recall. Traverse speed and speed-accessory spoofing have zero recall; Traverse DoS and fuzzing recall are .992 and .889. These examples show that shared timing and whole-payload change behavior transfers better for some traffic disruptions than for subtle or timing-preserving payload manipulations. Per-family AUPRC and counts are in each `cross_vehicle_report.json`.

## Interpretation and next gate

This answers the source-only transfer question positively for the *existing learned behavioral model*: it carries some cross-vehicle detection knowledge without target fitting. It does not establish a useful universal IDS. The normal-alert rate, attack-family gaps, short exposure, and unresolved recording/session independence remain material limitations. The raw-ID baseline is weak on unseen IDs, so the paired gain supports using behavioral inputs but does not isolate which behavioral feature caused the gain. A timing-only versus payload-change-only ablation at the same source-only threshold protocol is needed to attribute that gain. Any new payload feature should also demonstrate incremental recall on speed/gear/RPM and timed masquerade without increasing normal alerts, especially on Silverado.
