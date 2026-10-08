# Vehicle Graph Mamba data audit: first pass

## Scope and status

Read-only inspection of the local `can-train-and-test-v1.5` raw files, its bundled README, the current cross-vehicle split builder, and the version-3 cross-vehicle manifests. This establishes **file-count feasibility**, not recording/session independence or final snapshot parameters. No target attack scores or labels were used for model selection.

## Findings

| Item | Observed evidence | Implication |
|---|---|---|
| Target normal files | Each of the four vehicles has four `train_01_attack_free` CSVs. Current `CrossVehicleTemporalSource` splits these 2/2 into `target_calibration` and `target_normal_test`; it has no separate reference partition. | A reference/calibration/normal-test split is possible by count (for example 2/1/1), but requires a new manifest/cache and a recording/session overlap check. Reusing the existing calibration files for both context and threshold fitting would confound the paired experiment. |
| Normal durations | Impala: 144, 157, 875, 1350 s; Traverse: 265, 154, 445, 287 s; Silverado: 84, 174, 667, 504 s; Forester: 197, 48, 572, 208 s (rounded span of each CSV). | Reference-duration sensitivity and normal-test duration matter. Do not assign a very short capture to normal testing without reporting the resulting false-alarm-hour uncertainty. |
| Normal ID cardinality | Per file: Impala 51; Traverse 62; Silverado 98; Forester 13, 48, 54, 54. | The 8,192 local-ID tensor capacity is unrelated to observed normal cardinality. Forester `accessory-3` is not representative of the full normal ID set; context coverage needs a mode-aware check. |
| Frequency-only top-K mass | Across the four files, top-32 ID event mass is about 97.3% Impala, 93.8% Traverse, 76.2% Silverado, and 94–95% in three Forester files (100% in the 13-ID file). Top-64 covers all observed Impala/Traverse/Forester IDs and about 97.5% of Silverado events. | Event mass alone can hide rare but important IDs. The anchor policy below retains every reference ID; attack-ID coverage and transient capacity remain to be audited. |
| Payload source | CSV columns are `timestamp,arbitration_id,data_field,attack`; no explicit DLC column. Observed hex payload lengths span 2–16 characters, depending on vehicle/file. | Raw string length may convey byte count, but the CSV lacks an independent DLC/validity field. Check upstream preprocessing before treating length as a trustworthy validity mask. Preserve raw text in the new cache. |
| Suppression labels | The bundled README says suppress attacks have no labeled attack CAN messages and supplies suppressed ID and total removed-message count per file. The existing audit found zero positive event labels in suppression files. | A count is not a start/end interval. Suppression can be studied as an unlabeled stress case, but recall/time-to-alert needs independently justified intervals or a carefully specified capture-level protocol. |
| Vehicle coverage | Current cross-vehicle holdouts are Silverado, Forester, and Traverse. Impala appears as a source vehicle, not a fourth leave-one-vehicle-out target. | The paper's transfer population is three held-out vehicles under the current pipeline. A four-vehicle claim requires a new Impala holdout and a defensible source/target partition. |
| Timestamp provenance | All 16 inspected normal CSVs start at `1672531200.0`. | The shared start suggests timestamps were reset or normalized during curation; it does not prove files are independent sessions. Check upstream provenance and content overlap before treating filenames as independent recordings. |

### Normal exposure and threshold precision

The four normal CSV spans total about 0.70 h for Impala, 0.32 h for Traverse, 0.40 h for Silverado, and 0.28 h for Forester **before** splitting files among reference, calibration, and normal test. These are sums of recorded durations, not proven independent driving hours. A one-file target calibration partition is shorter still. Even observing zero false alerts in a 0.3 h normal test would give a rough one-sided 95% Poisson upper bound near `3 / 0.3 = 10` false alerts/hour under an independence/constant-rate approximation; dependence and operating-mode changes make the uncertainty harder, not easier. Thus this dataset cannot precisely certify a very low per-vehicle false-alarm rate from target-normal calibration alone. Report actual alert counts and exposure, use source-selected thresholds for the primary paired transfer comparison, and avoid claiming a target-specific low-FAR guarantee. Longer independent normal recordings would materially strengthen calibration and deployment claims.

## Normal-reference anchor coverage

Scanned all four `train_01_attack_free` CSVs in each set as text-valued arbitration IDs, counted each file's ID set and event frequencies, then enumerated every two-file reference pair against each of the two other files as a possible normal query. This is a **diagnostic sensitivity analysis**, not permission to choose a target reference pair by looking at held-out query performance. It checks ID membership only; it does not establish timing, payload, graph-edge, or session coverage.

| Vehicle | Union across four normal files | Two-file reference ID count | IDs absent from a separate normal query | Query event mass from absent IDs |
|---|---:|---:|---:|---:|
| Impala | 51 | 51 for every pair | 0 | 0% |
| Traverse | 62 | 62 for every pair | 0 | 0% |
| Silverado | 98 | 98 for every pair | 0 | 0% |
| Forester | 54 | 48 if reference is both `accessory` files; 54 otherwise | 6 for either `attack-free` query after the accessory-only reference; 0 otherwise | 3.16% after accessory-only reference; 0% otherwise |

The Forester's `accessory-3` file alone has 13 IDs; `accessory-4` has 48; each `attack-free` file has 54. The six IDs absent from an accessory-only reference account for 29,095 events in `attack-free-3` and 10,629 in `attack-free-4`. Thus, a small reference can produce benign query IDs that must use transient handling. Conversely, every two-file pair in the other three vehicles covers the observed normal ID set. This says nothing about new IDs under attacks.

**Design consequence:** anchor every distinct ID in the authorized normal reference, provided its count fits a declared `K_max` capacity ceiling. Use `K_actual = |reference ID set|` for each vehicle and keep a separate `M` budget for query-only transient IDs. Do not rank away rare reference IDs merely to force a top-K set. The largest observed normal union is 98 IDs, so all-reference anchoring is feasible by node count in this dataset; `K_max` remains unset until source-only memory/throughput and future-reference capacity checks. At 98 anchors, a complete directed graph could have 9,506 non-self edges per bin, whereas an eight-out-neighbor cap would allow at most 784; graph sparsity still matters even with fewer than 100 IDs. These are arithmetic bounds, not measured GPU costs.

For zero-shot evaluation, `K_actual = 0` target-specific anchors. Query IDs occupy keyed transient slots under the generic graph policy. A target-normal reference cannot be silently derived from a zero-shot query.

The exact-file duplicate audit in the earlier [temporal evaluation audit](2026-10-01-temporal-evaluation-audit.md) found aliases between `train_01_attack_free` and `train_02_with_attacks` for set_01. The current cross-vehicle builder checks SHA256 overlap across its partitions, but exact whole-file hashes do not detect partial capture reuse or common source sessions.

## Required second pass before freezing splits

1. **Independence and identity.** Build a file manifest with raw hash, row count, duration, vehicle, source set, and role. Check exact duplicates across all sets; compare long ordered frame/timing fingerprints for partial overlap; inspect upstream curation scripts and source metadata for shared sessions. Treat uncertain independence as a limitation, not a passed check.
2. **Three-way normal allocation.** For each held-out vehicle, choose reference, threshold-calibration, and normal-test files using normal-data-only criteria. Freeze assignments and hashes before target attack evaluation. Report durations and modes in each partition. Use the *same* target query recordings in zero-shot and conditioned runs. Primary paired results use source-selected thresholds in both modes; target-normal calibration is secondary.
3. **Source episodes.** Count independent normal reference and labeled query captures per source vehicle after grouping duplicates and sessions. Ensure training and validation query captures never build their own context. If a source validation vehicle has too little separate normal data, use a declared training-normal reference for validation queries or redesign the split; never let validation query traffic enter its context.
4. **Snapshot feasibility.** From source training/validation only, sweep candidate physical bin widths. Record events and positive events per bin, empty-bin fraction, short attack dilution, bin-close delay, and estimated `B*T*(K+M)` activations. Do not choose width on target attacks.
5. **Node budgets and graph.** For each candidate reference subset, report reference-to-query ID overlap, rare-ID recurrence, unseen benign IDs, and attack-ID coverage; test transient overflow under observed and synthetic flooding. Measure packed `K_actual + M` streams, graph edge density, and stability across normal modes and reference durations; set `K_max` and `M` from source-only resource/coverage evidence.
6. **Label provenance.** Inventory event labels and any external attack intervals by family and file. Verify suppression and masquerade annotations separately; mark absent intervals unavailable for interval-level recall. Define bin label and alert matching rules before fitting.
7. **Payload and time integrity.** Verify upstream `data_field` serialization and whether its length preserves DLC. Count malformed/odd-length/empty payloads, duplicate and decreasing timestamps, clock gaps, and within-file tie ordering. Preserve these counts in the snapshot cache manifest.

## Decision gate

The first replacement fit may proceed after the split/overlap, timestamp, payload, and label audits yield a frozen versioned manifest, and source-only sweeps set candidate bin widths and node budgets. A paired conditioned result is publishable only where target reference and normal-test captures are demonstrably distinct. Target-normal threshold calibration needs a third demonstrably distinct capture. With four nominal normal files per vehicle, the file count passes; independence, operating-mode coverage, and statistical precision remain open.
