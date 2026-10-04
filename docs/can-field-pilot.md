# Anonymous CAN payload-field pilot

> Historical pilot: the active CLI now runs the unified version-2 discovery pipeline. See [CAN behavior discovery](behavior-discovery.md). The commands and report fields below describe the original pilot run.

Run `python scripts/mine_can_fields.py --pilot-root PILOT_DIR --output OUTPUT_DIR` after `mine_can_patterns.py` has frozen `assignment.json` under each set directory. `--normal-only` skips attack files. The runner verifies all three normal-role hashes before fitting. It uses the same two references, one calibration file, and one normal-test file as the pattern and dynamics pilots. Attack files are diagnostic; suppression files are skipped because they have no labeled attack frames. Masquerade frame labels are reported as diagnostics, not interval recall.

The version 1 algorithm reads each full capture in stable timestamp order, takes the first 2,000 valid payloads per ID in each reference, and searches byte-aligned 8-bit and half-byte 4-bit fields. A half-byte becomes a **counter candidate** if enough value changes advance by one modulo 16 in both references. A byte becomes a **smooth-byte candidate** if it changes often enough, has at least eight distinct values, and has small reference deltas in both references. The scorer flags a counter candidate when a changed value fails to advance by one, and a smooth byte when a new absolute delta exceeds a reference-derived limit. Unchanged counter values are allowed. Rules and thresholds never use calibration, normal test, or attack traffic. These are anonymous statistical roles, not decoded speed, RPM, or verified protocol counters. The pilot does not yet infer bit-field boundaries, endianness, signedness, operating mode, or cross-field consistency.

`profile.json` records fixed settings, source hashes, sampled-ID counts, rules, and per-reference support. `calibration.jsonl` and `normal_test.jsonl` report 100 ms bin counts. `report.json` gives rule counts, covered frame fraction, normal alert bins per hour, and attack diagnostics. It counts overlapping attack and alert bins, which is not episode recall or a fixed false-alarm-budget comparison with the rhythm model.

## First normal-only run, 2026-10-04

The deterministic `scripts.mine_can_patterns.assign` assignment was frozen under `/fs/ess/PAS1266/graphids/runs/experimentruns/can_fields_pilot/assignments`. Output is under `/fs/ess/PAS1266/graphids/runs/experimentruns/can_fields_pilot/output`. This is a diagnostic pilot; independent source sessions remain unverified.

| Set | Counter candidates | Smooth-byte candidates | Normal-test covered frames | Normal-test counter alert bins | Normal-test numeric alert bins | Normal-test hours |
|---|---:|---:|---:|---:|---:|---:|
| 01 | 1 | 1 | 8.4% | 0 | 0 | 0.243 |
| 02 | 1 | 0 | 1.0% | 0 | 0 | 0.080 |
| 03 | 3 | 4 | 2.0% | 9 | 0 | 0.140 |
| 04 | 5 | 1 | 4.6% | 0 | 0 | 0.058 |

The set 03 counter rate is about 64 alert bins/hour on its normal test. Set 04 has one counter alert bin in only 0.013 calibration hours. Zero counts elsewhere have weak statistical meaning because exposure is short and coverage is low. A spot check of speed, RPM, interval, and timed masquerade files found no useful detection by these rules. Thus this extraction is **not** a competitive payload detector and does not establish cross-vehicle physical semantics. It shows that strict byte-aligned smoothness and 4-bit increment rules cover too little traffic to justify replacing the rhythm baseline.

## Next gate

Inspect bit-flip structure and infer candidate field boundaries within each ID, including 12- and 16-bit values and signed/endianness alternatives. Fit candidate roles on the two references only, retain per-rule normal exposure and coverage, then compare with rhythm and per-ID timing at the same source-selected alert budget. Operating-mode conditioning and cross-field relationships should follow only after reliable field extraction. If no field inventory transfers across the four vehicles, use normalized payload-change statistics as behavioral inputs to the existing model rather than treating discovered fields as semantic mappings.
