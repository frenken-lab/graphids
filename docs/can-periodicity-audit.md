# Post-hoc CAN periodicity audit

Run `python scripts/audit_can_periodicity.py --pilot-root PILOT_DIR --output REPORT.json`. This reads all four normal captures per set using the frozen assignment, including calibration and normal test. It is **descriptive only**: those held-out files must not be used to select detection rules or thresholds.

For each ID and file, the audit samples at most 2,000 positive interarrival gaps. An ID is called periodic here when at least 30 sampled gaps are available, median absolute deviation divided by median is at most 0.15, and the 90th-percentile gap divided by median is at most 1.5. The reported period band is the nearest declared center within 20%; `other` is not a common period. A four-file stable ID must meet this rule in all four normal files and have maximum/minimum median period at most 1.2.

| Set | Four-file stable IDs | Their fraction of normal-test frames | Reference phase-locked pairs | Stable on normal test |
|---|---:|---:|---:|---:|
| 01 | 50 | 99.99% | 90 | 90 |
| 02 | 61 | 99.99% | 187 | 187 |
| 03 | 97 | 99.99% | 644 | 574 |
| 04 | 13 | 10.08% | 2 | 2 |

All four sets contain periodic streams near 20 ms, 50 ms, 100 ms, 500 ms, and 1 s in every normal file. The set 04 normal test also has roughly 31% of frames in a 10 ms band and 50% in a 20 ms band. Its short `accessory-3` reference has only 13 IDs, so a rule requiring evidence in both references excludes most of its normal-test traffic even though its `attack-free-3` reference contains 54 IDs.

For the phase audit, candidate pairs have similar median periods in both references. Each target frame is expressed as a phase in the trigger's cycle. A pair is retained descriptively when phase concentration is at least 0.9 in both references and circular mean phase differs by at most 0.1 cycles. Normal-test stability requires concentration at least 0.8 and mean phase within 0.15 cycles of the first reference. These pairs are scheduling regularities, not evidence that one ECU sends a request to another.

The earlier pair miner uses an absolute lag window of 100–200 ms and a half-capture circular shift control. For dense periodic IDs, almost any target has a previous trigger within 100 ms, so both real and shifted completion can approach 100%. A half-capture shift can also be close to an integer number of periods and leave phase intact. Thus zero selected pairs in set 04 does **not** mean zero temporal relationships. The phase-lock audit finds two reference-stable ordered pairs, `6CF → 6D1` and `6D1 → 6CF`, with periods near 1.2 s; both remain phase-concentrated on normal test.

The next detector experiment should compare a period-normalized phase residual with per-ID timing at the same normal-alert budget. Learn from references only, retain an unmatched option for IDs absent from a short reference, and score calibration and normal test separately. These descriptive results must not be retroactively used to tune the existing held-out test evaluation.
