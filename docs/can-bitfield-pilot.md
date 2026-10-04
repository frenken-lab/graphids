# Calibrated anonymous CAN bit-field pilot

> Historical pilot: the active CLI now runs the unified version-2 discovery pipeline. See [CAN behavior discovery](behavior-discovery.md). The commands and report fields below describe the original pilot run.

Run `python scripts/mine_can_bitfields.py --pilot-root PILOT_DIR --output OUTPUT_DIR` from the repository root. The pilot uses the frozen 2/1/1 normal assignment produced for the pattern miner and verifies every normal-file SHA256 before fitting. The historical bit-field algorithm was independent of the version-1 byte/nibble pilot and now lives under `graphids.core.data.discovery.behavior.bitfields`.

The reference-only stage scans nibble-aligned 12- and 16-bit windows under both byte orders and signed/unsigned interpretations. It samples the first 1,000 valid payloads per ID in each of the two references to find fields with frequent, small changes, then validates every candidate against **all frames** in both references. A candidate with any jump beyond its reference-derived limit is rejected. The separate normal calibration capture selects candidates with at least 30 valid transitions and no jumps; at most two nonoverlapping bit ranges per ID survive. The untouched normal-test capture is then scored. The score counts 100 ms bins containing at least one selected-field jump. These candidates are anonymous statistical windows; neither field boundaries nor physical meanings are verified.

## Four-vehicle normal gate, 2026-10-04

The existing deterministic assignments were read from `/fs/ess/PAS1266/graphids/runs/experimentruns/can_fields_pilot/assignments`. Outputs are under `/fs/ess/PAS1266/graphids/runs/experimentruns/can_bitfields_pilot/output`.

| Set | Initial candidates | Full-reference stable | Calibration selected | Normal-test covered frames | Normal-test alert bins | Normal-test exposure |
|---|---:|---:|---:|---:|---:|---:|
| 01 | 10 | 8 | 2 | 8.4% | 0 | 0.243 h |
| 02 | 12 | 4 | 1 | 1.0% | 0 | 0.080 h |
| 03 | 59 | 45 | 14 | 16.8% | 0 | 0.140 h |
| 04 | 42 | 34 | 3 | 1.4% | 0 | 0.058 h |

Without full-reference validation, set 03 yielded 234 alert bins in 0.140 normal-test hours and set 04 yielded 11 in 0.058 hours. Thus the first 1,000 occurrences alone are insufficient to characterize a field. The validation gate removes those normal-test alarms, but zero alerts in these short captures cannot establish a low operational false-alarm rate.

A diagnostic spot check scored available `speed-3`, `rpm-3`, `interval-3`, and `masquerade-timed-1` files in the same-vehicle folders. No selected rule flagged a labeled attack bin. Set 03's timed masquerade file produced 30 alerts, all outside labeled attack bins. The selected windows therefore fail the attack-detection gate. No neural rhythm integration or shared physical-signal claim follows from these results. Candidate count and covered-frame fraction are structural diagnostics; they are not attack coverage.

## Decision

Keep the existing rhythm/timing model as the baseline. The next payload experiment should represent *changes* more broadly: bit-flip counts and locations, payload Hamming distance, repeated-payload run length, and per-ID change-rate deviations, normalized by normal reference behavior. These can cover many IDs without forcing an uncertain 12/16-bit interpretation. Compare that representation against rhythm alone at a source-selected alert budget, with the same capture split and an explicit attack-ID coverage audit. More decoded-field work requires either independently verified DBC/OBD/GPS/IMU anchors or longer normal captures spanning operating modes.
