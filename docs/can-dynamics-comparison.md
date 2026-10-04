# CAN dynamics comparison

Run `python scripts/compare_can_dynamics.py --pilot-root PILOT_DIR --output OUTPUT_DIR` from the repository root. The pilot directory must contain the frozen `assignment.json` for each set from `mine_can_patterns.py`. The runner verifies every normal-file SHA256 before fitting or scoring. Use `--raw-root` and `--sets` to override the default lake path and select sets.

This separate reference-conditioned experiment fits three rule families on the two reference files only:

- **Mode-aware per-ID timing:** classify each arrival by the count of bus arrivals in the preceding second relative to a reference-only median count. Fit an upper interarrival deadline for each ID and, where supported in both references, a separate low- and high-traffic deadline. Flag once when the next frame has not arrived by that deadline.
- **Cross-ID phase:** for sampled target arrivals, measure elapsed time since the last strictly earlier trigger arrival. Require support and lag stability in both references, compare against a half-capture circular shift, and keep at most 16 pairs. A target arrival outside its learned lag window increments the phase score.
- **Payload-change coupling:** use the same procedure on whole-ID payload-change events. This tests change timing, without claiming to decode any signal or to establish a causal ECU relationship.

The versioned `profile.json` stores source hashes, rules, candidate rejection reasons, and all fixed settings. `calibration.jsonl` and `normal_test.jsonl` contain one row per 100 ms bin with counts for all three methods. `report.json` gives normal exposure and alert bins per hour on the same normal files. A bin with one or more rule violations counts as one alert bin for that method. No attack labels or calibration/normal-test data select rules or limits.

The synthetic deletion chooses a frequent ID with an accepted timing rule in the normal test, removes that ID's frames for a declared interval, and reloads retained raw frames so payload-change events are recomputed. `synthetic_deletion.jsonl` contains its scores. The report gives the ID, interval, removed-frame count, normal alerts in the same interval, and delay to the first incremental synthetic alert. This is an explicitly synthetic diagnostic, not recall on the dataset's unlabeled suppression files. The deletion target is chosen from the normal test to ensure that frames can be removed; it does not affect fitting or thresholds.

These short normal captures cannot establish a low operational false-alert rate. A zero alert count over a fraction of an hour is weak evidence. The pair families can also flag many target frames after one trigger disappears, so alert-bin counts are correlated; report them alongside covered IDs, rule counts, and exposure.
