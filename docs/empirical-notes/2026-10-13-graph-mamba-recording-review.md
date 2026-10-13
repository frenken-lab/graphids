# Graph-Mamba recording review and first real-data pilot

The review authorizes only a capture-level CTAT normal/DoS pilot from the
2011 Chevrolet Impala to the 2017 Subaru Forester. It does not approve the
all-six preset or establish real-data detection-quality acceptance.

## Evidence and integrity

CPU Slurm job `56109324` inspected 207,033,146 rows across 252 canonical files
from the six registered datasets. The original inventory has 333 paths, with
81 exact duplicate aliases. Canonical file hashes were rechecked. Aliases are
grouped, not additional independent recordings; selected manifest paths are
also individually hash checked during freezing.

The new audit samples content-defined windows of 64 ordered ID/payload pairs
from zero-event-label traffic, omitting absolute clocks and injected rows.
This found 109 candidate reuse pairs. Sampling is a screen for review, not
proof of independence or an exhaustive alignment algorithm. Zero labels are
used for this screen only and never establish trusted normality.

Examples include 3,834 matching windows between Impala `attack-free-1` and
`standstill-1`, and 1,019 between `attack-free-2` and
`targeted_payload_fuzzing-1`. The latter examples have a 100,000-row offset.
Silverado normal recordings also share thousands of windows with timed
masquerade recordings. These captures cannot be assigned independent roles
by filenames or whole-file hashes alone.

All 16 normal CTAT file prefixes match their vehicle-specific original
`can-dataset` logs, including relative clocks. For the 12 selected captures,
the entire timed-frame multiset was compared with downloaded, commit-pinned
original logs at candump's microsecond precision. Original acquisition
intervals are disjoint within each selected vehicle. No sampled reuse pair
joins two selected captures. All selected local captures have monotonic finite
clocks, valid classic-CAN hex payloads, and binary event labels. Selected attack
CSVs are byte-identical to their upstream `can-ml` labeled files.

The upstream [conversion helper](https://bitbucket.org/brooke-lampe/can-ml/src/7268a6f564b7a1d4ad2bec089b60abaf68e18bdf/helper/convert_advanced.py)
subtracts a recording-specific offset to start at `1672531200`. This explains
the common curated origin without implying common acquisition. It copies
candump's hex payload text. The [label helper](https://bitbucket.org/brooke-lampe/can-ml/src/7268a6f564b7a1d4ad2bec089b60abaf68e18bdf/helper/label.py)
sorts timestamps without specifying a stable sort. The selected original DoS
logs have 12, 37, 8, and 3 backward steps respectively, with a largest step
of about 6 microseconds. Their curated copies are already monotonic and may
reorder clock ties. Full timed-frame correspondence accounts for this documented
upstream curation; this review never sorts or rewrites raw files. Payload lengths
preserve candump serialization, with no independent DLC column available.

Label verification means the curator's event-label provenance and serialization
were reviewed; it does not independently certify every event as ground truth.
No trusted attack intervals were inferred from event labels.

The upstream commits are `can-dataset`
`c7e146cc75a5bac5d79e2a064e300e2276d24c51` and `can-ml`
`7268a6f564b7a1d4ad2bec089b60abaf68e18bdf`. Selected normality follows the curated
attack-free subset and correspondence to original normal captures, rather than
zero labels alone. A session key identifies an original acquisition capture.
Distinct driving trips or days are not established by this pilot.

## Frozen roles

Roles were chosen from normal operating modes and exposure, before any model
scores. Source uses the first numbered normal/DoS captures for training and
the second for validation. Target holds out the longer driving capture for
normal testing, uses the remaining driving capture plus the shorter accessory
capture for context, and reserves the longer accessory capture for calibration.
The target calibration is a different operating mode with only 13 observed IDs;
it cannot establish a general target threshold guarantee. Primary thresholds
remain source-normal validation thresholds in both modes.

| Vehicle | Capture(s) | Role | Exposure (s) |
| --- | --- | --- | ---: |
| Impala | accessory-1 | train_reference | 143.641 |
| Impala | accessory-2 | val_reference | 156.532 |
| Impala | attack-free-1, DoS-1 | train_query | 1,350.216 + 56.039 |
| Impala | attack-free-2, DoS-2 | val_query | 874.927 + 197.102 |
| Forester | attack-free-4, accessory-4 | target_reference | 208.249 + 47.981 |
| Forester | accessory-3 | target_calibration | 196.546 |
| Forester | attack-free-3 | target_normal_test | 572.020 |
| Forester | DoS-3, DoS-4 | target_attack_test | 177.878 + 982.576 |

Forester normal files are in `set_04/train_01_attack_free`, while its selected
DoS captures are in `set_02/test_04_unknown_vehicle_unknown_attack`. Their
vehicle identity follows full correspondence to original Forester logs. Registry
set names are not dataset families. Impala source files are in `set_01`.
Only `set_01` is selected for source training; target roles explicitly include
their own registry sets. Suppression, masquerade, malformed payloads,
decreasing-clock curated files, and unreviewed HCRL captures remain excluded.

The target normal test has only 0.159 hours of exposure. Report counts and
exposure; this pilot cannot precisely certify a low false-alert rate.

## Artifacts and reproduction

Review artifacts are outside Git at:

```text
/fs/ess/PAS1266/graphids/dev/rf15/experimentruns/graph-mamba-real-data-review-v1/
  recording-review.json
  pilot-provenance.json
  can-recording-manifest-v1.json
  upstream/sources.json
  upstream/normal-origins.json
  upstream/pilot-full-sources.json
  freeze-56109477.log
```

Run `scripts/can_recording_review.py` in a CPU allocation with `--raw-root`,
`--inventory`, and `--output`. It preserves the raw lake and never assigns roles.
`scripts/can_pilot_manifest.py` freezes the explicitly reviewed subset only
after full upstream correspondence, individual selected-path hashes, local
integrity, sequence-screen checks, label-file hashes, and acquisition interval
checks pass. The review remains restricted to this subset; the broader audit
still has `provenance_reviewed: false` and `fit_ready: false`.

The preset is `configs/experiments/graph_mamba_ctat_pilot.yml`. The native
one-A100 diagnostic driver is `scripts/graph_mamba_real_pilot.py`, submitted
as Ascend job `7819680`. It keeps default bins, width, layers, and graph budgets,
fits 32 sampled windows for one epoch, validates whole source recordings,
restores the checkpoint, and evaluates whole target queries in both modes.
It records source census, packed workload, calibration, sampled training work,
and expected-versus-scored coverage. This is a real-data execution diagnostic;
32 optimizer steps do not establish trained detection quality.

Pilot outputs are at
`/fs/ess/PAS1266/graphids/dev/rf15/experimentruns/graph-mamba-real-pilot-v1/`.
`pilot.json` records stage, implementation commit, driver hash, and worktree
status. A new run records implementation commit
`d867c3737b8ac69d9fa181b156a8e1b421b0aa7f`; added diagnostic drivers are saved
with the artifacts. Model implementation files are unchanged.

CPU preflight job `56109538` completed successfully. It built the shared caches
and checked every one of the 1,099 source training windows. Source-derived
anchor and transient capacities are both 64; the largest observed live source
ID set is 52. Maximum packed source work is 13,312 node-bin tokens and 104,448
graph messages, below the default budgets. The largest source snapshot array
cache is 762,469,234 bytes, below the 4 GiB limit. Results are in
`graph-mamba-real-pilot-v1/preflight/pilot.json` and `source-census.json`.
The A100 pilot remains queued for priority with no start estimate; no new
checkpoint, target scores, or real-data coverage result exists yet.

Regression job `56109517` passed 73 tests with three CUDA tests skipped. Ruff,
documentation lint, strict MkDocs build, and whitespace checks passed. The new
tests check sequence reuse across offsets/injection, raw hash changes, malformed
payloads, clock backsteps, and complete upstream timed-frame correspondence.

HCRL remains outside this reviewed subset. Its Car-Hacking train/test attack
slices come from common recordings, and the available normal files do not
supply three independent target-normal captures per vehicle. The bundled
Car-Hacking link uses the singular `Dataset` URL, which currently resolves to
OTIDS; the corresponding [Car-Hacking description](https://ocslab.hksecurity.net/Datasets/car-hacking-dataset)
and [Survival Analysis description](https://ocslab.hksecurity.net/Datasets/survival-ids)
are separate sources. Shared vehicle names across HCRL families need acquisition
identity review before any pooled source/target split.
