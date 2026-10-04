# CAN behavior discovery module

`graphids.core.data.discovery.behavior` is the standalone reference-profile API for future graph-mamba inputs. The active CLI is `python scripts/mine_can_behavior.py --pilot-root ASSIGNMENTS --output OUTPUT` (or omit `--pilot-root` to create a deterministic 2/1/1 assignment). The four earlier pilot CLIs now call this unified runner. Their older `run_set` functions and the relocated `patterns`, `dynamics`, `fields`, and `bitfields` modules remain available for historical pilot tests, but are not the graph-facing API.

## Contract

- `load_capture(path)` parses raw frames once, preserves text CAN IDs and source-row order for equal timestamps, records SHA256 and validity diagnostics, and sorts chronologically. Invalid payloads remain arrivals but reset payload-change history. Each capture starts fresh state.
- `fit_profile([reference_1, reference_2], BehaviorConfig(...))` fits only on two distinct, attack-free normal captures. It yields a version-2 `BehaviorProfile` with frozen source hashes, slot order, timing and broad payload rules, bounded field hypotheses, directed relations, and motif templates. `BehaviorProfile.to_dict()` and `.from_dict()` serialize the artifact. Reference capture/session independence still needs a data audit beyond hash comparison.
- `BehaviorTracker.step(frames, bin_close, profile, state)` consumes one closed bin and returns fixed-size node, relation, and motif feature arrays, masks, separate family scores, and state. Pass empty bins to advance silence deadlines. Construct a new `BehaviorState` at a recording boundary. `score_capture` applies the same step contract to an entire capture.
- Node channels are `arrival_count`, `change_count`, `age_seconds`, `payload_bit_flips`, `payload_deviation`, `run_length`, `run_deviation`, `change_rate_deviation`, `field_violations`. Relation channels are `target_count`, `lag_violations`. Motif channels are `completed`, `overdue`, `active_prefixes`. Slot keys and rules are held in the profile; padded slots have false masks. The outputs are ordinary Python values and can later be converted to tensors without importing model code into discovery.

## Selection and interpretation

Timing, payload, and field candidates require support in both reference captures. Field search considers nibble-aligned 4/8/12/16/24/32-bit windows under both byte orders and signedness, then validates shortlisted candidates over full references. It retains alternatives instead of asserting one decoding. Labels such as `speed`, `rpm`, and `steering_angle` are behavior-based hypotheses. Their confidence is a stability score, **not** the probability that the physical label is correct; `unknown` remains available. External DBC, diagnostic, or sensor evidence would be needed to verify a label.

Directed relations use the dynamics pilot's cross-capture support and shifted-stream control. The template library composes accepted pairs into bounded pairs, chains, forks, joins, and returns; each motif is checked across references against a within-capture circular-shift control. The tracker evaluates timestamps and emits overdue evidence only after the relevant deadline. Capacity settings bound node, relation, motif, and active-prefix counts; `prefix_dropped` exposes tracker overflow. These relations are statistical dependencies, not physical causality.

The unified runner verifies every frozen normal-role hash before fitting. `profile.json` contains definitions from references only; `tables/*.jsonl` contains features and independent scores for calibration, normal test, and optional diagnostic attack files; `report.json` records counts, positive-evidence bins per hour, and normal exposure. Positive evidence is not a calibrated alert threshold. Output version 2 is intentionally separate from historical pilot artifacts. Attack labels do not select rules or thresholds, and no graph-mamba model or temporal dataset preprocessing consumes the new output yet.

## Full-capture smoke check

The default extractor was fitted separately on each set's two frozen normal references and scored on its disjoint normal-test file. Counts below are **bins with positive evidence**, not thresholded alerts or false-alarm estimates.

| Set | Fit seconds | Nodes | Fields | Relations | Motifs | Normal bins | Payload | Relation | Motif | Timing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 01 | 62 | 51 | 4 | 25 | 9 | 8,750 | 6,245 | 0 | 0 | 1 |
| 02 | 43 | 62 | 4 | 32 | 36 | 2,869 | 2,868 | 226 | 23 | 0 |
| 03 | 69 | 64 | 39 | 32 | 44 | 5,039 | 5,037 | 8 | 150 | 1 |
| 04 | 18 | 54 | 20 | 11 | 0 | 2,083 | 783 | 6 | 0 | 0 |

Set 04 retained 11 relations and no motifs. The initial zero-relation result came from ranking pair candidates by pooled frequency before checking that an ID appeared in both references. Selecting shared IDs before applying the cap corrected it. Motifs apply a separate trigger-root completion gate. For example, the accepted set-04 `152 -> 660` arrival relation has near-complete target-conditioned support, while only 81/2,000 and 77/2,000 sampled `152` triggers in the two references are followed by `660` inside the narrow lag window. It is a phase relation, not a reliable expected-successor rule. Motif root sampling uses deterministic random indices so periodic stride aliasing cannot erase an entire phase. Positive payload evidence is common even on normal traffic, especially in sets 02 and 03; pooling any positive rule as an alert would be unusable. These results establish execution and coverage behavior, not a detector operating point or transfer quality. A later source-validation or disjoint-normal calibration stage must set thresholds and test incremental detection value before graph-mamba integration.

## Dataset availability

The default 2/1/1 runner covers `set_01` through `set_04`. The raw lake also contains `hcrl_ch` and `hcrl_sa`, whose CSV columns can be parsed by the shared capture layer. They cannot meet this runner's independent-reference contract: the `train_02_with_attacks/attack-free*.csv` files are exact SHA256 duplicates of their `train_01_attack_free` counterparts. HCRL-CH has one unique normal capture; HCRL-SA has one unique normal capture each for Kia and Sonata. The runner reports this explicitly if either HCRL dataset is selected. Creating additional roles by splitting a file or mixing Kia and Sonata would change the reference protocol and is not done here.
