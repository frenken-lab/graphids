# Historical CAN behavior profile experiment

This page archives the retired version-11 normal-profile experiment. Its `can_discovery.behavior` package and `scripts/mine_can_behavior.py` runner were removed. The reusable `relationships.behavior` timing and motif modules remain, as do the structural and physical benchmark runners. The APIs and commands below describe historical artifacts and are no longer executable.

## Import map

- `can_discovery.preprocessing`: `capture` (raw frames and a lazy event view), `message_types`.
- `can_discovery.decoding`: `signals`, `field_values`, `dbc`.
- `can_discovery.relationships`: `behavior/` for timing and motifs, `physical/` for roles and factors, and `shared/` for alignment and evidence records.
- `can_discovery.behavior`: profile fitting, payload rules, tracking, and normal-reference orchestration. The `can_discovery` root formerly re-exported the normal-profile API. Profiles from versions 7–10 must be regenerated; the loader accepts version 11.

## Contract

- `load_capture(path)` parses raw frames once, preserves text CAN IDs and source-row order for equal timestamps, records SHA256 and validity diagnostics, and sorts chronologically. Invalid payloads remain arrivals but reset payload-change history. Each capture starts fresh state.
- `fit_profile([reference_1, reference_2], BehaviorConfig(...))` fits only on two distinct, attack-free normal captures. It yields a version-11 `BehaviorProfile` with CAN-D tokenization evidence by default, frozen source hashes, slot order, timing and broad payload rules, verified protocol fields, multiple provisional physical-role hypotheses, factor evidence, weighted compatible combinations, directed relations, and motif templates. `BehaviorProfile.to_dict()` and `.from_dict()` serialize the artifact. Reference capture/session independence still needs a data audit beyond hash comparison. Target DBCs are confined to post-fit benchmarks; calibrate roles from independently checked source vehicles with `fit_source_calibration` and activate eligible factor weights with `fit_source_factor_validation`.
- `BehaviorTracker.step(frames, bin_close, profile, state)` consumes one closed bin and returns fixed-size node, relation, and motif feature arrays, masks, separate family scores, and state. Pass empty bins to advance silence deadlines. Construct a new `BehaviorState` at a recording boundary. `score_capture` applies the same step contract to an entire capture.
- Node channels are `arrival_count`, `change_count`, `age_seconds`, `payload_bit_flips`, `payload_deviation`, `run_length`, `run_deviation`, `change_rate_deviation`, `field_violations`, and `physical_role_violation`. Relation channels are `target_count`, `lag_violations`. Motif channels are `completed`, `overdue`, `active_prefixes`. The physical score is the weighted expectation of per-hypothesis update or state violations across retained combinations. Slot keys and rules are held in the profile; padded slots have false masks.

## Selection and interpretation

Timing, payload, and field candidates require support in both reference captures. `decoding.signals.propose_fields` enumerates overlapping byte orders, signs, widths, and padding alternatives before physical interpretation. Consecutive unsampled frames in both references verify counter increments; byte checksums require a held-out calculation. Only these verified protocol bits are reserved from physical-role selection. `relationships.physical.combination_ranker.fit_physical_roles` scores explicit speed, RPM, gear, acceleration, steering, brake, and throttle properties, collapses observed value aliases for shortlist diversity, scores a few time-aligned relations against shifted controls, and searches compatible assignments with a null option per role. A role score or ensemble weight is **not** a calibrated probability or a confirmed physical label. Unknown names stay unresolved. `scripts/experiment_can_physical.py` reports exact DBC layout and held-out value equivalence separately; it opens the target DBC only after fitting.

`relationships/shared/signal_time.py` supplies masked causal alignment; the optional catch22, STUMPY, and PCMCI experiments have been retired from active code. Historical empirical notes remain archived.

Directed relations use the dynamics pilot's cross-capture support and shifted-stream control. The template library composes accepted pairs into bounded pairs, chains, forks, joins, and returns; each motif is checked across references against a within-capture circular-shift control. The exploratory tracker evaluates timestamps and emits overdue evidence only after the relevant deadline. Capacity settings bound node, relation, motif, and active-prefix counts; `prefix_dropped` exposes tracker overflow. These relations are statistical dependencies, not physical causality.

The normal-data runner verifies every frozen normal-role hash before fitting. `profile.json` contains definitions from references only; `tables/*.jsonl` contains exploratory features and evidence for disjoint calibration and normal-test files; `report.json` records counts, positive-evidence bins per hour, and normal exposure. Positive evidence is not a calibrated alert threshold. Output version 11 is separate from historical pilot artifacts. The historical runner did not read attack files; no graph-mamba model or temporal dataset preprocessing consumes this output.

The pinned four-vehicle structural audit and cross-vehicle held-out results are in [the version-5 empirical note](empirical-notes/2026-10-04-compositional-can-v5.md). `scripts/benchmark_can_physical_vehicles.py` reruns Prius, Civic, and Sonata from the `benchmarks.json` emitted by `scripts/prepare_can_source_calibration.py`.

Version 6 uses a threshold pool with explicit role and message-family budgets. Physics factors test masked, lagged prediction against target history and time-shift controls; gear and steering factors require corresponding motion. All roles can be null. Discovery scores may use new factors immediately, while live tracker combination weights use only factors validated on third trips in at least two source vehicles. New yaw and lateral-acceleration roles remain discovery-only in the tracker. The optional `causal-diagnostic` extra installs pgmpy for `scripts/diagnose_can_physical_pgmpy.py`; its time-sliced edges are conditional-dependence diagnostics, not physical field labels. [The version-6 empirical note](empirical-notes/2026-10-04-physical-v6.md) records held-out results.

## Full-capture smoke check

In a historical smoke check, the extractor was fitted separately on each set's two frozen normal references and scored on its disjoint normal-test file. Counts below are **bins with positive evidence**, not thresholded alerts or false-alarm estimates.

| Set | Fit seconds | Nodes | Fields | Relations | Motifs | Normal bins | Payload | Relation | Motif | Timing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 01 | 62 | 51 | 4 | 25 | 9 | 8,750 | 6,245 | 0 | 0 | 1 |
| 02 | 43 | 62 | 4 | 32 | 36 | 2,869 | 2,868 | 226 | 23 | 0 |
| 03 | 69 | 64 | 39 | 32 | 44 | 5,039 | 5,037 | 8 | 150 | 1 |
| 04 | 18 | 54 | 20 | 11 | 0 | 2,083 | 783 | 6 | 0 | 0 |

Set 04 retained 11 relations and no motifs. The initial zero-relation result came from ranking pair candidates by pooled frequency before checking that an ID appeared in both references. Selecting shared IDs before applying the cap corrected it. Motifs apply a separate trigger-root completion gate. For example, the accepted set-04 `152 -> 660` arrival relation has near-complete target-conditioned support, while only 81/2,000 and 77/2,000 sampled `152` triggers in the two references are followed by `660` inside the narrow lag window. It is a phase relation, not a reliable expected-successor rule. Motif root sampling uses deterministic random indices so periodic stride aliasing cannot erase an entire phase. Positive payload evidence is common even on normal traffic, especially in sets 02 and 03; pooling any positive rule as an alert would be unusable. These results establish execution and coverage behavior, not a detector operating point or transfer quality. These counts are historical smoke-check diagnostics; the current research gates are structural validity and independent-normal relationship replication.

## Dataset availability

The default 2/1/1 runner covers `set_01` through `set_04`. The raw lake also contains `hcrl_ch` and `hcrl_sa`, whose CSV columns can be parsed by the shared capture layer. They cannot meet this runner's independent-reference contract: the `train_02_with_attacks/attack-free*.csv` files are exact SHA256 duplicates of their `train_01_attack_free` counterparts. HCRL-CH has one unique normal capture; HCRL-SA has one unique normal capture each for Kia and Sonata. The runner reports this explicitly if either HCRL dataset is selected. Creating additional roles by splitting a file or mixing Kia and Sonata would change the reference protocol and is not done here.
