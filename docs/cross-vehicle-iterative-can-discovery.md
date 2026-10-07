# Iterative cross vehicle CAN discovery methodology and implementation

## Purpose and research claim

Test whether a shared, versioned body of evidence improves physical CAN signal discovery across vehicles with incomplete labels. Each vehicle retains its own bit layouts, scales, operating contexts, and missing roles. The shared model contains calibrated evidence about roles, relationships, and transferable layout conventions. A new vehicle can contribute evidence to a later round without converting a plausible decoding into ground truth.

The primary claim is measurable improvement over a **one-pass source-only decoder** in held-out exact field layout, raw-value agreement, and physical-role retrieval. Relationship prediction is a separate outcome; a physics fit or event-graph edge does not establish a field name. Compare iterative rounds with no-sharing, one-pass sharing, and oracle-anchor conditions under the same candidate budget and recordings.

This proposal extends the existing [research aims](../can_discovery/RESEARCH_AIMS.md) and [pipeline contracts](can-discovery-pipeline.md). The current code transfers source DBC calibration and frame matches in one pass. It does not yet maintain a multi-round shared evidence state.

## Unit of inference

For vehicle (v), message family (m=(bus, ID, length)), and candidate field (c), retain the bit start and width, byte order, signedness, possible encoding, raw value series, and mutually exclusive layout memberships. A role assignment (z_{vc}) may be speed, acceleration, yaw, a related proxy, or unknown. Vehicle-specific scale, offset, lag, and physical parameters are nuisance variables, not shared constants.

Represent evidence at three levels:

| Level | Examples | Permitted use |
| --- | --- | --- |
| Grounded | DBC paired with its recording; diagnostic PID; synchronized GPS or IMU; controlled action with independently recorded outcome | Train and assess role or scale calibration, subject to source quality and capture coverage |
| Replicated provisional | Candidate group passes predeclared physics and event tests on independent recordings and contexts | Rerank alternatives as a soft hypothesis; contribute bounded, provenance-linked evidence to later rounds |
| Exploratory | One trip, one correlation, untested format match, or graph edge without controls | Retain for review and candidate generation; do not update shared role weights |

Keep DBC annotations granular. A DBC may validate one field while leaving other fields unknown; an absent DBC entry is not a negative label. Store whether a named field was active in each recording and whether its decoded values were assessable.

## Shared model and local inference

Use a hierarchical factor model rather than copying decoded bit positions between vehicles. In conceptual form,

\[
P(Z, C, \theta \mid X, A) \propto
\prod_v P(C_v \mid X_v, \phi_{\mathrm{layout}})
P(Z_v \mid C_v, X_v, \phi_{\mathrm{role}})
P(A_v \mid Z_v, C_v)
P(X_v \mid Z_v, C_v, \theta_v, \phi_{\mathrm{relations}}).
\]

Here (X_v) is a vehicle's normal recordings, (A_v) is its available grounded evidence, (C_v) is its compatible candidate layout, (Z_v) its role assignments, (	heta_v) its scales, lags, and vehicle parameters, and (phi) the shared evidence model. The equation is an implementation target, not a claim that the full posterior can currently be estimated. A bounded top-(K) search over compatible field groups with an unknown assignment is a practical first implementation.

Share **distributions and reliability**, not target labels: role-conditioned value behavior; factor effect size and failure frequency by context; plausible width and alignment distributions by platform; and source-specific frame-match reliability. A same-ID DBC match is a candidate prior requiring payload-profile agreement. Do not transfer wheelbase, steering ratio, bit position, or scale as universal constants.

## Physical and event evidence without full labels

Start with the measured chassis core where the recordings have coverage. Candidate pairs can test longitudinal (a_x \approx d v/dt); candidate trios can test low-slip (a_y \approx v r) and yaw response to speed and steering. Wheel speeds can test shared movement and turn asymmetry. Gear-conditioned RPM/speed can be used only when shifts and multiple gear states are observed. Fit unknown affine transforms and plausible lags on fitting recordings, then evaluate residual and predictive gain on a separate recording. Restrict each equation to its valid operating regime. An unavailable regime yields no factor, not a failed candidate.

Event and timing evidence is another view: arrivals, payload changes, braking episodes, turn episodes, and lagged candidate values. Compare every edge or equation with target own-history, shifted or schedule-preserving controls, and matched activity windows. Require effective support by recording and vehicle, not overlapping time bins. Cap contributions from factors that derive from the same underlying motion or anchor so one event cannot be counted repeatedly.

An equation can identify a compatible **group** while leaving individual names, bit widths, and scales uncertain. For example, if both speed and acceleration scales are free, derivative agreement alone may admit multiple affine or width-equivalent solutions. A grounded GPS speed or DBC field can fix one part of this ambiguity. Without such an anchor, retain equivalent assignments and report value equivalence separately from exact layout and named-role accuracy. The event graph is an inference constraint and held-out prediction measure; independent DBC, diagnostic, sensor, or controlled-maneuver evidence remains the role benchmark.

## Iteration protocol

1. **Freeze round zero.** Fix vehicle and recording splits, DBC revisions, capture hashes, candidate budgets, factor definitions, controls, and promotion measures. Reserve a final vehicle cohort and final trips for evaluation. A development failure may motivate a later method version, but not a rewrite of the sealed round.
2. **Generate local candidate pools.** Tokenize each vehicle's fitting recordings; retain alternative layouts, byte order and signedness, protocol fields, unresolved bits, and unknown roles. Verify counters and checksums independently. Exclude attack recordings and transport messages according to the existing contracts.
3. **Fit shared evidence from grounded sources.** Use only source-vehicle DBC–recording pairs and other declared anchors. Calibrate role priors, physical factors, frame matching, and error or abstention rates with leave-one-vehicle-out and independent-trip checks. Keep source identity and evidence lineage.
4. **Infer locally with the frozen shared model.** Rank compatible candidate groups and null assignments for each target. Fit target-specific scales and lags only on its fitting recordings. Score relationships in different regimes; do not let the target DBC into fitting in the source-only track.
5. **Validate provisional findings.** Test frozen candidates and factors on development recordings. Admit a finding to the next shared version only if the factor replicates, clears its predeclared controls, and improves calibration or retrieval in source-side cross-validation. Promotion increases a bounded soft weight; it does not create a verified DBC label. Record failed and unassessable tests.
6. **Run another round.** Refit shared weights from grounded data plus admitted provisional evidence, then rerank affected vehicles. Track every changed assignment and the independent evidence that changed it. Stop when ranks and coverage stabilize, the source-side gate fails, or the round budget is exhausted. Assess the final frozen model once on the sealed cohort.

Use an evidence ancestry graph: each factor points to its fields, captures, anchors, and parent hypotheses. A target hypothesis cannot gain independent support through a cycle that returns to itself. Related factors from one anchor or driving episode share a contribution cap. A newly inferred field can help another vehicle as a soft source hypothesis only after cross-recording validation and calibration; its status remains provisional until independent truth is available.

## Benchmark design

The first test should **mask known labels**, not wait for genuinely unlabeled vehicles. On a DBC-backed vehicle, hide one speed or acceleration field during fitting, retain the other as an explicitly declared target anchor, and ask whether shared evidence recovers the hidden field on a disjoint trip. This is an anchor-assisted condition. Then hide both and test a source-only condition with group ranking and an unknown option. Repeat leave-one-vehicle-out so a vehicle never validates a shared prior learned from its own DBC. Compare rounds 0, 1, and 2 at the same candidate budget and runtime cap; never combine anchor-assisted and source-only scores in one aggregate.

Report per vehicle and role: candidate coverage, top-1/top-(K) exact layout, held-out raw-value agreement, name rank, scale and lag error where truth exists, null and ambiguous rates, factor availability, relationship gain over controls, and computation. Report losses as well as gains. Include shuffled source roles, shuffled or shifted relationships, unary-only, one-pass sharing, and oracle-anchor baselines. Use a deliberately corrupted source label to measure error propagation. A later genuinely unlabeled cohort can test stability and downstream utility, but cannot establish named-role accuracy on its own.

The first factor should be speed–longitudinal acceleration. In the current [source-factor audit](empirical-notes/2026-10-04-physical-v6.md), it alone met the two-vehicle validation gate. [Coverage audit](empirical-notes/2026-10-05-physical-role-coverage-audit.md) shows that Prius and Sonata have the measured chassis core, while the complete powertrain graph lacks confirmed simultaneous roles. A broader chassis trio should be attempted only after speed–acceleration shows a gain. The existing [candidate-graph benchmark](empirical-notes/2026-10-05-can-relationships-v11/REPORT.md) did not pass its promotion gate; it is the baseline to beat.

## Implementation sequence

1. **Shared artifact schema.** Add a versioned artifact with source vehicle IDs, DBC and capture hashes, round number, candidate/factor definitions, calibrated weights, support, dependencies, uncertainty, and promotion status. Keep per-vehicle candidate registries separate. Extend `FieldRegistry` and the [signal catalog](../can_discovery/engineering/catalog.py) only through explicit schema versions.
2. **Evidence adapters.** Reuse `preprocessing.traces` for message grouping and `decoding.tokenization` and `decoding.signals` for candidates; `engineering.integrated` for frame matches and anchored decoding; `relationships.shared.signal_time` for aligned values; `relationships.behavior` for events; and `relationships.physical.physics_factors`, `vehicle_models`, and `source_calibration` for tested equations and weights. Add an explicit adapter that emits comparable factor records with source and dependency IDs.
3. **Round engine.** Implement `fit_shared_model(sources, round_state)`, `infer_vehicle(captures, shared_model)`, `validate_proposals(validation_captures)`, and `promote_round(evidence, policy)` as pure, artifact-producing stages. An orchestrator iterates them over a frozen manifest. Each stage reads a prior immutable version and writes a new version; no stage mutates a DBC or a previous round.
4. **Local search.** Extend the bounded combination ranker to retain top-(K) compatible layouts and role groups plus null. Fit scales and lags per vehicle; use source-calibrated factor likelihoods or normalized held-out residuals, with a dependency cap. Preserve competing assignments with close scores.
5. **Evaluation runner.** Add a label-mask manifest and round-by-round benchmark next to `scripts/benchmark_can_integrated.py` and `scripts/benchmark_can_candidate_graph.py`. Target DBCs are opened only by post-fit evaluation. Freeze and hash each round's shared artifact before target scoring.

The initial deliverable is a two-round speed–acceleration masked-label benchmark and its versioned evidence ledger. Promotion to a wider role graph requires improved held-out retrieval without a material per-vehicle regression or an increase in unsupported confident names.

The round engine should follow this order:

```text
state[0] = fit_grounded_sources(frozen_source_manifest)
for round in (0, 1):
    proposals = infer_all_vehicles(state[round], fitting_trips)
    evidence = validate(proposals, development_trips, controls)
    state[round + 1] = promote_soft_evidence(state[round], evidence)
    freeze_and_hash(state[round + 1], proposals, evidence)
evaluate(frozen_state, sealed_trips, masked_DBC_truth)
```

Each proposal record needs `vehicle`, `bus`, `id`, `length`, candidate bit coordinates, role alternatives including `unknown`, fitted scale and lag, layout membership, evidence IDs, parent hypothesis IDs, per-recording support, score components, status, and the round that first created it. A promotion record also needs the changed weight, source vehicles, validation hashes, gate result, and a reason when it abstains. This ledger makes round-to-round gains and error propagation auditable.

## Research basis

- [CANMatch](https://orbilu.uni.lu/handle/10993/48502) demonstrates reuse of known vehicle formats; the [CAN survey](https://rtcl.eecs.umich.edu/rtclweb/assets/publications/2023/ieeecomm23-buscemi.pdf) proposes expanding a DBC collection through multiple methods.
- [Multi-task system identification](https://arxiv.org/abs/2301.01430) supports sharing structure while retaining system-specific dynamics.
- [Partial-observation system identification](https://proceedings.mlr.press/v144/galioto21a.html) motivates explicit model and measurement uncertainty.
- [Weak-supervision dependency modeling](https://pmc.ncbi.nlm.nih.gov/articles/PMC6417840/) motivates tracking correlated evidence sources; [debiasing self-training](https://proceedings.neurips.cc/paper_files/paper/2022/hash/d10d6b28d74c4f0fcab588feeb6fe7d6-Abstract-Conference.html) motivates separating provisional predictions from their later use.
