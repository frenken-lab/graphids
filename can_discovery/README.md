# CAN discovery research package

`can_discovery` studies two questions independently of GraphIDS models:

1. **Reverse engineering:** Which message structures, bit fields, protocol fields, and physical-role hypotheses can be recovered from raw CAN recordings? Compare proposed layouts and values with independent DBC or sensor truth where available. Keep uncertain and unobserved fields unresolved.
2. **Relationships:** Which timing or value relationships repeat across independent normal recordings and, where observable, across vehicles? Compare each relationship with own-history prediction and shifted or schedule-preserving controls. Report context, lag, support, and failures separately.

The [research aims](RESEARCH_AIMS.md) set out the publication questions, the iterative decoding–relationship pipeline, current evidence gates, and the later GraphIDS handoff.

The [cross-vehicle iterative discovery methodology](../docs/cross-vehicle-iterative-can-discovery.md) specifies a proposed multi-round shared evidence model, treatment of missing labels, promotion rules, benchmark design, and implementation sequence. It is a research design; the current runners perform one-pass source transfer.

The [pipeline and module contracts](../docs/can-discovery-pipeline.md) define the unit of analysis, stage inputs and outputs, valid standalone and joint routes, and the distinction between CAN-D structural inference and PicaCAN's isolated graph baseline.

## Package shape

| Package | Responsibility |
| --- | --- |
| `preprocessing/` | Capture loading and per-family trace construction. |
| `decoding/` | Transport classification, CAN-D and other structural inference, source boundary models, field candidates, raw value decoding, and DBC evaluation. |
| `scoring/` | DBC-free role-guided scores and separate DBC-assisted anchors and source evidence. |
| `relationships/behavior/` | Timing and motif evidence learned from observations. |
| `relationships/physical/` | Physical roles, factors, calibration, and experimental graphs. |
| `relationships/shared/` | Field identity, alignment, lag joins, and common evidence records. |
| `experiments/` | Isolated paper baselines, including PicaCAN. |
| `engineering/` | DBC-assisted provisional signal catalogs and review queues. |

Decoding is one possible input to relationship discovery, not a prerequisite for every relationship. Arrival and whole-payload change relations can be studied without physical labels. A replicated predictive relation is not automatically causal or a shared physical invariant.

The standalone normal-profile experiment has been retired. The reusable timing and motif analyses remain in `relationships/behavior/`, and structural and physical discovery still have independent benchmark runners.

The current DBC annotation completion command is `python -m scripts.run_can_label_completion`. Run `prepare` for every vehicle-role case, then `rank` for every case, then `report`. Every command requires `--track unsupervised` or `--track dbc_assisted`, the staged manifest, pinned source manifest, and a fresh output root outside the repository; case phases also require `--case-index`. Artifacts are separated under `OUTPUT_DIR/unsupervised/` and `OUTPUT_DIR/dbc_assisted/`. Preparation admits message families directly from valid fitting frames present in both captures and excludes transport traffic. It calls `decoding.tokenization.tokenize(method='exhaustive')` to generate lazy, unscored field layouts, then scores them in `scoring.role_guided.score_family`. `can_discovery.prepare` builds unscored families and layouts; `label_completion.case` orchestrates vehicle-role artifacts and evaluation. The `unsupervised` track uses no DBC labels or DBC-trained votes for fitting or ranking, but it remains **role-guided** by a hand-written score profile. The `dbc_assisted` track calls `scoring.dbc_assisted` to exclude known other-role anchors and apply source contrast and speed–RPM relationship votes when their gates pass. DBC references are opened for evaluation only after each track's candidate scores have been fixed and are never fed back into the unsupervised track. CAN-D tokenization is used only by a separate DBC-assisted baseline control; it cannot gate either exhaustive candidate pool. Final routes remain sealed; historical development results have not passed the final-assessment gate.

For the pinned four-vehicle cohort, submit all 20 `prepare` cases for each track before that track's `rank` cases. Use `scripts/slurm/can_label_completion.sbatch` with `PHASE=prepare` or `PHASE=rank`, `TRACK=unsupervised` or `TRACK=dbc_assisted`, and `CAN_LABEL_OUTPUT_DIR` set outside the repository. After all rank cases complete for a track, run:

```bash
python -m scripts.run_can_label_completion report STAGED_MANIFEST SOURCE_MANIFEST OUTPUT_DIR --track unsupervised
python -m scripts.run_can_label_completion report STAGED_MANIFEST SOURCE_MANIFEST OUTPUT_DIR --track dbc_assisted
```

The generated artifacts use a new consolidated schema. Historical v2–v6 reports and summaries are retained for comparison; their per-case data was pruned from the repository.

The experimental integrated decoder combines CANMatch-style same-ID/DLC reference lookup, payload-profile checks, tokenization of unmatched families, field equivalence, and AutoCAN-style derivative/integral and wheel-mean evidence. Run `python -m scripts.benchmark_can_integrated MANIFEST --output-dir OUTPUT --track both --evaluate-dbc`. `source_only` reads other vehicles' DBCs but opens the target DBC only in evaluation; `dbc_assisted` explicitly uses the target DBC as fitting anchors. The assisted evaluation recomputes a pool with **all** target labels hidden, so anchor fields cannot inflate recovery. The version-1 artifacts retain uncertain same-ID matches as alternatives and all proposed labels as review entries. Physical scaling from DBCs is recorded separately from raw bit coordinates; `engineering.integrated.decode_physical` applies the recorded scale and offset to a capture. The command never opens final routes.

Each integrated run also writes a `*-signal-catalog-v1.json` file. It has a per-vehicle family index with route-specific period, payload length, bit-one, and bit-flip summaries, plus role profiles with ranked ID and field hypotheses, measured relationships, and expected physical constraints. A role can have an ID hypothesis with no bit-span hypothesis. Source-vehicle DBC profiles contribute weak timing/bit-activity search priors even when IDs differ; these never promote a name. Same-ID families and those priors are searched first under the bounded relation budget. Every catalog records capture hashes and the evidence track so source-only and DBC-assisted claims remain separate.

For each experiment, report extraction accuracy, relationship replication, data provenance, and fit/runtime cost. [Historical normal-profile notes](../docs/behavior-discovery.md) and [empirical notes](../docs/empirical-notes/) retain the measured pilots. Historical detector-oriented pilots remain in the repository record, but detection transfer and graph–Mamba integration are outside this package's current research scope.

## Experimental candidate evidence graph

Candidate graphs remain in the separate physical benchmark; `GraphWeights` stays at version 1. The artifact records source vehicle identities, capture and DBC hashes, feature definitions, validation, nonnegative weights, and a threshold. Benchmark fitting checks target identity and hashes before ranking. The target DBC is used only by post-fit evaluation. Role names and types use a small mapping pinned to [COVESA VSS v6.0](https://github.com/COVESA/vehicle_signal_specification/tree/v6.0/spec).

`relationships/physical/` contains unary role scoring, source calibration, distinct physical factor evaluators, and the bounded combination ranker. `relationships/behavior/` contains timing and event motifs as active controls. `relationships/shared/` contains field identity, evidence types, time joins, and causal alignment. `relationships/README.md` maps the modules and evidence gates.

To reproduce the pinned four-vehicle folds after preparing the Forester and source calibration data, run:

```bash
.venv/bin/python scripts/benchmark_can_candidate_graph.py \
  /tmp/graphids_source_calibration_repro/benchmarks.json \
  --forester-dir /tmp/graphids_comma_benchmark \
  --output-dir /tmp/graphids_candidate_graph_benchmark --max-ids 64
```

The runner writes one source-only artifact per held-out vehicle, plus decoding-only, physical, and one/eight-layout graph results. The graph remains experimental until aggregate exact-layout recall and named-role retrieval improve over decoding-only without an assessed vehicle losing more than two percentage points in exact-layout precision or cross-byte order accuracy.

## Experimental signal graph

`scripts/benchmark_can_signal_graph.py` writes schema version 2 per vehicle. It keeps up to eight message layouts and represents exact/affine copies, delayed descendants, and synchronous equivalents across IDs. It does not infer multiplexing selectors. Role groups can retain several observed roots. The field graph informs which representatives are tested for role-level predictive edges. Its `origin: unknown` marker is intentional: passive bus timing cannot identify the physical sensor or transmitting ECU.

The command fits layouts and roles on fitting routes and checks graph edges on fitting plus development-validation routes. It opens only source-vehicle DBCs during fitting. `--evaluate-dbc` opens the target DBC after fitting for layout and nine-role retrieval metrics. It never loads a target final capture. The historical manifest uses its held-out route as development validation, so those results are exploratory and cannot support a new sealed promotion claim.

```bash
.venv/bin/python scripts/benchmark_can_signal_graph.py \
  /tmp/graphids_candidate_graph_v11_repro/benchmark-manifest.json \
  --prospective --evaluate-dbc --output-dir /tmp/graphids_signal_graph_study
```

The prospective cohort preparation command freezes five device-distinct routes per vehicle, including a sealed final route, in `prospective-manifest.json` and records capture, compressed segment, and DBC hashes in `provenance.json`. It writes `prospective-failed-preflight.json` and stops when fewer than two new vehicles qualify. The final route is converted and hashed during preparation; its DBC fit and roles are assessed only by the benchmark.

```bash
.venv/bin/python scripts/prepare_can_prospective_cohort.py /tmp/graphids_can_prospective_cohort
.venv/bin/python scripts/benchmark_can_candidate_graph.py \
  /tmp/graphids_can_prospective_cohort/prospective-manifest.json \
  --prospective --output-dir /tmp/graphids_can_prospective_benchmark --max-ids 64
```

Manifest code is centralized in `can_discovery/manifests/`: `common.py` owns content hashes and immutable JSON writes; `label_completion.py` owns the five-route sealed-final schema and staged-manifest checks; `iterative.py` owns iterative-round splits; and `graph.py` loads graph benchmark cohorts. These schemas remain separate because their split and final-route rules differ.

For the `vehicle_speed` label-completion case, `scoring.id_priority` computes a DBC-free execution order before layout scoring. It compares non-protocol bit-change activity across IDs and across two-byte lanes within an ID, requiring the association in both fitting captures. The prepared artifact records the priority order and its evaluation-only DBC-reference rank. Every eligible family still receives exhaustive layout scoring; priority is not a semantic label or an exclusion gate.

Scoring terms now use `scoring.evidence.ScoreEvidence` and `scoring.composition`: the unary payload score, source contrast adjustment, and speed–RPM relationship adjustment remain individually recorded and sum to the prior ranking value. `scoring.timing` records family arrival cadence in prepared artifacts with zero ranking weight. `scoring.physical` adapts the existing joint physical evaluators and marks missing role candidates unavailable; its result receives no ranking weight until source-only calibration and independent validation establish one. The candidate pool and ranked-case schemas were advanced for the new evidence records. PicaCAN's graph matcher remains a separate experimental method rather than a term silently merged into this score.

For DBC-assisted `vehicle_speed` cases, ranking also records a diagnostic wheel-speed factor between the top speed hypothesis and each known wheel-speed DBC field. It uses fitting captures and has zero ranking weight; the unsupervised track never opens those target DBC wheel fields. This diagnostic must be calibrated on independent source vehicles and checked on development routes before it can become a ranking term.
