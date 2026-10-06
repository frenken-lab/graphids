# CAN discovery research package

`can_discovery` studies two questions independently of GraphIDS models:

1. **Reverse engineering:** Which message structures, bit fields, protocol fields, and physical-role hypotheses can be recovered from raw CAN recordings? Compare proposed layouts and values with independent DBC or sensor truth where available. Keep uncertain and unobserved fields unresolved.
2. **Relationships:** Which timing or value relationships repeat across independent normal recordings and, where observable, across vehicles? Compare each relationship with own-history prediction and shifted or schedule-preserving controls. Report context, lag, support, and failures separately.

The [research aims](RESEARCH_AIMS.md) set out the publication questions, the iterative decoding–relationship pipeline, current evidence gates, and the later GraphIDS handoff.

## Package shape

| Package | Responsibility |
| --- | --- |
| `preprocessing/` | Capture loading, message classification, tokenization, and source boundary models. |
| `decoding/` | Field candidates, raw value decoding, and DBC evaluation. |
| `relationships/behavior/` | Timing and motif evidence learned from observations. |
| `relationships/physical/` | Physical roles, factors, calibration, and experimental graphs. |
| `relationships/shared/` | Field identity, alignment, lag joins, and common evidence records. |
| `behavior/` | Normal profile schema and fitting, payload rules, tracking, and the standalone runner. |
| `experiments/` | Isolated paper baselines, including PicaCAN. |

The root package exports the normal-profile API. Old `can_discovery.profile`, `payload`, `tracker`, and `runner` imports are compatibility modules; new code should use `can_discovery.behavior`. Version-11 profile serialization remains unchanged.

Decoding is one possible input to relationship discovery, not a prerequisite for every relationship. Arrival and whole-payload change relations can be studied without physical labels. A replicated predictive relation is not automatically causal or a shared physical invariant.

The active normal-data runner is `python scripts/mine_can_behavior.py --output OUTPUT [--raw-root RAW]`. It fits two frozen normal references and audits disjoint normal recordings. It does not read attack files. The resulting profile and normal evidence tables are research artifacts, not detector features or alert scores. Separate scripts benchmark field extraction against held-out DBC-backed recordings and evaluate source-vehicle factors.

The experimental integrated decoder combines CANMatch-style same-ID/DLC reference lookup, payload-profile checks, tokenization of unmatched families, field equivalence, and AutoCAN-style derivative/integral and wheel-mean evidence. Run `python -m scripts.benchmark_can_integrated MANIFEST --output-dir OUTPUT --track both --evaluate-dbc`. `source_only` reads other vehicles' DBCs but opens the target DBC only in evaluation; `dbc_assisted` explicitly uses the target DBC as fitting anchors. The assisted evaluation recomputes a pool with **all** target labels hidden, so anchor fields cannot inflate recovery. The version-1 artifacts retain uncertain same-ID matches as alternatives and all proposed labels as review entries. Physical scaling from DBCs is recorded separately from raw bit coordinates; `engineering.integrated.decode_physical` applies the recorded scale and offset to a capture. The command never opens final routes and does not change normal profiles.

Each integrated run also writes a `*-signal-catalog-v1.json` file. It has a per-vehicle family index with route-specific period, payload length, bit-one, and bit-flip summaries, plus role profiles with ranked ID and field hypotheses, measured relationships, and expected physical constraints. A role can have an ID hypothesis with no bit-span hypothesis. Source-vehicle DBC profiles contribute weak timing/bit-activity search priors even when IDs differ; these never promote a name. Same-ID families and those priors are searched first under the bounded relation budget. Every catalog records capture hashes and the evidence track so source-only and DBC-assisted claims remain separate.

For each experiment, report extraction accuracy, relationship replication, data provenance, and fit/runtime cost. [Module details](../docs/behavior-discovery.md) and [empirical notes](../docs/empirical-notes/) retain the measured pilots. Historical detector-oriented pilots remain in the repository record, but detection transfer and graph–Mamba integration are outside this package's current research scope.

## Experimental candidate evidence graph

Normal profiles write version 11. Older profiles require regeneration. Candidate graphs remain in the separate physical benchmark; `GraphWeights` stays at version 1. The artifact records source vehicle identities, capture and DBC hashes, feature definitions, validation, nonnegative weights, and a threshold. Benchmark fitting checks target identity and hashes before ranking. The target DBC is used only by post-fit evaluation. Role names and types use a small mapping pinned to [COVESA VSS v6.0](https://github.com/COVESA/vehicle_signal_specification/tree/v6.0/spec).

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

`scripts/benchmark_can_signal_graph.py` writes `signal_graph.json` schema version 1 per vehicle. It keeps up to eight message layouts, proposes conditional layouts for conservatively detected byte-wide modes, and represents exact/affine copies, delayed descendants, and synchronous equivalents across IDs. Role groups can retain several observed roots. The field graph informs which representatives are tested for role-level predictive edges. Its `origin: unknown` marker is intentional: passive bus timing cannot identify the physical sensor or transmitting ECU.

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
