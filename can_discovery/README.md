# CAN discovery research package

`can_discovery` studies two questions independently of GraphIDS models:

1. **Reverse engineering:** Which message structures, bit fields, protocol fields, and physical-role hypotheses can be recovered from raw CAN recordings? Compare proposed layouts and values with independent DBC or sensor truth where available. Keep uncertain and unobserved fields unresolved.
2. **Relationships:** Which timing or value relationships repeat across independent normal recordings and, where observable, across vehicles? Compare each relationship with own-history prediction and shifted or schedule-preserving controls. Report context, lag, support, and failures separately.

Decoding is one possible input to relationship discovery, not a prerequisite for every relationship. Arrival and whole-payload change relations can be studied without physical labels. A replicated predictive relation is not automatically causal or a shared physical invariant.

The active normal-data runner is `python scripts/mine_can_behavior.py --output OUTPUT [--raw-root RAW]`. It fits two frozen normal references and audits disjoint normal recordings. It does not read attack files. The resulting profile and normal evidence tables are research artifacts, not detector features or alert scores. Separate scripts benchmark field extraction against held-out DBC-backed recordings and evaluate source-vehicle factors.

For each experiment, report extraction accuracy, relationship replication, data provenance, and fit/runtime cost. [Module details](../docs/behavior-discovery.md) and [empirical notes](../docs/empirical-notes/) retain the measured pilots. Historical detector-oriented pilots remain in the repository record, but detection transfer and graph–Mamba integration are outside this package's current research scope.

## Experimental candidate evidence graph

The default profile remains version 7 (version 8 with a supervised source boundary model). Set `BehaviorConfig.candidate_graph_artifact` to a fitted `GraphWeights` JSON file and pass `target_vehicle` to `fit_profile` to run the graph reranker and write version 9. The artifact records source vehicle identities, capture and DBC hashes, feature definitions, third-trip validation, nonnegative weights, and a threshold. Fitting checks target identity and hashes before ranking. The target DBC is used only by benchmark evaluation. Role names and types use a small mapping pinned to [COVESA VSS v6.0](https://github.com/COVESA/vehicle_signal_specification/tree/v6.0/spec).

`relationships/source_calibration.py` fits source DBC calibration, `unary_roles.py` scores candidate traces, and `combination_ranker.py` searches bounded role assignments. `candidate_graph.py` intersects alternative layout memberships, enforces protocol and bit conflicts, and caps evidence from the same relationship. `signal_time.py` supplies causal alignment. Catch22, STUMPY, and Tigramite analyses live in `experimental_analysis.py`; the standalone event pattern miner remains experimental. `observational_edges.py` contains the earlier physical edge checks.

To reproduce the pinned four-vehicle folds after preparing the Forester and source calibration data, run:

```bash
.venv/bin/python scripts/benchmark_can_candidate_graph.py \
  /tmp/graphids_source_calibration_repro/benchmarks.json \
  --forester-dir /tmp/graphids_comma_benchmark \
  --output-dir /tmp/graphids_candidate_graph_benchmark --max-ids 64
```

The runner writes one source-only artifact per held-out vehicle, plus version 7, physical, and one/eight-layout graph results. The graph remains experimental until aggregate exact-layout recall and named-role retrieval improve over version 7 without an assessed vehicle losing more than two percentage points in exact-layout precision or cross-byte order accuracy.
