# CAN relationship module map

The version-11 normal-profile runner is retired. The candidate graph stays in the benchmark, where `GraphWeights` remains schema version 1.

| Area | Modules | Evidence role |
| --- | --- | --- |
| `physical/` | `role_types`, `unary_roles`, `combination_ranker` | Rank compatible physical roles with null choices and retained uncertainty. |
| `physical/` | `source_calibration`, `physics_factors` | Calibrate from source vehicles and retain factor-specific assumptions and held-out results. |
| `physical/` | `observational_edges`, `vehicle_models`, `candidate_graph` | Report conditional evidence separately; use it in ranking only when source validation gates pass. The graph is experimental. |
| `behavior/` | `dynamics`, `motifs` | Timing, change, and event motif evidence for independent benchmarks. |
| `shared/` | `signal_time`, `time_join`, `field_analysis`, `evidence` | Shared causal alignment, field identity, lag joins, and factor evidence. |

The candidate graph promotion gate in `scripts/benchmark_can_candidate_graph.py` requires aggregate gains in exact layout recall and named-role retrieval, with no assessed vehicle losing more than two percentage points of layout precision or cross-byte order accuracy. The version 9 report in `docs/empirical-notes/2026-10-04-candidate-graph-v9/` is a historical comparison; its artifact format is unchanged. Standalone pattern mining, STUMPY, catch22, and PCMCI paths have been retired. Historical notes that name their commands describe archived experiments.
