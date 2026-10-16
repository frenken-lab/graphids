# Module responsibilities

Experiment YAML in `configs/experiments/` defines a launchable run: source data,
model, representation, resources and `fit` or `test` stage. The typed boundary
in `graphids/exp/config.py` expands inheritance, checks representation agreement,
and rejects retired contracts. `graphids/primitives*.py` exports only `can_event`,
`can_event_dm` and `event_graph_mamba` configuration primitives.

`graphids/core/data/recordings.py` owns reviewed recording/provenance validation.
`graphids/core/data/events/` owns lossless caches, chronological independent-stream
scheduling, eligible target totals, and source ingestion verification.
`graphids/core/models/event_graph_mamba.py` owns continuous learned state,
truncated backpropagation, separately normalized objectives, epoch checkpoints,
frame-score shards, coverage checks and evaluation. Shared exact AP lives in
`graphids/core/metrics.py`.

`graphids/exp/runtime.py` constructs data, models and callbacks.
`graphids/exp/ray_backend.py` launches Ray workers and runs Lightning, journals
run status and queues offline metrics for serialized ingestion.
`graphids/exp/slurm.py` and `graphids/cli/exp.py` render and submit allocations.
`graphids/core/callbacks.py` provides checkpoint hashes and resource callbacks;
`graphids/_mlflow.py` owns tracking and system metrics.

The active flow is YAML → typed run → Ray/Slurm launch → Lightning `fit`/`test`
→ `.graphids/` journal and score artifacts → `gx exp ingest`.
Training uses one GPU. Snapshot reproduction lives in the verified
`archives/snapshot-2026-10-16.tar.gz` working-tree archive.
