# Module responsibilities

Experiment YAML in `configs/experiments/` defines a launchable run: source data,
model, representation, resources and `fit` or `test` stage. The typed boundary
in `graphids/exp/config.py` expands inheritance, checks representation agreement,
and rejects retired contracts. `graphids/data/config.py` and
`graphids/models/config.py` define `CANEventDataConfig` and
`EventGraphMambaConfig`; `graphids/data/representation.py` defines the
`CANEventRepresentationConfig` contract. The runtime resolves the YAML types
`can_event_dm` and `event_graph_mamba` directly to their config classes.

`graphids/data/recordings.py` owns reviewed recording/provenance validation.
`graphids/data/events.py` owns lossless caches, chronological independent-stream
scheduling, and eligible target totals. `graphids/data/verify.py` owns source
ingestion verification.
`graphids/models/event_graph_mamba.py` owns continuous learned state,
truncated backpropagation, separately normalized objectives, epoch checkpoints,
and Lightning evaluation hooks. `graphids/evaluation.py` owns packed frame
scores, recording coverage checks and the evaluation index; it uses only NumPy
and the standard library. Shared exact AP and AUC live in `graphids/metrics.py`, using one
tied-score ranking pass.

`graphids/exp/runtime.py` constructs data, models and callbacks.
`graphids/exp/ray_backend.py` launches Ray workers and runs Lightning, journals
run status and queues offline metrics for serialized ingestion.
`graphids/exp/slurm.py` and `graphids/cli/exp.py` render and submit allocations.
`graphids/callbacks.py` provides checkpoint hashes and resource callbacks;
`graphids/_mlflow.py` owns tracking and system metrics.

The active flow is YAML → typed run → Ray/Slurm launch → Lightning `fit`/`test`
→ `.graphids/` journal and score artifacts → `gx exp ingest`.
Training uses one GPU. Snapshot reproduction lives in the verified
`archives/snapshot-2026-10-16.tar.gz` working-tree archive.
