# Graph Mamba training and evaluation

GraphIDS replaces event-model training with a shared CAN snapshot detector. Two
causal Mamba blocks encode each ID's history; bounded graph messages mix ID
states at the same physical cutoff. Classification scores the closed bin and
normal forecasting predicts the next bin. Their scores remain separate.

## Environment and resources

Develop on one Ascend A100. Cardinal H100 is an alternative. Install the pinned
official Mamba `v2.2.6.post3` revision and causal-conv1d 1.6.1 in a compute
allocation. This revision is compatible with the project's PyTorch 2.8/Triton
dependency constraints; the previously inspected newer revision is not.

```bash
module purge
module load gcc/12.3.0 cuda/12.8.1
export CC=gcc CXX=g++ MAX_JOBS=4
uv sync --extra dev
uv sync --extra dev --extra mamba --no-build-isolation
```

The production backend requires the actual CUDA selective-scan and causal
convolution extensions. Imports or an installed core wheel alone do not pass
the gate. `backend: reference` selects an explicit diagnostic Mamba recurrence;
it does not provide a production performance result. GPU tests, fits and
package compilation run in Slurm allocations. Cluster module availability must
be checked on the selected cluster.

## Recordings and provenance

Run `gx can audit --output /path/to/compatibility.json` to inventory all six
registered CAN datasets. The report detects exact duplicates and clock/payload
errors. It does not establish session independence, infer normality from zero
labels, or automatically authorize a training split. Review acquisition and
session provenance, including partial capture reuse, before constructing a
manifest with `version: 1` and `provenance_reviewed: true`.

Each recording needs a unique key, dataset, globally meaningful vehicle key,
session key, path, SHA256, role, timestamp unit, and explicit normal/label/payload
validity declarations. Paths are relative to the manifest location or absolute.
`payload_length_verified` means upstream serialization has been checked;
otherwise raw payload inputs remain available but payload targets are masked.

Roles are `train_reference`, `train_query`, `val_reference`, `val_query`,
`target_reference`, `target_calibration`, `target_normal_test`, and
`target_attack_test`. A session cannot cross roles and a target vehicle cannot
enter source training. Reference and normal target roles require independently
verified-normal captures. Validation can reuse declared training reference
captures when no separate validation reference exists; queries never build
their own context. Optional `attack_intervals` contain independently verified
start/end times in seconds, allowing silence to have a label. Filenames supply
no supervision.

The `datasets` option selects source training families. Reviewed target
recordings may belong to entirely different datasets. Every requested source
family must supply training queries; missing or unresolved datasets fail
explicitly. Target-conditioned evaluation requires a disjoint target reference.

## Bins, context and batching

Initial defaults are 10 ms bins, 128 scored bins per training window, up to 128
preceding warm-up bins, and one recording episode per stream batch. Empty bins
are real observations. A terminal frame at an exact cutoff retains its
classification score; its zero-exposure forecast target is ineligible. Clock
units, source hashes, feature version, bin settings and anchor membership enter
the cache identity. Existing event caches are never converted in place.

Reference context freezes per-ID normal profiles, numeric normalization and a
directed graph with at most eight incoming neighbors. Degree, strength and lag
describe observed dependence, not physical causality. Zero-shot has no target
anchors or profiles and uses a self-only graph. Source training withholds
reference context in 25% of windows by default.

Reference IDs persist through silence. Transient IDs expire after one second
without observations; reappearance has a new cache generation. Source-only
workload counts set power-of-two capacity defaults. Overflow or excessive
node-bin/message work fails with counts; observations are never silently
truncated. Default per-batch limits are 131,072 node-bin tokens and 1,048,576
graph messages, subject to device preflight. Recording caches also have an explicit 4 GiB
array budget (`max_snapshot_bytes`); larger caches fail before dense allocation
and require a source memory preflight or a source-validated bin width.

**Stream batching** runs independent live ID segments through full Mamba scans
on one GPU. Padding never advances carried state. **Data parallel training**
uses separate recording episodes on each GPU with one shared parameter set.
`trainer.devices` selects the worker count; `resources.gpus_per_worker` is one
for GPU workers. Workers share a nonduplicating schedule and globally reduced
loss denominators. Unscored dummy tails equalize optimizer steps. Evaluation
assigns a complete recording/mode sequence to one rank and carries both caches
across its chunks. Epoch sampling counts are equal across datasets; available
unique windows limit the balanced schedule and omitted windows rotate across
epochs. Report this sampled work rather than claiming every source bin is
scored in each training epoch.

## Fits, scores and gates

Start with the YAML presets `graph_mamba_joint`, `graph_mamba_joint_test`,
`graph_mamba_joint_cardinal`, and `graph_mamba_joint_parallel`. Set the reviewed
manifest path before launching. The runner preserves `gx exp config`, `launch`,
`submit`, journals, checkpoint sidecars, and serial MLflow ingestion.

Classification trains on known bin labels. Forecasting trains only when the
window's observed history and next-bin target are verified normal. The initial
heads use Poisson counts/new-ID arrivals and Bernoulli payload change. Every
loss is normalized by its eligible units. Warm-up initializes state without
contributing scored classification loss. All available classifier and forecast
terms start with equal weights; source validation chooses later changes.

`CausalCANBinner` keeps bounded raw-event feature state and emits closed bins
without storing recording history. Continuous inference steps live IDs, performs a same-time graph readout, scores
the prior forecast, and emits the next forecast. It rejects gaps or replay in
carried bin indices. Checkpoints contain a model-schema marker, source manifest
digest, score normalization, and source-normal alert thresholds. Target normal
calibration is a separately declared query partition; the primary evaluator
does not refit thresholds on it. Initial alerts are single-bin candidates,
without persistence or cooldown. Candidate alert counts and normal exposure
must be interpreted together; they do not certify a deployment false-alarm rate.

Evaluation writes `artifacts/bin-scores.jsonl` and a full expected-versus-scored
coverage report. Native metrics are per-dataset/per-mode classification and
anomaly average precision and candidate false alerts per normal hour. Bin
scores retain recording identity and eligibility for downstream interval-level
recall/delay analysis; absent trusted intervals are not fabricated.

Run these in allocations:

```bash
python -m pytest tests/core/data/test_snapshots.py tests/core/data/test_snapshot_datamodule.py tests/core/models/test_graph_mamba.py
python -m pytest tests/core/models/test_graph_mamba_cuda.py
python scripts/graph_mamba_gate.py --output /path/to/gate
python -m torch.distributed.run --standalone --nproc-per-node=2 scripts/graph_mamba_ddp_gate.py --output /path/to/ddp
python scripts/graph_mamba_memory_gate.py --output /path/to/memory
```

The functional fixtures are synthetic recordings with reviewed fixture
provenance. They test the implementation and do not establish real-dataset
transfer quality. Real pooled experiments remain blocked until their manifest
review is complete. Compare quality and full-epoch throughput against the
archived baseline on identical hardware after functional cutover.

## Legacy reproducibility

The old model/data stack is preserved at Git revision
`daa218eb262fb1c9f9dde067f68d4cc5309d3213`, tagged
`graphids-temporal-legacy`. Use an isolated checkout of that revision and its
locked environment to evaluate old checkpoints. Existing lake caches, run
artifacts and checkpoint files remain untouched. Legacy model weights and
event schemas are incompatible with graph-Mamba and must not be silently loaded
or migrated. GraphIDS 2.0 removes the five old model families and lane-specific
APIs once the functional gates pass.
