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

## Chunk execution and CPU loading

`stream_execution: chunked` is the default for both single-bin online input and
whole-recording validation. Independent ID/generation segments run through
state-carrying Mamba-1 scans, followed by chunk-wide graph readout and heads.
The wrapper retains convolution history and adds the incoming SSM state's
analytic decay contribution to the official zero-initialized scan. Correction
work is tiled to a 64 MiB temporary budget. The pinned backend is unchanged.
`stream_execution: recurrent` uses batched one-bin Mamba updates for diagnosis;
it shares vectorized forecasting and scoring with the chunk path.

State retains only live boundary identities in bounded batched caches; expired
and reborn IDs cannot inherit history. Runtime `GraphMambaState` is version 2;
version-1 live states require an explicit reset. Existing model checkpoints
remain schema version 1 and load without weight migration. Recording/context
changes reset state, while gaps and replay in the same episode are rejected.

Training uses CPU-discovered segment indices with differentiable tensor gather
and scatter. Frozen context-to-local-node mappings use a bounded worker-local
LRU; model activations and optimizer-dependent values are never cached.

`num_workers` applies to training, validation and all test loaders. Workers
spawn safely, omit the attached Lightning trainer when serialized, reopen
read-only snapshot memmaps lazily, and use persistent workers with prefetching.
`pin_memory: true` enables custom-batch pinning and nonblocking CUDA transfers.
Library defaults remain zero workers and unpinned memory; GPU presets select
two workers and pinning. Short one-pass evaluations can be faster with zero
workers because every test loader starts its own pool; benchmark the actual
workload before increasing workers or stream batch size.

Score collection transfers numeric columns in bulk. Rank zero calibrates on
source-normal validation, broadcasts calibrated buffers, and computes metrics
from gathered compact columns. JSONL and coverage retain their existing fields.
The pilot records preparation, optimizer batches, validation, calibration,
metrics and artifact timings separately. Packing preflight evidence is reused
only when manifest, representation, packing configuration and implementation
fingerprints match; source hash verification still runs.

On one Ascend A100, the 2026-10-14 real-checkpoint gate scored all 385,808 target
bins in 53.26 s, compared with the original 3,721.82 s. Every score matched within
`rtol=2e-4, atol=2e-5`, with no alert disagreements. A complete fresh 32-step
pilot with two workers took 163.63 s, including full source validation and both
target modes. These are execution measurements, not detection-quality evidence.

Use `scripts/graph_mamba_benchmark.py` for a five-trial warm benchmark and optional
whole-target comparison against saved JSONL scores. Use
`scripts/graph_mamba_parallel_gate.py --mode native` under two-rank `torchrun`
for distributed calibration/coverage, or `--mode ray` for the public two-GPU
fit path. Store each run under a new external output directory.

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

### Corrected evaluation and frozen reference diagnostics (v2)

`score_columns.raw_average_precision` ranks FP64 raw scores with a stable sort
and groups ties before integrating precision over recall increments. It never
sigmoids forecast scores. Unknown labels and `target_calibration` are excluded
from AP; reports explicitly identify groups lacking both classes. Vehicle AP
pools normal and attack test recordings across registry dataset names. Dataset
and recording breakdowns remain available. The macro metric averages vehicles.

`CalibrationBundle` v2 binds calibration to a SHA256 of learned checkpoint
parameters (excluding calibration buffers), the source-only recording manifest,
reference policy, inference mode and calibration track. Each source mode fits
its own three component statistics and two thresholds at quantile 0.999, with a
1e-4 scale floor. Classification uses every scored verified-normal bin;
forecast calibration also requires exposure, prior forecasts and component
availability. Missing modes fail. Unavailable components are recorded and
excluded. Bundles serialize as primitive checkpoint metadata without changing
learned parameter shapes. Historical pooled buffers can only be used with
`legacy_calibration_replay=True`; fresh source validation upgrades an old
checkpoint without retraining.

The separate target-normal-calibrated conditioned track retains source component
statistics and fits only thresholds from verified-normal `target_calibration`.
Scores and AP therefore stay identical to the source-calibrated track. Strict
zero-shot remains source-only. Report alert counts are classification-only,
anomaly-only, overlapping and combined **bin alerts**, with normal exposure and
per-head normal bin alerts/hour; they do not measure incidents.

`ReferencePolicy` controls profile inputs, canonical/reference normalization,
cross-ID messages, anchor persistence, reference flags and classifier pooling.
Defaults preserve full conditioning. Recent classifier pooling excludes silent
reference anchors from the classifier while keeping all forecast pooling
unchanged. Metadata and raw contexts remain available for auditing hidden
inputs. Expiring references retain reference membership; snapshots are rebuilt,
missing neighbors are masked, and cache/state identities include the policy.
The model and data policies must agree.

Run `scripts/graph_mamba_conditioning_diagnostics.py --help` inside a Slurm
allocation. It first corrects retained target scores, then replays source
validation and freezes a bundle before each intervention's target evaluation.
Its versioned JSON report combines strict and target-calibrated baselines,
source and target intervention effects on matching eligible bins, coverage,
raw components and score distributions. The full-reference trial also supplies
the reference-free baseline. Lifecycle eligibility differences are reported
separately. Target effects can include interactions and do not select a winning
intervention.

Historical anomaly-AP curves are affected by automatic sigmoid conversion.
Recompute only checkpoints or raw scores that were retained; intermediate
epoch values cannot be reconstructed. The completed learning experiment retains
its original epoch-4 selection rule. The separate
`graph_mamba_ctat_learning_corrected.yml` next-run preset maximizes
`val/source_min_vehicle_macro_ap`: the minimum of the four source vehicle-macro
APs over both heads and modes. Equal values keep the earlier checkpoint.
Forester is now diagnostic evidence. Broader transfer claims require reviewed
additional vehicles and a new independent holdout.


The v2 audit found that the completed run's `last.ckpt` duplicates its epoch-4
best file. Ray's per-epoch snapshots retain the actual final weights and all
intermediate epochs, so their corrected source curves can be replayed without
inventing metrics. The corrected next-run preset uses a separate unmonitored
`final_checkpoint` callback to save current weights even when monitored AP does
not improve. Learning-curve metadata reads the configured checkpoint metric.
Distributed validation broadcasts global metrics to every rank for checkpoint
callbacks as well as broadcasting calibration bundles.

The full report and raw evidence are under
`/fs/ess/PAS1266/graphids/dev/rf15/experimentruns/graph-mamba-conditioning-v2/`.
Use `scripts/graph_mamba_recompute_curve.py` only with an inventory of retained,
SHA256-checked checkpoints. Its source curve preserves the original selection;
only the actual final snapshot receives an additional diagnostic target replay.


Evaluation also rejects score tables whose calibration bundles are missing or
assigned to the wrong inference mode. Anomaly bin alerts require at least one
component that is both present in the bin and available in source calibration;
a zero or negative threshold cannot turn a componentless bin into an alert.
These masks are applied during collection, replay and reporting. Disabled heads
are marked `disabled`, contribute no AP or alerts, and are omitted from score
distributions in both strict and target-calibrated reports.
