# Message-preserving graph–Mamba input

The active model path uses `can_event_dm` with `event_graph_mamba`. Every observed
CAN frame enters the temporal encoder in physical recording order. This follows
the requirement to preserve raw observations and let the model learn its own
representations. The snapshot implementation is retired to the verified [source archive](snapshot-archive.md).

The active preset is `configs/experiments/graph_mamba_events_source.yml`.
It selects the per-ID timing-gap model from the October 30 matched ablations:
source-validation classification AP 0.9257 at epoch 25, compared with 0.0478
for the one-stream baseline. This is one seed and a source-validation comparison;
weak attack-family coverage and independent operating-point evaluation remain open.
Snapshot weights and calibration use a separate architecture and cannot load here.

## Selected per-ID timing gap

The model and `can_event` representation default to schema 2.
Each frame retains raw elapsed time and bus interarrival time, plus three header
features: normalized preceding same-ID gap, normalizer support, and first-seen
indicator. The encoder has 51 input fields and the forecast header has 35.
Representation 2 prepares same-ID predecessor indices across chunks without
changing the lossless on-disk caches. The index costs eight bytes per event;
sorting needs additional temporary CPU memory during setup. Preceding payload
copies and graph-entry timestamps are no longer carried.

Normalizers fit only verified-normal `train_query` captures, using at most 65,536
deterministic evenly spaced pairs per capture. They apply `log1p(gap / scale)`
with median positive-gap scale, robust median/IQR centering, and a frozen 99.9th
percentile clip at least as large as the scale. Near-zero IQR uses unit spread.
Supported numeric IDs have their own tables; other IDs use the pooled same-ID-gap
table and a false support indicator. `timing_min_support` defaults to 32 sampled
pairs. Fitted tables are checkpoint buffers, validated on load and never refitted
from evaluation traffic. The legacy unused bus-normalizer row remains in the
buffer to preserve selected-checkpoint identity.

The active preset uses one stream, unweighted BCE, a joint payload objective,
25 epochs, 16,384-event chunks and a 65,536-event optimizer window.
`python -m scripts.event_quality_suite prepare OUTPUT --seed 42` freezes one
candidate. `submit OUTPUT` creates three jobs: a CUDA gate, the fit after a
successful gate, and a report after the fit terminates. `status OUTPUT` refreshes
accounting and `candidate-results.json`. Separate directories and `--seed` select
repeat runs. Preparation alone does not submit anything.

Each epoch summary includes `recording_macro_ap` over mixed-class recordings and
`recordings_by_ap` ordered weakest first. Undefined single-class AP is excluded
from that mean; all recording metrics remain available. The CLI shows the three
weakest recordings at the aggregate-AP-selected epoch. Checkpoint selection still
uses aggregate AP.

The weakest-recording lines compare classification and payload AP at that same
epoch, including supported payload rows versus known classification rows. These
APs may cover different populations. Classification-only runs show `untrained`
instead of presenting their payload scores as a detector result.

For the separate classifier diagnostic, run `python -m scripts.event_classifier_pilot
--config WINNER.yml --checkpoint WINNER.ckpt --output NEW_DIRECTORY` inside one
A100 allocation. It freezes the checkpoint and timing buffers, disables dropout,
and trains two identically initialized linear heads for five epochs: ordinary
BCE and training-recording-balanced attack positives capped at 10. Negatives and
total positive weight retain their original exposure. Both arms share each
causal forward pass and the original loader/windows. The final heads and original
classifier are evaluated on the same complete validation cohort. `pilot.json`
contains metrics, weights and coverage; `classifier-heads.pt` contains only the
adapted heads and base checkpoint identity. This is a diagnostic adaptation stage,
not full-model resume or a change to the selected preset. The repository's
`MODEL_IMPROVEMENT_PLAN.md` records the pilot protocol and measured results.

`python -m scripts.event_failure_report --pilot PILOT_ROOT --focus FOCUS.json
--output NEW_DIRECTORY` diagnoses saved pilot predictions in a CPU allocation.
The focus file maps explicit validation recording keys to lists of matched
training keys, including empty lists for unmatched captures. The report verifies
all source-validation rows, fits vehicle-scoped byte/transition support only on
verified-normal training queries, reconstructs checkpoint-owned timing support,
and compares heads at empirical normal-validation frame budgets. Unsupported
payload rows remain in recall denominators. `report.json` and `report.md` record
exposure, support, score distributions, disagreement, matched-budget decisions
and physical-row examples. These are source-selection diagnostics; they do not
refit the encoder, open target files or provide calibrated alerts/hour.

The no-elapsed, normalized-bus, payload delta/XOR and graph-age paths and the
weighting/interleaving/classification-only generators have been retired. Their
completed artifacts and frozen source remain in the October 30 suite. The
selected schema-2 checkpoint and schema-1 baseline checkpoints remain loadable;
neutral historical feature flags are accepted and removed during configuration
validation. Active flags for removed variants fail explicitly and require the
original frozen source. Schema 1 requires `use_id_gap: false`.

## Input contract

One event contains the numeric CAN arbitration ID, observed payload bytes,
payload length, and timestamp. The cache preserves all 29 ID bits, exact byte
values, FP64 physical timestamps, equal-timestamp row ordering, and every physical
CSV row, including duplicates and messages subsequently overwritten within the
same 10 ms interval. Source rows already subject to reviewed clock repair retain
their reviewed ordering; this path does not independently repair or sort clocks.
It does not reconstruct unavailable wire fields such as an absent IDE/RTR flag.

The neural input directly encodes the available frame fields:
eight byte values scaled by 255, eight byte-presence masks, 29 ID bits, length
scaled by eight, elapsed recording time, the preceding bus interarrival time,
and the three per-ID timing features described above.
Elapsed time and interarrival are computed in FP64 and converted to the model's
usual floating point dtype at projection. Absolute clock origin is retained in
the cache but is not a learned recording-identity feature. There are no binned
counts, payload-change summaries, byte-difference heuristics, or
correlation-selected graph edges. Normal timing statistics supply only the
explicit per-ID gap normalization. Labels, normality flags,
recording names, roles, vehicles, and source interfaces do not enter the encoder.
Normality and payload provenance only control eligible training targets.

Chunks are execution units, not summaries. Every constituent event is encoded
and scored individually. During training, each frame is a scored target exactly
once per epoch. Both Mamba caches, latest per-ID memory, and previous-bus state
carry across chronological chunks and detach at each chunk boundary for truncated
backpropagation. History resets at recording boundaries. Training and inference
use the same functional scan; inference wraps it in `no_grad`. Unknown labels
remain unknown and are excluded from supervised loss.

`stream_batch_size` selects 1 to 16 independent recordings on one GPU, limited
by the distinct reviewed training recordings available. The
scheduler batches equal-length chunks and groups short tails by their exact
length, replacing completed recordings with fresh streams. It never inserts
fake events, replays warmup rows or drops tails. `chunk_events * stream_batch_size`
must fit both `max_microbatch_events` (default 65536) and
`effective_batch_events` (default 65536). Raise the explicit microbatch cap
when testing more streams at a fixed chunk length; it can be configured up to
262144 events for at most 16 streams. Each optimizer
window declares its complete eligible classification and normal-payload target
counts on the CPU, and each objective uses its own denominator across all chunks.
Groups can split between independent streams to fill optimizer windows, without
splitting their chronological chunks. The final partial window uses its actual
eligible counts. Use
`accumulate_grad_batches=1`; accumulation belongs to the data schedule.
The retired `warmup_events` argument has been removed.

CPU preparation supplies routing indices plus recording, source, row and incoming
ID-history provenance. Each cache retains only compact first-appearance metadata;
dense routing exists only for scheduled chunks. Host timestamp metadata avoids
GPU scalar reads during continuity checks. An independent serial graph/routing/loss oracle lives in test support; the
unused `scan(reference=...)` argument has been removed. Model
execution batches projections and heads across recording banks, then attends
within each stream using local history indices and bounded matrix-product tiles.
Unknown/future ID slots receive zero attention mass, including frames with
entirely empty history.

`num_workers` (default 2) controls CPU preparation; pinned memory defaults to true. Explicitly select zero workers for synchronous CPU diagnostics. `prefetch_factor` (default 2) and
`persistent_workers` (default true) apply only with workers enabled. Workers
never perform CUDA operations. `cuda_prefetch` (default false) opts training into
a main-process loader that stages one future optimizer window on a dedicated
CUDA copy stream; use `pin_memory: true` for asynchronous host transfers. The
loader remains a `DataLoader` for Lightning. CPU execution and the synchronous
path remain available, and loader seeding uses a private RNG. Prefetched windows
count toward the same GPU memory ceiling as model activations.

The official scan computes the transformed time step once and shares it between
cumulative decay and outgoing-state reconstruction, retaining both gradient paths.
At epoch start, the model captures expected source-training recording counts;
each optimizer step compares physical coverage against those counts.

The model clips gradients to norm 1 before each AdamW update. Checkpoints may be
saved only after complete epoch coverage, including source/config identities,
optimizer state (Lightning), and Python/NumPy/PyTorch/CUDA RNG states. Resume via
`config.ckpt_path` starts at the next epoch; missing epoch identity and mid-epoch
resume are rejected. Dropout and all existing model dimensions and objectives
retain their defaults. Production uses the pinned official CUDA operators in
FP32, without compilation or automatic precision changes.


## Learned temporal and graph context

A shared Mamba stack advances on the chronological **bus message sequence**. Its
output at each event therefore incorporates earlier messages from all observed
IDs. A graph readout uses learned query/key/value attention to the latest
*preceding* Mamba state for each encountered CAN ID. The current event's encoded
state queries those historical node states; its updated temporal state becomes
that ID's memory for subsequent events. Future IDs and future observations are
masked out. Attention learns relationships instead of receiving a graph chosen
from count/change correlations. This differs from the snapshot model's per-ID
clock-tick Mamba streams and its fixed reference-neighbor readout.

These hidden states are learned compression **after** individual messages enter
the network. Raw inputs are not replaced by a hand-selected per-bin payload.
Node and graph-work capacities fail explicitly when exceeded; they never evict
IDs, truncate messages, or merge events. This first version supports one training
or evaluation device. It rejects multi-device configurations rather than silently
padding evaluation or splitting a stream's chronological state across ranks.

## Prediction and learning targets

The supervised head predicts attacker-frame labels with BCE. A separate
normal-only head predicts eight categorical byte distributions from preceding
bus/per-ID states and the arriving frame's header/timing, before that frame's
payload can enter the predictor. Valid observed bytes from independently
verified-normal training recordings supply this objective. It is a simple
factorized byte likelihood, not a joint payload distribution or an operational
fault-state model. Source attack recordings do not train the normal-only loss.

Payload NLL is reported as an uncalibrated anomaly ranking. It does not reuse
snapshot calibration. Both scores remain distinct: an abnormal payload score is
not an assertion of attacker origin. No operational fault labels have been added.
An observed-frame interface has no prediction unit during complete silence;
absence/suppression detection will require a separately evaluated clock or
next-arrival survival objective. No such objective is implied by frame scores.

The preset retains the reviewed Impala/Silverado source split, filters out target
roles before opening data files, and uses neither target evaluation nor reference
summary features. Reference recordings remain reserved and are not used as query
training examples. Explicit source fitting is:

```bash
gx exp config configs/experiments/graph_mamba_events_source.yml
gx exp submit configs/experiments/graph_mamba_events_source.yml -C ascend
```

Do not interpret this preset as a controlled architecture ablation: event units,
temporal routing, graph construction, supervision granularity, normality targets,
and optimizer-step exposure differ from the snapshot experiment. A subsequent
learning experiment must report frame attribution and common bus-bin comparison
separately, fit calibration on source data only, and retain existing model results
as controls.

## Verification and artifacts

Run tests in Slurm, including `tests/test_event_graph_mamba.py`. They cover exact
payload and row preservation, timestamp ties, empty payloads, unknown labels,
source-only loading, continuous state and exact scored coverage, future-perturbation
causality, current-payload exclusion from forecasting, arbitrary chunk boundaries,
state identity, capacity failures, checkpoint round trips, a Lightning learning
smoke test, and pinned official CUDA versus reference outputs and gradients.

For a complete source-ingestion audit, run in a CPU allocation:

```bash
gx can verify-events --config configs/experiments/graph_mamba_events_source.yml \
  --output "$GRAPHIDS_RUN_ROOT/experimentruns/graph-mamba-events-source-v1/verification/source-input-audit.json"
```

This rereads every selected source query row and checks every cache field against
it, persisting per-record progress and hashes. A failed audit raises and marks the report incomplete. Model evaluation appends bounded chunks to one `scores.bin`
per epoch, plus `index.json` recording physical row offsets, source identities,
exact coverage and frame-level metrics. Incomplete validation is rejected. Thresholds and
snapshot calibration are not silently transferred to the new model.

Evaluation computes categorical per-byte NLL once for publishing ranks and reuses
it in loss logging and anomaly scores. Loss eligibility remains normal-only,
verified, and scored; anomaly scores use the existing payload-support mask.
Sanity checks and non-publishing ranks retain the loss-only path.

`graphids.evaluation.EventScores` owns packed score writing, coverage checks and
exact epoch AP independently of the model and Lightning. The model supplies
the six score columns and logs the returned metrics. Artifact schema 2 stores
little-endian FP64 values in row-major order, with shape/dtype and chunk offsets
in the index; the six score columns and checkpoint identities are unchanged.
`iter_score_chunks` reads complete schema-1 NPZ epochs and schema-2 packed epochs.
Metric scratch space uses local `TMPDIR` rather than the persistent lake.
`scripts/pack_event_scores.py EPOCH --remove-shards` migrates a complete legacy
epoch: it verifies source metadata, exact coverage and the packed-file SHA256
before replacing the index and removing the old shards. Existing metrics are
preserved exactly. Direct artifact
and metric tests live in `tests/test_evaluation.py` and
`tests/test_metrics.py`; end-to-end coverage and checkpoint tests remain
in `tests/test_event_graph_mamba.py`.
The CI `cpu-tests` job includes these direct tests; training and CUDA checks
run through the Slurm gate.

Implementation verification on 2026-10-16 checked all **21,285,666 query frames
across 21 source recordings**, with every cached field matching its CSV row.
CPU regression job `56288831` passed 49 tests (two GPU-only cases skipped).
A100 job `7891355` passed 15 tests, including official/reference kernel parity;
final event-specific checks passed across jobs `7891363` and `7891364` (the latter
corrected a test that checked device placement after Lightning teardown instead
of during training). The complete ingestion audit is CPU job `56288850`.
Machine-readable verification, logs, and an implementation source archive are at
`/fs/ess/PAS1266/graphids/dev/rf15/experimentruns/graph-mamba-events-source-v1/verification/`.
These are correctness and small learning smoke checks, not a new detection result.


## Bounded implementation gate

Run `python scripts/event_model_gate.py --mode cpu --output <directory>` in a CPU
Slurm allocation. `sbatch scripts/slurm/event_model_gate.sh` requests one A100 for
at most two hours and runs official/reference correctness, source-only benchmarks,
and a tiny complete synthetic fit through `gx exp launch` and Ray/Lightning.
It does not run a complete source learning epoch or target evaluation.

The gate resolves defaults from the selected experiment YAML and measures only
requested configurations. The default ten-recording cohort includes both reviewed
vehicles, normal traffic and attacks, with 65,536 scored frames per recording
(655,360 total) and continuous preceding history. Three independently reset
repetitions use identical weights, optimizer, recording histories and RNG.
At the production 16,384-event optimizer window this means 40 updates.

```bash
python scripts/event_model_gate.py --mode gpu --output <external-directory>
# Explicit larger-chunk comparison at a common optimizer window:
python scripts/event_model_gate.py --mode benchmark --chunks 16384 32768 65536 \
  --streams 1 --max-microbatch-events 65536 --effective-events 65536 \
  --events-per-recording 65536 --output <external-directory>
```

Supported overrides include `--workers`, `--cuda-prefetch` / `--no-cuda-prefetch`,
`--chunks`, `--streams`, `--effective-events`, `--max-microbatch-events`,
`--events-per-recording`, `--repeats` and `--no-profiles`. Sweeps must explicitly
raise their microbatch and optimizer-window limits where needed; invalid requests
are errors. OOM configurations are recorded without shrinking them. Correctness
failures abort the gate. Historical `pipeline`, `--pipeline`, control and `--steps`
interfaces have been removed; their original source is preserved externally.

Throughput timing synchronizes at measurement boundaries. Separate profiles
record CUDA-event phases, CPU/CUDA operators, transfers and launch gaps for each
successful requested configuration. The 32 GiB reserved-memory ceiling includes
prefetched tensors. Graph attention uses 256 MiB temporary tiles and Mamba
incoming-state correction retains its 64 MiB bound and official CUDA operators.
The GPU mode also runs a bounded public Ray/Lightning fit using production
settings and synthetic source data.

The measured production defaults are 16,384-event chunks, one stream, a
16,384-event optimizer window, two persistent workers, prefetch factor two,
pinned memory, CUDA prefetch off, graph capacity 2,097,152 and FP32. The earlier
full-pipeline measurement was approximately 344k events/s versus 243k for the
serial baseline at equal optimizer exposure, with 3.87 GiB reserved memory.
A larger-chunk comparison at a common 65,536-event optimizer window found 32K
chunks 4.9% slower and 64K chunks 16.7% slower than 16K.

Detailed historical results remain in `docs/empirical-notes/`, including
`2026-10-16-event-batched-pipeline.md` and `2026-10-16-event-larger-chunks.md`.
`docs/empirical-notes/2026-10-16-event-runtime-consolidation.md` records
corrected independent verification: the earlier
`scan(reference=True)` switch did not select a separate graph implementation.
Historical timings remain measurements of their recorded configurations; those
particular parity comparisons were not independent.
