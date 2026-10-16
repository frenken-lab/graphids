# Message-preserving graph–Mamba input

The active model path uses `can_event_dm` with `event_graph_mamba`. Every observed
CAN frame enters the temporal encoder in physical recording order. This follows
the requirement to preserve raw observations and let the model learn its own
representations. The snapshot implementation is retired to the verified [source archive](snapshot-archive.md).

The new experiment is
`configs/experiments/graph_mamba_events_source.yml`. It is a separate architecture
and checkpoint identity; snapshot weights and calibration cannot be loaded into
it. Adding this path does not establish improved attack detection, and no full
encoder fit has been run as part of implementing it.

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
scaled by eight, elapsed recording time, and the preceding bus interarrival time.
Elapsed time and interarrival are computed in FP64 and converted to the model's
usual floating point dtype at projection. Absolute clock origin is retained in
the cache but is not a learned recording-identity feature. There are no binned
counts, payload-change summaries, byte-difference heuristics, normal-reference
traffic profiles, or correlation-selected graph edges. Labels, normality flags,
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
must fit both `max_microbatch_events` (default 16384) and
`effective_batch_events` (default 16384). Raise the explicit microbatch cap
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
execution packs disjoint recording banks, batches projections and heads, and
uses tiled matrix products for attention. Unknown/future ID slots receive zero
attention mass, including frames with entirely empty history.

`num_workers` (default 2) controls CPU preparation; pinned memory defaults to true. Explicitly select zero workers for synchronous CPU diagnostics. `prefetch_factor` (default 2) and
`persistent_workers` (default true) apply only with workers enabled. Workers
never perform CUDA operations. `cuda_prefetch` (default false) opts training into
a main-process loader that stages one future optimizer window on a dedicated
CUDA copy stream; use `pin_memory: true` for asynchronous host transfers. The
loader remains a `DataLoader` for Lightning. CPU execution and the synchronous
path remain available, and loader seeding uses a private RNG. Prefetched windows
count toward the same GPU memory ceiling as model activations.

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
it, persisting per-record progress and hashes. A failed audit raises and marks the report incomplete. Model evaluation writes one bounded NPZ shard per chunk,
with physical row offsets and source identities, plus an index recording exact
coverage and frame-level AP. Incomplete validation is rejected. Thresholds and
snapshot calibration are not silently transferred to the new model.

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
