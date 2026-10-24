# Complete-recording profiling and local graph readout — 2026-10-24

This pass profiles the complete fit/evaluation path before simplifying graph
readout. The comparison starts at commit `0642583`; the user's existing deletion
of `CLAUDE.md` is outside this change.

## Workload and measurement

The cohort contains the smallest complete recording in each source
role/vehicle/normality class: four training and four validation recordings from
the reviewed Impala/Silverado manifest. Each fit processes 3,516,700 training
frames in 217 optimizer steps, then publishes all 4,205,712 validation frames.
No recording is truncated, repeated, or assigned a different role. Target files
are never opened. The cohort manifest retains the original recording hashes,
reviewed provenance, and physical row order.

Worker-fit runs use one A100-PCIE-40GB in SLURM allocation `8060133`;
public-launcher runs use allocation `8060493`. Both use four allocated CPU cores,
the project environment, and the production model/data settings: width 128, two
Mamba layers, dropout 0.1, FP32, 16,384-event chunks and optimizer windows, one
stream, two persistent workers, pinned memory, and CUDA prefetch disabled.
Each independently seeded run completes one epoch and writes scores, exact
metrics, checkpoints and SHA256 sidecars. Baseline and candidate source snapshots
are separate; the final three worker-fit repetitions alternate implementations
on the same GPU. Python/CUDA tracing runs are excluded from throughput samples.

The worker-fit interval includes data setup, model and Trainer construction,
sanity checks, training, validation, score finalization, callbacks and checkpoint
writing. It excludes initial Python/Torch imports and Ray startup. The public
launcher comparison additionally exercises Ray initialization, its worker loop,
reporting, offline journal/ingest publication and checkpoint transfer. The public
launcher reserves two Ray CPU resources per worker within the four-core allocation;
the model and two dataloader workers use the same settings.

## Findings

The final three alternating worker-fit repetitions have these medians:

| Measurement | Baseline | Local readout |
|---|---:|---:|
| Complete worker fit call | 19.547 s | 19.001 s |
| Training epoch including validation/finalization | 16.205 s | 16.015 s |
| Training phase | 9.248 s | 9.207 s |
| Validation batches, excluding finalization | 4.598 s | 4.458 s |
| Peak reserved CUDA memory | 3.719 GiB | 3.686 GiB |

The worker-fit call is 2.8% shorter; the epoch interval is 1.2% shorter.
These small differences, alongside startup variation, do not establish a
production training speedup. The change removes bookkeeping and slightly reduces
reserved memory.

All six alternating public Ray fits finish with exact cohort coverage. Their
three-repetition medians are:

| Public-launcher measurement | Baseline with explicit checkpoint enabling | Final candidate |
|---|---:|---:|
| Process interval, including imports/shutdown | 52.896 s | 51.482 s |
| Public `launch_run` call | 49.026 s | 47.588 s |
| Training epoch including validation/finalization | 16.708 s | 16.740 s |
| Training phase | 9.464 s | 9.383 s |
| Validation batches, excluding finalization | 4.935 s | 4.805 s |

The public epoch interval is effectively unchanged (+0.2%). Baseline launcher
samples fall from 52.625 to 49.026 to 47.646 seconds, while candidate samples are
47.581, 47.691 and 47.588 seconds. The lower candidate launch median is therefore
confounded by startup/cache drift; it is not evidence of faster production
training. All runs publish the complete source-cohort scores and finish Ray
reporting, offline publication and checkpoint transfer.

Lightning timing separates validation batches from the module's epoch-end metric
finalization. The phase telemetry callback ends its validation interval before
that module hook, so its throughput alone is not complete evaluation throughput.
The epoch timing includes both. Startup/import/cache variation also makes the
first three sequential fit-call timings unsuitable for a speedup claim.

The Python profile and short CUDA traces identify temporal state correction,
graph gathers/readout, worker tensor transfer and score finalization as relevant
work. Callback dispatch and checkpoint writing are comparatively small on this
cohort. The initial cProfile diagnostic included an additional phase callback;
all untraced throughput samples use only the production runtime callback.

## Runtime integration failure

The first unmodified public launch failed before training: the Ray runtime
unconditionally defaulted `enable_checkpointing` to false while the production
preset supplied `Sha256ModelCheckpoint`. Lightning correctly rejected the
conflicting settings. The original failure and offline journal are preserved in
`baseline-public-1/` and `baseline-public-1.log`.

The runtime now enables Lightning checkpointing by default when a configured
callback is a `ModelCheckpoint`. Runs without one still use Ray's checkpoint
reporter, and explicit Trainer settings take precedence. The existing runtime
integration test now exercises both callback configurations without increasing
the collected case count. Baseline public timing runs explicitly enable
checkpointing to work around the old runtime failure; the final candidate uses
the automatic default. This gives equivalent checkpoint behavior for timing.

## Simplification

Graph attention now uses each stream's local routing indices. It no longer adds
global bank offsets, pads routes to another stream's ID count, flattens streams,
then reverses the offsets inside a nested tile-gather helper. Shared key/value
projections and prediction heads remain batched; each attention tile retains the
256 MiB workspace calculation. Valid gather indices are computed once for both
keys and values. The causal mask still produces exactly zero aggregate for an
empty history.

The model's unused `_recordings` copy and the datamodule's otherwise unused
`records` mapping are removed. The reviewed manifest and active event recordings
remain the source of provenance and coverage. The datamodule's `test_data` and
`train_sampler` attributes have active runtime/callback consumers and remain.

## Verification and reproduction

The final tree collects **110 cases** and passes all 110 without skips on A100.
With CUDA hidden, the CPU selection passes **90**, with the remaining 20
deselected. Every pytest invocation, including collection, ran under SLURM with
the project preamble/environment. Existing forced-tiling, independent numerical
and gradient oracles, causal masks, provenance/coverage, and checkpoint/RNG-resume
contracts remain covered. The runtime test additionally checks automatic
checkpoint enabling with a configured callback and the Ray-only default without
one.

The production-size independent oracle also passes for 16,384-event chunks with
one and two streams, including nonzero incoming history, outputs, state caches
and parameter gradients, using the existing tolerances and 32 GiB ceiling.

Complete-fit comparisons verify exact physical score fields, coverage, source
identities, optimizer-step counts, checkpoint contracts, saved Python/NumPy/Torch/
CUDA RNG states, and checkpoint sidecar hashes. Learned scores and weights are
not bitwise identical: same-seed fits of the unchanged CUDA baseline also vary.
The raw numerical differences are retained in `comparison.json`; these runs do
not establish equivalence of complete learned trajectories. Numerical/gradient
parity is checked separately against the independent serial oracle.

Ruff, formatting, lockfile validation, documentation lint and strict documentation
build pass.

Artifacts, frozen source trees, per-run source hashes, resolved configurations,
cohort manifests, packed scores, checkpoint sidecars, Python profiles, CUDA traces,
and raw logs are under:

`/fs/ess/PAS1266/graphids/dev/rf15/experimentruns/research-profile-20261024T185500Z/`

`baseline.sbatch` and `candidate.sh` record the worker-fit sequence;
`public.sbatch` records final validation and alternating public launches.
`profile_run.py`, `public_run.py`, `production_oracle.py`, `compare.py`, and
`compare_public.py` reproduce the measurements and comparisons. Use fresh output labels/directories when
repeating them, source `scripts/slurm/_preamble.sh`, and run inside SLURM.
The harness stays with these artifacts; no new execution framework is added.

This is a bounded source-cohort performance check, not a ten-epoch production
experiment, target evaluation, or detection-quality improvement claim.
