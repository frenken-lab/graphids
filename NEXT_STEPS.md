# Next session: graph–Mamba cutover

Handoff date: 2026-10-13. Start with this document, [the implementation guide](docs/reference/graph-mamba.md), and [the research/implementation plan](vehicle-graph-mamba-joint-plan.md). `Future_steps.md` describes the historical temporal stack.

The reviewed 12-capture Impala-to-Forester pilot manifest remains at
`/fs/ess/PAS1266/graphids/dev/rf15/experimentruns/graph-mamba-real-data-review-v1/can-recording-manifest-v1.json`.
The original Ascend job `7819680` completed successfully: 32 optimizer steps,
full source validation, and 385,808 target bins in both modes with no missing or
duplicate bins. Its 1h49m runtime was dominated by per-bin streaming evaluation.
This diagnostic does not establish trained detection quality; the broader
all-six provenance review remains open.

The 2026-10-14 vectorization replaces per-bin CUDA scalar reads and per-node
cache copying with state-carrying chunk scans, batched caches, bulk scoring and
calibration. It preserves weights, checkpoint schema, coverage and thresholds.
Real-checkpoint job `7834020` compared every target bin against the original
scores: zero tolerance violations, zero alert disagreements. Target evaluation
fell from 3,721.82 s to 53.26 s with zero loader workers. The full two-worker
pilot `7834149` completed in 163.63 s: fit plus full source validation 46.82 s,
target evaluation 103.98 s, 32 optimizer steps, and complete target coverage.
Worker startup dominates much of the difference between these target timings.
Source validation inference was 15.20 s; calibration was 0.13 s.

Evidence is outside Git at:

```text
/fs/ess/PAS1266/graphids/dev/rf15/experimentruns/graph-mamba-vectorized-v1/
  checkpoint-eval/benchmark.json
  checkpoint-eval/operators.txt
  checkpoint-eval/artifacts/coverage.json
  pilot-worker-fixed/pilot.json
  memory/memory-gate.json
  regression-7834191.log
  acceptance.json
  implementation.patch
```

Warm 256-bin inference measured 13.07 ms versus the previous 2.92 s. Warm
batch-one training measured 19.44 ms versus the previous 30.43 ms. Maximum
training and chunk-inference geometry reserved 5.65 GB on an A100 40 GB.
The latest single-GPU regression passed 87 tests, including CUDA, nonzero-cache
chunk carry, tiled initial-state correction, lifecycle reuse, gradient parity,
and spawned/pinned loader workers attached to a trainer. All results record
HEAD plus a dirty worktree; the implementation changes are not yet committed.

Same-node two-GPU Ascend job `7834136` passed native training synchronization,
distributed calibration/coverage against a single-rank reference, and a public
Ray fit on two A100-SXM4-80GB devices. Reports are in `ddp-training/`,
`ddp-evaluation/`, and `ray-two-gpu/` under the evidence directory above.
The full one-GPU pilot Slurm allocation lasted 3m07s including interpreter
startup and worker cleanup; the in-script runtime above was 2m44s.

The first full-source learning run, Ascend job `7835011`, completed successfully
(exit 0) in 11m45s: 10 full epochs and 10,990 optimizer steps. Output is
`/fs/ess/PAS1266/graphids/dev/rf15/experimentruns/graph-mamba-learning-10ep-v1/`.
`learning-run.json` records fit/test completion; `fit/artifacts/learning-curve.json`
records every epoch. Mean training loss fell from 0.6541 to 0.3573. Epoch 4 was
selected by source zero-shot classification AP (0.9999995); held-out target
scores were evaluated only after source checkpoint selection.

The historical v1 Forester results (superseded by the v2 audit below) showed a transfer/calibration gap despite learning:
zero-shot classification/anomaly AP were 0.99899/0.47825; conditioned AP were
0.67125/0.86177. Combined candidate false alerts/hour were 6,551.52 zero-shot
and 774.10 conditioned over 0.15889 normal hours. These are bin-level OR alerts,
not grouped incidents or classifier-only false alarms. All 385,808 expected
bins were scored, with no duplicates or missing bins. Audit classification vs
forecast contributions, mode-dependent calibration and source/target feature
shifts before extending epochs or tuning on the target. Keep target scores out
of source checkpoint selection. Both best and final optimizer checkpoints are
preserved under `fit/checkpoints/`.

## Current state

GraphIDS 2.0 replaces all five old model families with a shared CAN graph–Mamba classifier and normal forecaster. Causal snapshots, immutable reference contexts, continuous inference, checkpoint calibration, stream batching, and data parallel training are implemented. Legacy code and checkpoints require an isolated checkout of `graphids-temporal-legacy` (`daa218eb262fb1c9f9dde067f68d4cc5309d3213`). Existing external data and run artifacts were preserved.

The agreed scope is multiple CAN datasets, both zero-shot and normal-reference conditioned evaluation, and functional cutover before detection-quality acceptance. Do not restore the old models to the active factory. Transfer to other semantic domains, Mamba-2/3, adapters, and mined motifs remain later research.

Functional validation passed:

- Full regression: 70 passed, three CUDA tests skipped; those three passed separately on an Ascend A100.
- Official CUDA backend: synthetic fit, checkpoint restore, calibration, streaming evaluation, and coverage gates passed. Profiler confirmed selective-scan and causal-convolution kernels.
- Native two-rank A100 training: gradients and parameters synchronized; no duplicated scored training bins. This used two allocations, not a same-node two-GPU Ray run.
- Public Ray launch: one-GPU fit finished successfully.
- Worst default batch geometry: 131,072 node-bin tokens and 1,048,576 messages used about 4.64 GB allocated / 5.65 GB reserved on an A100 40 GB.
- Ruff, lock consistency, documentation checks, strict MkDocs build, and whitespace checks passed.

These are synthetic implementation checks. Real-data detection quality and comparative throughput remain unverified. Validation runs predate the implementation commit and recorded the old HEAD with a dirty working tree; new runs must record the new commit.

Evidence lives outside Git at:

```text
/fs/ess/PAS1266/graphids/dev/rf15/experimentruns/graph-mamba-cutover/
  implementation-validation.json
  compatibility.json
  official/gate.json
  ddp-two-node/ddp-gate.json
  memory-a100/memory-gate.json
  cutover-reference/gate.json
  regression-15478206.log
  ray-7818517.log
```

## Broader priority: extend the reviewed real-data manifest

Read `compatibility.json` and [the earlier data audit](docs/empirical-notes/2026-10-03-vehicle-graph-mamba-data-audit.md). The inventory found 81 exact duplicate aliases. Clock/payload errors were:

| Dataset | Decreasing-clock rows | Malformed/overlength payload rows |
| --- | ---: | ---: |
| hcrl_ch | 0 | 0 |
| hcrl_sa | 0 | 0 |
| set_01 | 3,572 | 0 |
| set_02 | 5,856 | 0 |
| set_03 | 3,390 | 1,001 |
| set_04 | 8,055 | 0 |

1. Establish vehicle/session identities, timestamp units, trusted normality, label provenance, payload validity, exact aliases, and partial capture reuse. The four CTAT registry sets overlap vehicles/captures and are not four independent dataset families. HCRL normal captures also have aliases; reference/query independence cannot be inferred from filenames.
2. Resolve clock resets and malformed payloads through documented upstream evidence or explicit, provenance-preserving segmentation/cleaning. Never silently sort clocks or truncate payloads. Determine whether trusted attack intervals exist for suppression attacks; zero event labels do not establish normality.
3. Produce a version-1 recording manifest with disjoint session roles and target vehicles excluded from source training. Target reference, calibration, normal test, and attack test partitions must follow the documented contract. Set `provenance_reviewed: true` only after review, not to bypass the gate.
4. Select only verified source families in the experiment config. The default all-six preset is not permission to launch all six before this review. A small defensible subset is a better first real-data pilot.

The original audit is an inventory, not an automatic split generator. The new
review found 109 candidate reuse pairs across 252 canonical captures, including
normal captures reused in attack files. Only the small documented subset is
reviewed. No raw files were corrected, sorted, or truncated. Selected original
DoS logs have microsecond clock backsteps already sorted by the upstream curator;
this curation is explicitly recorded in `pilot-provenance.json`.

## Then run a source pilot

1. Census source snapshot size, live-node capacity, and graph workload before fitting. Defaults are 10 ms bins, 128 scored bins plus up to 128 warm-up bins, width 128, two Mamba blocks, one graph readout, eight incoming neighbors, and one-second transient retention. Choose changes using source validation only.
2. Check dense recording-cache feasibility. `max_snapshot_bytes` defaults to 4 GiB and fails before allocation. Long captures or fuzzing with many distinct IDs may need a chunked/streaming cache implementation; do not blindly raise the limit or split evaluation in a way that resets carried state.
3. Fit a small reviewed source manifest on one Ascend A100, restore its checkpoint, and evaluate whole target recordings in both modes. Inspect expected-versus-scored coverage, duplicate counts, eligibility masks, terminal bins, sampled training work, and per-dataset/per-vehicle results.
4. Keep classifier and forecast scores separate. Thresholds and normalization come from source-normal validation. Target calibration is a declared partition and is not used to refit primary evaluator thresholds. Single-bin candidate alerts do not yet implement persistence/cooldown.
5. Validate Cardinal H100 execution before relying on that platform for larger jobs. Same-node two-GPU native and Ray A100 gates have now passed.

Current zero-shot inference uses a self-only graph; reference-conditioned inference supplies relational context. The forecasting encoder is trained jointly with supervised labels, so this is not a purely unsupervised detector. Forecast heads predict counts, arrivals, and payload change, not full payload bytes.

## Research gates after the pilot

Lock dataset/vehicle holdouts, normal exposure accounting, interval metrics where independently supported, and source-only model selection. Compare classification-only, forecasting-only, Mamba-only, graph contribution, and context withholding across multiple seeds. Report per-family/mode and worst-family results alongside macro averages. Compare the archived stack on identical hardware and an identical defensible evaluation protocol; synthetic timings are not evidence of a speed advantage.

## Practical entry points

- Model and streaming state: `graphids/core/models/graph_mamba.py`.
- Manifest, snapshots, contexts, scheduling, and online binner: `graphids/core/data/snapshots/`.
- Presets: `configs/experiments/graph_mamba_joint*.yml`; set the reviewed manifest path before launching.
- Functional probes: `scripts/graph_mamba_gate.py`, `scripts/graph_mamba_ddp_gate.py`, and `scripts/graph_mamba_memory_gate.py`.
- Runtime and migration checks: `graphids/runtime_checks.py` and `graphids/migration.py`.

Run pytest, Torch workloads, fits, and CUDA extension builds inside Slurm allocations, following `CLAUDE.md`. Ascend A100 is the primary platform; Cardinal H100 is the alternative. The presets use account `pas1266`; verify current allocation/module availability when submitting.

The lock pins PyTorch 2.8, official Mamba `v2.2.6.post3` at `10b5d6358f27966f6a40e4bf0baa17a460688128`, and causal-conv1d 1.6.1. The newer Mamba revision inspected during research conflicted with this Torch/Triton combination. Build with GCC/CUDA explicitly, avoiding inherited Pitzer compiler paths:

```bash
module purge
module load gcc/12.3.0 cuda/12.8.1
export CC=gcc CXX=g++ MAX_JOBS=4
uv sync --extra dev
uv sync --extra dev --extra mamba --no-build-isolation
python -m pytest
python scripts/graph_mamba_gate.py --output /path/to/new-session-gate
```

Use `backend: reference` only for explicit diagnostic checks. Production requires working CUDA selective-scan and causal-convolution extensions. The historical cutover, both one-GPU pilots, and the new two-GPU native/Ray gate have completed. Learning job `7835011` also completed; review its transfer/calibration results above before submitting another source-learning fit.


## Corrected evaluation and conditioning audit, 2026-10-14 (v2)

The corrected human report is
`/fs/ess/PAS1266/graphids/dev/rf15/experimentruns/graph-mamba-conditioning-v2/final/report-v2.txt`.
`final-report-v2.json` contains all vehicle/dataset/recording breakdowns, both
calibration tracks, per-head bin alerts, distributions and coverage. All eight
predeclared interventions are accounted for; full reference also supplies the
reference-free baseline. Each policy scored all 385,808 expected target bins.
Source validation and calibration were replayed separately before target use.
Original artifacts and epoch-4 selection remain unchanged. Forester is now
known diagnostic evidence, not an independent holdout.

Saved-score AP corrected with historical pooled normalization is
0.99887668/0.97970199 (classification/anomaly, zero-shot) and
0.65128708/0.99816880 (conditioned). With corrected per-mode source calibration,
anomaly AP is 0.98681100 zero-shot and 0.97852997 conditioned. Classification AP
is unchanged by threshold calibration. Target-normal threshold fitting leaves
scores/AP unchanged; it does not reliably improve normal-test bin alert rates.

At selected epoch 4, removing graph messages raises target classification AP
to 0.97743439; expiring reference anchors raises it to 0.99908586. Expiry also
reduces source anomaly AP from 0.99803473 to 0.99407209 and increases primary
target false-bin alerts/hour. Recent classifier pooling (reference IDs observed
in the current bin; all live transients retained) raises target classification
AP to 0.81921039 and leaves every saved forecast component/score unchanged over
214,408 source and 385,808 target bins. Persistence, graph messages and pooling
interact; these interventions do not establish a unique cause. Source validation
supports broader follow-up of profile and graph-message ablations under the
minimum-of-four AP criterion; target ranking does not select an intervention.

The original `last.ckpt` is byte-identical to epoch-4 `best.ckpt`, not the final
weights. Ray retained all ten epoch snapshots. The true final snapshot is
`fit/checkpoint_2026-10-14_14-32-21.050864/checkpoint.ckpt` under the original
learning run. Slurm job 7837833 recomputed all ten source epochs from retained
weights and evaluated the actual epoch-10 checkpoint. Its target classification
AP is 0.99883301 zero-shot and 0.99786863 conditioned. The severe conditioned
classification failure is therefore specific to selected epoch 4. This finding
does not replace the original checkpoint using target results. Corrected curves
and source scores are under `graph-mamba-conditioning-v2/corrected-curve/`.
Historical curve values are explicitly annotated as affected by sigmoid.

Use `configs/experiments/graph_mamba_ctat_learning_corrected.yml` for a future
run. It monitors `val/source_min_vehicle_macro_ap` and independently saves the
final weights every epoch, including when monitored AP declines or ties. The
completed experiment's original preset/rule remains intact. Learning-curve
metadata now reports the configured selection metric and evaluation version.

Validation: the final full regression passed 114 tests including official CUDA;
final targeted checks covered checkpoint ties and nonfinite unknown/ineligible
scores. Two CPU ranks inside Slurm matched single-rank calibration, received
identical global checkpoint metrics, and covered all 1,920 synthetic bins.
A two-GPU job never received an allocation and was canceled; no new two-GPU
claim is made. Ruff and whitespace checks passed. Additional reviewed vehicles
and a new independent holdout are needed before broader generalization claims.
