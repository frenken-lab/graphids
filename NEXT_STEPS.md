## 2026-10-16 consolidation

The sole active pipeline is now `can_event` / `can_event_dm` /
`event_graph_mamba`, with continuous history, chunk-boundary detachment and
independent streams on one GPU. Snapshot code, configurations and tests,
including the previous uncommitted working tree, are preserved in
`archives/snapshot-2026-10-16.tar.gz` with verified hashes and extraction smoke.
Earlier operational instructions below are historical. Current commands and
contracts are in `docs/reference/graph-mamba-events.md`.

# Next session: graph–Mamba cutover


## 2026-10-16 input requirement: preserve individual messages

The user requires raw message-level observations so the model learns its own
representations. The next learning path is `can_event_dm` + `event_graph_mamba`,
configured by `configs/experiments/graph_mamba_events_source.yml`; see
`docs/reference/graph-mamba-events.md`. Each CAN ID, payload, length and timestamp
reaches Mamba in recording order; execution chunks do not merge observations.
Learned graph attention uses preceding per-ID hidden states, without the fixed
count/change reference graph. The existing snapshot path/checkpoints remain
comparison artifacts. This requirement supersedes the earlier suggestion to make
additional hand-engineered summary features the primary next input design.

The new path uses source-only data, frame BCE and a separate normal-only byte
likelihood objective. It is initially single-device, with explicit capacity
errors and no silent drops. It does not yet implement missing-arrival survival,
trusted operational fault-state labels, reference-prefix conditioning, or new
calibration. Implementation verification is separate from detection evidence:
no full encoder fit or target evaluation is part of this change. The next
accuracy comparison must distinguish attacker-frame attribution from anomaly
ranking and report native frames alongside common bus bins.

Verification completed: all 21,285,666 frames in 21 source query recordings were
compared field by field with their CSVs; no rows were dropped, merged, or changed.
CPU regression and official A100 forward/backward, state-carry and Lightning
learning smoke checks passed. Evidence is in
`/fs/ess/PAS1266/graphids/dev/rf15/experimentruns/graph-mamba-events-source-v1/verification/`.


Handoff updated: 2026-10-15. Start with the expanded-source follow-up at the end
of this document, [the implementation guide](docs/reference/graph-mamba.md), and
[the research/implementation plan](vehicle-graph-mamba-joint-plan.md).
`Future_steps.md` describes the historical temporal stack. Earlier sections
below preserve historical run state; evaluation fixes and vectorization are now
committed in `e944f21`.

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

## Expanded-source follow-up, 2026-10-15

The expanded pilot uses Impala and Silverado as sources and reserves Traverse for
later Graph-Mamba evaluation. Forester is excluded as previously inspected
diagnostic evidence. The explicit pre-score role plan is
`configs/recording_reviews/ctat_impala_silverado_to_traverse_v1.json`.
Source numbered normal/DoS captures split train versus validation; accessory
captures supply separate contexts. Traverse holds the longer driving capture
(`attack-free-1`, 444.589 s) for normal testing, the longer accessory capture
(`accessory-1`, 264.698 s) for calibration, and the remaining two normal captures
for reference. Only Traverse DoS-4 (253.148 s) is included. Its local clocks are
monotonic; DoS-3 is excluded for 31 decreasing-clock rows.

CPU job `56219353` froze 17 captures covering 12,334,605 rows after complete
timed-frame correspondence to the same pinned upstream commits, selected-path
hash checks, label-file identity, integrity checks, acquisition-interval
separation and the existing sequence-reuse screen. No candidate reuse pair
joins selected captures. Original DoS logs have upstream-curated microsecond
backsteps, explicitly recorded in the new provenance; local files were not
changed. Review artifacts and the frozen manifest are under:

```text
/fs/ess/PAS1266/graphids/dev/rf15/experimentruns/graph-mamba-expanded-review-v1/
```

The scope remains one dataset family and capture-level independence. Traverse
was part of historical temporal-stack holdout plans, so it is reserved for this
Graph-Mamba follow-up, not a wholly untouched dataset. Its normal-test exposure
is only 0.12350 hours. These data cannot certify a low false-alert rate.

The expanded preset is `configs/experiments/graph_mamba_ctat_expanded_source.yml`.
It inherits corrected source-only checkpoint selection and independent final
weight saving. Use `scripts/graph_mamba_learning_run.py --source-only`; this
stops after fitting without scoring target queries. The driver's default is
now the corrected preset. The diagnostic driver also supports `--source-only`
and explicit `--datasets`, with exact source-bin coverage checks. The pilot
preflight accepts explicit source datasets rather than hard-coding Impala.

Review job `56219353` completed with its three regression tests passing. CPU
source-preflight job `56219365` also completed: 11 targeted checkpoint/config
tests passed, the expanded config resolved, and all 1,907 source training windows
passed packing checks. Source-derived anchor and transient capacities are both
128. Maximum source work is 25,088 node-bin tokens and 200,704 graph messages,
within defaults. Source run artifacts, the predeclared pilot protocol, source
snapshot and one-A100 learning script are under:

```text
/fs/ess/PAS1266/graphids/dev/rf15/experimentruns/graph-mamba-expanded-source-v1/
```

Ascend job `7867129` was submitted after successful preflight. It runs ten source
epochs at seed 42, then replays frozen selected weights with full reference, no
profiles and no graph messages on source validation only. The balanced sampler
uses 808 windows per vehicle per epoch (1,616 total), rotating omitted Impala
windows across epochs; a full epoch means this complete balanced schedule,
not all 1,907 available windows each epoch. The criterion remains the minimum
of four source vehicle-macro APs across both heads and modes. This is a one-seed
pilot; broader source validation needs multiple seeds before choosing a
conditioning policy. Keep Traverse model scores deferred until the source
protocol is frozen. Check `learning-run.json` for fit state and
`source-conditioning/diagnostics-v2.json` for completed policy replays; absence
of these files does not establish a completed learning run. New changes remain
uncommitted; evidence includes the source archive and per-file hashes.
Job `7867129` completed successfully (exit 0) in 23m44s, including 41 targeted
tests, the source fit and all three frozen-weight policy replays. Fitting took
1,118.92 s, completed ten epochs and 16,160 optimizer steps, and reduced mean
training loss from 0.510270 to 0.304173. The corrected source criterion selected
epoch 9, with a minimum-of-four vehicle-macro AP of 0.99996632. The selected
checkpoint's source replay matches that epoch's metrics. Epoch 10's conditioned
anomaly macro AP dropped to 0.99027318 despite nearly flat training loss, so
preserve source-only selection rather than assuming final weights are best.
Best and last checkpoint hashes are distinct and match their SHA256 sidecars;
the selected hash also matches the learning and diagnostic reports. All recorded
Python source hashes still match the working tree.

Each policy replay covered all 368,308 expected source bins across both modes,
with no duplicate or missing bins. The following are vehicle-macro APs, not
pooled-across-vehicle APs:

| Frozen-weight policy | Zero-shot classifier | Zero-shot anomaly | Conditioned classifier | Conditioned anomaly | Minimum of four |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full reference | 1.00000000 | 0.99996634 | 1.00000000 | 0.99998853 | 0.99996634 |
| No profiles | 1.00000000 | 0.99996634 | 0.99999987 | 0.98786137 | 0.98786137 |
| No graph messages | 1.00000000 | 0.99996634 | 1.00000000 | 0.99997466 | 0.99996634 |

The expanded-source evidence does not support adopting either ablation.
Removing profiles especially harms Silverado conditioned anomaly AP
(0.99999133 to 0.97609663). Removing graph messages ties the selection criterion
because zero-shot anomaly AP is the limiting term, while slightly reducing
conditioned anomaly AP. These are frozen-weight interventions, not comparisons
of independently retrained architectures.

Full reference produced 266 zero-shot and 276 conditioned normal candidate bin
alerts over 0.383005 source-normal hours (694.51 and 720.62 bin alerts/hour).
These recordings also fit the thresholds, so these are calibration diagnostics,
not independent false-alert estimates. At 10 ms bins, a 0.999 marginal threshold
quantile already implies roughly 360 exceedances/hour/head on calibration data;
both heads measured 360.31/hour. Combined OR bin alerts do not count incidents.

`completion-summary.json` collects completion, checkpoint integrity, APs, coverage
and interpretation in the source run directory. Traverse remains unscored:
`learning-run.json` records `target_evaluation_deferred: true`, and the diagnostic
report records `target_status: not evaluated`. The user waived additional seeds
and authorized the frozen Traverse evaluation with parallel inference. Do not
extend epochs or remove profiles based on this one seed. No additional jobs
were launched during the completion review itself.

## Frozen Traverse evaluation with parallel inference

The selected epoch-9 checkpoint, full-reference policy, recording manifest and
reporting protocol are frozen under
`/fs/ess/PAS1266/graphids/dev/rf15/experimentruns/graph-mamba-traverse-eval-v1/`.
The checkpoint SHA256 is
`0cf4c7f58533fbb07f3649d60ac96f1ad08586daeb92ffd1670a3b54f113035d`.
Primary evaluation uses the checkpoint's existing per-mode source calibration,
with no refitting. Secondary conditioned evaluation fits only thresholds on
the declared target calibration recording. Both modes score identical queries;
AP excludes calibration and unknown labels. No incident grouping is added.

The prior per-role/mode test loaders could leave a second GPU idle when each
loader contained one capture. Test inference now combines all role/mode entries
in one schedule, retaining whole recording/mode ownership and causal state.
The regression test checks that both ranks receive work and no stream crosses
ranks. The distributed synthetic gate additionally checks target scores against
the former serial per-role/mode iteration within `rtol=2e-4, atol=2e-5`.

Ascend job `7870653` requests two A100s on `nextgen`; the debug partition rejects
two-GPU requests under its QoS. The job runs targeted CUDA regressions and the
two-GPU synthetic gate before Traverse. A separate CPU job `56221656` checks the
schedule and two-rank reference gate while waiting for allocation. The production
driver is `scripts/graph_mamba_parallel_eval.py`; it explicitly requires multiple
ranks, real scored work on every rank, disjoint recording/mode ownership,
identical frozen calibration and exact global coverage. Completion and actual
rank workloads will be recorded in `evaluation/parallel-evaluation.json`.
CPU job `56221656` completed successfully in 52 s: six datamodule tests passed
and one CUDA-only test was skipped. The two-rank reference gate scored 960 target
bins per rank, matched serial scores within tolerance, matched single-rank
source calibration, and covered all 1,920 synthetic bins without gaps or
duplicates. Ruff, documentation lint and whitespace checks passed.
Job `7870653` passed 46 targeted CUDA tests but stalled at the first NCCL barrier
on `a0328`, before creating model caches or scoring Traverse. A separate
two-GPU all-reduce smoke test on the same allocation passed with explicit CUDA
device binding and `NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1` (shared-memory transport).
The stalled job was canceled; retry `7870709` uses those settings and explicit
device binding, with the same frozen checkpoint/protocol and no target results
from the first attempt. The retry reruns the distributed CUDA gate before
Traverse. Both communication settings are captured in the final run report.
Retry `7870709` completed successfully (exit 0) in 1m32s. Its two-A100 CUDA gate
passed serial score parity and exact coverage with 960 synthetic target bins per
rank. The real evaluation took 63.43 s including preparation, inference, score
gathering and artifact output. Both ranks scored 96,244 actual target bins:
rank 0 owned zero-shot streams and rank 1 owned conditioned streams. All 192,488
expected bins, including the separate calibration recording in both modes, were
covered with no duplicates or missing bins. Do not interpret the runtime as a
measured speedup over serial real-data inference; no such benchmark was run.

The frozen source calibration remained byte-equivalent at the JSON value level
before/after evaluation and identical on both ranks. The checkpoint, manifest,
configuration and recorded Python source hashes were verified. Target calibration
retained the source component centers, scales and availability masks; every AP
in the primary and secondary reports is identical. Only the secondary thresholds
and associated track/exposure metadata change.

| Track and mode | Classification AP | Anomaly AP | Normal classifier bins | Normal anomaly bins | Normal combined OR bins | Combined normal bin alerts/hour |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Source-calibrated zero-shot | 1.00000000 | 0.99936652 | 221 | 1,409 | 1,568 | 12,696.67 |
| Source-calibrated conditioned | 1.00000000 | 0.99983716 | 20 | 156 | 176 | 1,425.14 |
| Target-calibrated conditioned (secondary) | 1.00000000 | 0.99983716 | 924 | 233 | 1,151 | 9,320.06 |

Counts use 444.589 s (0.123496994 h) of verified-normal driving test exposure;
they count candidate bins, not grouped incidents. Full conditioning reduces
primary normal OR bin alerts by 88.78%. Target accessory-mode threshold fitting
worsens normal driving-test alerts, despite unchanged ranking and normalization.
This supports strong ranking transfer on this reviewed normal/DoS-4 subset,
while threshold transfer and usable alert reporting remain unresolved. No target
result changes the epoch-9 selection or full-reference policy.

Human results are in `report.txt`; `completion-summary.json` contains compact
metrics, verification and interpretation. Detailed per-recording distributions,
per-head/overlap alerts, both tracks and raw scores are under `evaluation/artifacts/`.
`evaluation/parallel-evaluation.json` records rank ownership, actual scored bins,
GPUs, peak memory, timing and the unchanged primary source bundles. The retry's
exact implementation is archived in `source-snapshot-retry1.tar.gz`.

Traverse is now inspected diagnostic evidence. The next work is to predeclare
a persistence/cooldown alert policy using source data and obtain longer
independent normal exposure, then use an independent target for acceptance.
Do not tune that policy or choose another checkpoint using this Traverse result.
No additional seeds or fits were launched.

### Threshold metrics from the retained Traverse scores

CPU job `56223139` derived confusion counts, precision, recall, F1, accuracy,
FPR, specificity, balanced accuracy, MCC and ROC-AUC from the existing scores,
without model inference or threshold fitting. Mode-wise processing used two
threads. Results are in `threshold-metrics.json` and `threshold-metrics.txt`
under the Traverse evaluation root and appended to `report.txt`.

The following use the frozen primary source thresholds. All values are percent:

| Mode/head | Precision | Recall | F1 | Accuracy | FPR |
| --- | ---: | ---: | ---: | ---: | ---: |
| Zero-shot classifier | 84.81 | 100.00 | 91.78 | 99.66 | 0.3507 |
| Zero-shot anomaly | 47.40 | 99.93 | 64.30 | 97.87 | 2.1715 |
| Zero-shot combined OR | 44.71 | 100.00 | 61.79 | 97.63 | 2.4213 |
| Conditioned classifier | 97.88 | 100.00 | 98.93 | 99.96 | 0.0424 |
| Conditioned anomaly | 84.21 | 99.93 | 91.40 | 99.64 | 0.3668 |
| Conditioned combined OR | 82.82 | 100.00 | 90.60 | 99.60 | 0.4062 |

Classifier and combined denominators are 69,774 known-label query bins per
mode: 1,340 positive and 68,434 negative. Anomaly metrics exclude two negative
bins without eligible forecasts, leaving 69,772 bins. Calibration and unknown
labels are excluded. FPR counts all labeled-negative query bins, including
background in the attack capture, whereas the earlier normal alert rates use
only verified-normal test recordings. Thus overall FP and normal-test alert
counts differ.

The conditioned classifier has TP=1,340, FP=29, FN=0, TN=68,405. Conditioned OR
has TP=1,340, FP=278, FN=0, TN=68,156. At these frozen thresholds, adding forecast
OR contributes 249 false-positive bins and no additional true-positive bins.
The zero-shot OR similarly adds 1,417 false-positive bins and no true positives.
Classifier ROC-AUC is 1.0 in both modes; anomaly ROC-AUC is 0.99991793 zero-shot
and 0.99999592 conditioned. No continuous OR ranking score was declared, so OR
ROC-AUC is intentionally unavailable.

These results flag the alert combination rule for source-only follow-up. They
do not authorize selecting or tuning a replacement using Traverse. Grouped
incident recall, detection delay and persistence/cooldown remain unmeasured.

## Source-only alert-policy audit

CPU jobs `56225108` and `56225124` completed successfully in 20 s each. The final
audit used two worker processes and the saved full-reference source-validation
scores only: 368,308 bins, both modes, two source vehicles and 90 policy settings.
No model weights or target scores were read, no inference ran, and the saved
source thresholds remained unchanged.
The protocol was fixed in
`configs/analysis/graph_mamba_source_alert_policy_v1.json` before inspecting these
source results. It compares classifier, anomaly and OR decisions with 1/3/5/10/20
consecutive-bin confirmation and 0/1/5 s cooldown. The script is
`scripts/graph_mamba_alert_policy_audit.py`.

Artifacts are under
`/fs/ess/PAS1266/graphids/dev/rf15/experimentruns/graph-mamba-source-alert-audit-v1/`:
`report.txt`, `completion-summary.json`, `candidate-policy.json` and the full
`results-v2/audit.json`. Initial results remain preserved. The second pass adds
exact positive-run identity checks, so equal hit counts cannot hide a lost run,
and verifies coverage and raw normal-alert counts against the saved source
report. All underlying policy metrics match the initial pass. Eight causal
policy regression tests passed, including prefix invariance, no backdating,
record resets, bin gaps, cooldown delays and missed positive-run identities.

Notifications are emitted at most once per contiguous candidate run after
confirmation. If cooldown has not expired, notification waits while the candidate
remains active. Cooldown changes notifications only, never active-bin decisions.
Recordings/modes reset both run state and cooldown. Labels never affect the
policy state. Counts below are pooled active-bin F1 and verified-normal
notifications; normal exposure is 0.383005 h per mode. Modes are paired repeats
of the same captures, not independent extra exposure.

| Policy | Zero-shot bin F1 | Conditioned bin F1 | Zero-shot normal notifications | Conditioned normal notifications | Maximum positive-run notification delay |
| --- | ---: | ---: | ---: | ---: | ---: |
| OR, one-bin confirmation, no cooldown | 93.28% | 90.95% | 258 | 272 | 0 ms |
| Classifier, one bin, no cooldown | 96.33% | 96.38% | 133 | 134 | 0 ms |
| Anomaly, one bin, no cooldown | 96.40% | 94.14% | 137 | 138 | 10 ms / 0 ms |
| Classifier, one bin, 5 s cooldown | 96.33% | 96.38% | 73 | 74 | 0 ms |
| Classifier, three bins, no cooldown | 99.88% | 99.88% | 0 | 2 | about 20 ms |
| Classifier, five bins, no cooldown | 99.84% | 99.84% | 0 | 0 | about 40 ms |

Forecast OR adds zero positive-bin detections and 166/304 false-positive bins
in zero-shot/conditioned source evaluation. Classifier three-bin confirmation
has TP=2,447, FP=2, FN=4 in each mode: precision 99.9183%, recall 99.8368%, F1
99.8776%. Its worst source vehicle has recall 99.5833% and F1 99.6872%. Both
observed positive-label runs still receive notifications. The four lost positive
bins are the first two bins of each run, so this candidate fails the strict
no-positive-bin-loss audit flag; it is an explicit delay/false-alarm tradeoff.
Five-bin confirmation loses eight positive bins per mode and adds about 40 ms
delay. It should not be promoted solely for observing zero false alarms here.

Cooldown alone leaves active-bin F1 unchanged. A five-second cooldown on the raw
conditioned OR policy delays one positive-run notification by about 2.8 seconds.
The leading research candidate is therefore classifier-only, three consecutive
10 ms bins, no cooldown, using the same frozen per-mode thresholds. It is saved
as `candidate-policy.json`; no production default or active model config changed.
The conservative alternative preserving every positive-bin hit is classifier-only
with one-bin confirmation; the observed cooldown benefit is notification reduction
only and needs separate acceptance of its delay risk.

There are only two underlying positive-label DoS runs, with no independently
verified attack intervals. Run hit/delay metrics are diagnostic proxies relative
to the first positive bin close, not certified incident recall or physical
attack-onset latency. The source normal captures also fit the thresholds.
Shorter attacks may be fully suppressed by persistence and remain untested.
Next broaden the reviewed attack coverage before promoting this DoS-derived
candidate. Do not tune or promote it using the already-inspected Traverse
recordings.

## Current priority: broaden attack coverage before alert-policy selection

The user correctly identified that all recent training, target evaluation and
alert-policy analysis cover normal traffic and DoS only. The original small
manifest was an execution/provenance pilot. Extending vehicles while retaining
only DoS did not address the intended multi-attack research question. In
particular, the observed lack of incremental forecast detections is evidence
about these DoS captures only; it does not justify removing forecasting for
spoofing, timing-preserving manipulations, missing messages or unseen attacks.
The three-bin candidate may suppress short attacks and is deferred pending
broader attack evidence.

Other attacks exist in the lake. The original recording inventory shows eight
non-DoS `set_01/train_02_with_attacks` captures (force-neutral, RPM, standstill,
targeted payload fuzzing), all with clock backsteps. Of ten non-DoS training
candidates in `set_03`, only fuzzing-4 passes the initial clock/payload/label
screen; it still needs full provenance review and a defensible split. The
`set_02` inventory has four such screen-passing captures (fuzzing-1/2 and
systematic-1/2), with reuse leads affecting fuzzing-1. These are inventory leads,
not reviewed training/evaluation permission. Some valid fuzzing/systematic
captures have 1,935-2,048 distinct IDs and estimated dense caches of 5.1-12.4 GB,
above the 4 GiB guard. These require a justified sparse/chunked cache strategy,
not blind budget increases. Existing reuse/clock/payload issues and absent
suppression interval annotations need documented resolution, rather than
indefinite exclusion from the research program.

The next substantive work is to review and freeze additional attack families,
resolve their provenance/time/capacity requirements, and refit a multi-attack
source cohort. Predeclare per-family source validation and separate held-out
vehicle/attack-family tests, keeping the heads separate and including a simple
traffic-rate baseline. Evaluate forecasting and persistence on short and
non-flooding attacks before choosing a default alert policy. Suppression recall
requires independently justified attack intervals; absent those, use clearly
marked sensitivity probes without presenting them as real-attack recall.


### Multi-attack retraining now authorized (2026-10-15)

The user approved including other attacks and retraining, with parallel evaluation
and no extra seeds. The proposed 34-record cohort was reduced to 30 after checking
pinned upstream labeled hashes: targeted-payload-fuzzing and valid-systematic
curated variants differ from the original labeled recordings and require a
separate derivation audit. Do not describe them as reviewed or trained.

The working plan is `configs/recording_reviews/ctat_multiattack_v1.json`; evidence
lives in `$GRAPHIDS_RUN_ROOT/experimentruns/graph-mamba-multiattack-review-v1/`.
Source training includes DoS, gear, RPM, standstill, double spoofing and triple
spoofing across Impala/Silverado. Random-ID fuzzing is added to source validation,
not optimizer training. Validation-only fuzzing is visible to checkpoint selection
and therefore is not an independent unseen-family test.

Global candidate-reuse connected components (including unselected intermediates)
and overlapping original acquisition intervals must stay within one role/split.
Normal/standstill variants and gear/double variants sharing traffic are grouped
as related queries, not independent captures. Separate source references remain
required. Full raw/upstream frame correspondence and label-file identity precede
any derived clock repair. The new freezer preserves every frame and its order,
repairs clock backsteps within each verified replay interface (prefix maximum,
rejecting corrections over 1 ms), then stably merges interface streams by corrected
time while preserving their internal order. Every moved/corrected row has an audit
entry with original and derived line numbers/timestamps/interface. The full clock
audit found up to 7.895 ms of global interleaving but at most 69 microseconds of
within-interface backsteps; a global prefix maximum would move valid frames by
several milliseconds. Raw lake files remain untouched. Ordered row/frame/label
correspondence is required before assigning CSV rows to replay interfaces.
Existing reviewed normal/DoS correspondence may be reused only when the raw,
upstream log URL/hash and normality match; provenance records hashes of that prior
review. Full verification is performed for new recordings.

The new source config `graph_mamba_ctat_multiattack_source.yml` requests two A100s,
one seed, ten complete vehicle-balanced source schedules, and source-only fit.
Selection uses `val/source_min_family_macro_ap`: equal mean across attack families,
within-family vehicle macro, then minimum across classifier/forecast and both
conditioning modes. Per-family comparisons include verified-normal query traffic
from the same vehicle; they do not report attack-only AP. Fixed-threshold reports
now also include TP/FP/FN/TN, precision, recall and F1. Normal exposure and false
bin alerts/hour remain available. The original vehicle AP metric remains for
compatibility, but does not select this run.

`graph_mamba_parallel_eval.py --partition source` replays the chosen checkpoint
without refitting calibration, requires real work on every rank and exact scored
coverage, and writes per-family metrics plus scores and a high-frame-rate baseline
(source-normal 99.9th percentile; event count / bin exposure). The baseline is
reported at the same per-family level to expose flood-detection shortcuts. Target scoring remains
deferred until source review; existing Traverse normal/DoS is previously inspected
diagnostic evidence. No persistence/head deployment decision follows from DoS alone.

Initial download job: Pitzer 56227810 (completed). Revised test/freezer job:
56228228. Source preflight 56228352 was canceled after the original-log correspondence
check rejected a non-DoS capture. Non-DoS labeled CSVs originate from pinned
`can-ml/post-attack-unlabeled` replay logs, not same-named original `can-dataset`
logs. Replacement replay download: 56228527; review/freezer: 56228612; preflight:
56228628. Replay provenance also verifies event labels from the odd-numbered
vcan interfaces used by the pinned conversion helper. Two-GPU native family
metric parity and public Ray fit gates: Ascend 7872378, completed successfully;
native target fixture coverage 1,920 bins (960 per rank), no missing/duplicates,
scores match serial, and source metrics match on both ranks. Initial CPU checks:
45 passed, one skipped. Update final review/preflight and retraining state before
handoff. Source artifact root:
`$GRAPHIDS_RUN_ROOT/experimentruns/graph-mamba-multiattack-source-v1/`.


Current submission state, superseding the investigation job IDs above:

- Final content/clock/split review **56230530 completed** in 4m31s: 30 recordings,
  26 capture groups; 25 source recordings and five reserved target recordings.
  Twelve derived replay CSVs preserve all frames; 11 timestamps changed by at most
  69 microseconds, and 770 rows moved across interfaces. Replay clock overlap
  is grouped only when a virtual interface is shared; all known background-reuse
  graph components are grouped regardless of bus. Concurrent disjoint virtual
  buses do not by themselves establish shared acquisition. Nine focused checks
  passed; the earlier regression batch passed 46 checks with one CUDA skip,
  supplemented by the successful native/Ray two-GPU gates.
- Source cache/capacity/packing preflight **56230544 is running**. It checks every
  training and validation window and verifies positive training exposure for
  every included attack family in all ten balanced schedules. The 4 GiB cache
  guard and batch-work limits remain unchanged.
- Ascend retraining **7872845 is submitted and held** until automatic release job
  **56230770** runs after a successful preflight. No user approval is pending.
  The single two-A100 allocation first gates the exact parallel source-replay
  driver using the retained toy checkpoint, then fits ten epochs at seed 42,
  and finally replays the source-selected best checkpoint in parallel without
  refitting calibration. Earlier standalone source-driver gate 7872553 and
  replaced training submission 7872822 were canceled while pending.
- Read `graph-mamba-multiattack-source-v1/pipeline-status.json`, `learning-run.json`,
  `fit/artifacts/learning-curve.json`, and
  `selected-source-eval/parallel-evaluation.json` for subsequent state/results.
  The selected-source artifacts include family metrics, fixed-threshold confusion
  counts/precision/recall/F1, false bin alerts/hour and `traffic-rate-baseline.json`.
  Submission code is archived in `source-snapshot.tar.gz`; resolved experiment
  config is `experiment.yml`, and reviewed cohort metadata is in `cohort-review/`.

Do not report this fit as completed until its job/results say so. No real target
query evaluation is scheduled. Next compare per-family heads against the rate
baseline, then review excluded payload-fuzz/systematic derivations and independent
low-rate/masquerade/suppression evaluation; suppression still requires trusted
interval evidence. The three-bin DoS alert policy remains deferred.


### Completed mixed-attack source fit (status checked 2026-10-16)

Ascend 7872845 completed successfully in 33m42s. Preflight 56230544 and automatic
release 56230770 also completed. All ten epochs ran (12,750 optimizer steps;
fit 1,763.7 seconds). The source-selected best is epoch 6 using
`val/source_min_family_macro_ap` = 0.34399578; checkpoint SHA256
`e513006088ffd06a6135a21dcc30fdf208ae1aed17ddbcdaeecb30c6ca86dcc6`.

The frozen-checkpoint source replay completed on two A100s, with 694,916 scored
bins on each rank, 1,389,832 total expected/scored bins, no missing/duplicate bins,
and unchanged source calibration. No real target query evaluation ran.
`pipeline-status.json` now says complete and contains pooled conditioned family
confusion counts and vehicle-macro APs. Full metrics and rate baseline remain in
`selected-source-eval/artifacts/`.

At fixed source thresholds, conditioned classifier recall is 100% on DoS,
99.85% standstill, 84.74% double spoofing and 97.83% triple spoofing. Gear pooled
recall is only 0.73% (classifier) / 2.19% (forecast), and RPM is 0% / 0.80%.
Validation-only random-ID fuzzing is missed by classification; forecasting reaches
94.19% recall but only 19.40% precision (F1 32.17%). These are source-validation bin
metrics, not independent target or incident-level results. The rate baseline has
very low AP on the non-DoS families; strong standstill/combined classification
therefore adds value beyond this simple high-rate detector, while gear/RPM remain
unresolved. The next substantive investigation should audit gear/RPM supervision,
training exposure/positive-bin balance, features and loss contributions before
another tuning run or expanding target claims. Do not adopt a classifier-only
policy: forecasting adds random-ID-fuzzing sensitivity, albeit with substantial
false alerts at this threshold. No new seeds or fits were launched in this status
check.

### Source-only gear/RPM diagnosis completed (2026-10-16)

The completed investigation is documented in
[the evidence report](docs/empirical-notes/2026-10-16-missed-attack-diagnostics.md)
and [CLI protocol](docs/reference/missed-attack-diagnostics.md). Artifacts are in
`$GRAPHIDS_RUN_ROOT/experimentruns/graph-mamba-missed-attacks-v1/`.
Ascend 7890431 completed in 50m10s; final CPU reporting 56283156 completed in
18m25s. The source-only audit covers all 25 reviewed recordings / 21 capture
groups / 22,382,542 frames. Frozen extraction covers 1,145,440 query bins on two
A100 ranks with no missing/duplicate bins; both heads match all 694,916 retained
conditioned validation scores exactly after applying saved evaluator arithmetic.
The epoch-6 SHA256 remains
`e513006088ffd06a6135a21dcc30fdf208ae1aed17ddbcdaeecb30c6ca86dcc6`.
No production schema, checkpoint, manifest or calibration artifact changed.
No new full encoder fit, target evaluation, extra seed or dataset admission ran.

The result is **mixed; recommend a hybrid message-preserving anomaly direction,
with origin attribution evaluated separately**. Gear/RPM bus AP is only
0.00216/0.00235 for balanced pooled readouts and 0.00546/0.00292 with explicit
ID supervision. The message probe reaches 0.46893/0.28904, but RPM attacker-frame
AP is only 0.000236 and its attacker-frame recall at 0.5 is zero. Its 142 detected
RPM ID-bins come from legitimate messages in the same bin. Do not describe this
as recovering RPM attacker frames or as certified operational incident detection.
Gear message attacker-frame AP is 0.42326, with 242/548 true positives at 0.5.

At the pooled empirical 10/hour normal false-bin budget (achieved 9.708/hour on
only 0.51503 hours), message gear recall is 28.65%, entirely from Impala; Silverado
gear and RPM recall are zero. At about 99.02/hour, gear/RPM recall is
35.40%/19.25%. The normal-only byte baseline improves RPM frame AP to 0.11494 and
bus AP to 0.13172, but also has zero gear/RPM recall at 10/hour. These are
exploratory comparisons and diagnostic threshold sweeps, not new calibration or
a detector promotion.

Overwrite contributes but is insufficient as a sole explanation: Impala gear/RPM
payload survival changes from 98.5%/88.0% in training to 78.5%/55.3% in validation,
with lower injection density and different payloads. Reference coverage and
payload validity are present. Exact gear payload `1F5:0d0d000300000800` also occurs
in legitimate Silverado traffic; this is payload/ID-only overlap across
recordings/vehicles, not proof of indistinguishability with history and context.
All ten schedules match saved work signatures and expose every trained family;
their union covers all 3,523 training windows. Saved epoch metrics show persistent
weak gear/RPM validation, while frozen training RPM loss is already small.
Balancing/localizing a frozen readout does not settle whether end-to-end balance
could help, but it does not support another blind balance-only tuning fit.

The next model experiment should preserve and score individual messages / local
ID context and test normal-only payload-value distributions, including their
normal-state coverage and per-ID/vehicle likelihood calibration. Keep attacker
origin and operational anomaly targets distinct. Retain the existing pooled
classifier and count/change forecast branches as controls: message-only scoring
regresses on double/triple attacks (AP 0.1859/0.0046 versus 0.6807/0.8049), and
forecasting remains strong on validation-only random-ID fuzzing (AP 0.9463).
RPM at stringent false-bin budgets remains a narrower unresolved follow-up.
No incident recall, duration or detection delay can be claimed without trusted
interval/state labels. Existing zero-shot results remain a reference.

Machine evidence includes full native/frame/ID/bin metrics, vehicle/group counts,
overwrite and novelty strata, causal feature shards, repaired-row links,
normal-only transition counts, all five ten-epoch heads, exact scored coverage,
training schedule omissions, original-episode loss recomputation and a separate
paired-rank global-denominator loss audit. Sixteen focused tests and the earlier
regressions passed in Slurm; the native two-GPU fixture matched serial scores.
