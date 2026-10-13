# Next session: graph–Mamba cutover

Handoff date: 2026-10-13. Start with this document, [the implementation guide](docs/reference/graph-mamba.md), and [the research/implementation plan](vehicle-graph-mamba-joint-plan.md). `Future_steps.md` describes the historical temporal stack.

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

## First priority: establish a reviewed real-data manifest

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

The audit is an inventory, not an automatic split generator. No raw-data corrections or reviewed real-data manifest were produced this session.

## Then run a source pilot

1. Census source snapshot size, live-node capacity, and graph workload before fitting. Defaults are 10 ms bins, 128 scored bins plus up to 128 warm-up bins, width 128, two Mamba blocks, one graph readout, eight incoming neighbors, and one-second transient retention. Choose changes using source validation only.
2. Check dense recording-cache feasibility. `max_snapshot_bytes` defaults to 4 GiB and fails before allocation. Long captures or fuzzing with many distinct IDs may need a chunked/streaming cache implementation; do not blindly raise the limit or split evaluation in a way that resets carried state.
3. Fit a small reviewed source manifest on one Ascend A100, restore its checkpoint, and evaluate whole target recordings in both modes. Inspect expected-versus-scored coverage, duplicate counts, eligibility masks, terminal bins, sampled training work, and per-dataset/per-vehicle results.
4. Keep classifier and forecast scores separate. Thresholds and normalization come from source-normal validation. Target calibration is a declared partition and is not used to refit primary evaluator thresholds. Single-bin candidate alerts do not yet implement persistence/cooldown.
5. Validate a same-node two-GPU Ray fit and Cardinal H100 execution before relying on those paths for larger jobs. Native two-rank synchronization and single-GPU Ray have already passed.

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

Use `backend: reference` only for explicit diagnostic checks. Production requires working CUDA selective-scan and causal-convolution extensions. No Slurm jobs were left running at this handoff.
