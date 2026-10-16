# Event runtime consolidation — 2026-10-16

The active runtime now shares model execution across stateless `forward`,
differentiable `scan` and evaluation `stream`, one CPU routing preparer, and one
batched training-loss implementation. Stateless arbitrary slices, causal payload
forecasting, detached recording history, complete objective denominators and
checkpoint parameter names are preserved.

Constructor, typed-config and production defaults agree: 16,384-event chunks,
one stream, a 16,384-event optimizer window and microbatch cap, two persistent
workers, prefetch factor two, pinned memory, CUDA prefetch off and 2,097,152 graph
tokens. Explicit tuning overrides and the CPU reference backend remain supported.

Removed interfaces are the unused `scan(reference=...)` argument, retired
`warmup_events` settings, benchmark-only `routing=False`, historical pipeline
stages/control flags/`--steps`, and retired-component migration registries. Unknown
primitives now fail generically during construction. Dynamic class paths remain
supported. The unreferenced historical trial fixture was also removed.

## Independent verification

The earlier `scan(reference=True)` switch executed the optimized readout.
Those comparisons did not independently validate graph execution. Historical
measurements remain intact, with corrections appended to the affected reports.

`tests/event_reference.py` independently constructs serial routing, graph attention
and losses without calling production routing, scan, readout or loss helpers.
It shares CUDA temporal operators; CPU temporal checks use an explicit recurrence.
A deliberate optimized-output perturbation proves comparison failures are detected.
Outputs, losses, caches, input/history gradients and parameter gradients use the
existing tolerances. Causality, exact coverage, masked objectives, recording
replacement, prefetch/dropout parity and epoch-boundary resume remain covered.

Pre-removal job `7894290` passed ten independent parity checks. Final A100 job
`7894336` passed **109 tests**, the public Ray/Lightning fit with production settings
and exact synthetic validation coverage, and all matrix checks below. Five
separate CUDA profiles completed. Ruff, documentation lint, strict MkDocs build
and shell syntax checks passed. An orchestration test's stale re-export mock was
corrected to target the shared runtime; the initial failure log is preserved.

## Same-allocation throughput check

FP32, seed 42, TF32 disabled, pinned official CUDA operators, A100-PCIE-40GB.
Three independently reset unprofiled repetitions each scored 655,360 frames from
the same ten reviewed recordings and performed 40 optimizer updates. Both
implementations used the new independent verification helper and identical initial
weight hashes. The preserved implementation's production code was unchanged.

| Implementation | Median events/s | Optimizer step ms | Peak allocated / reserved GiB |
|---|---:|---:|---:|
| Preserved optimized implementation | 408,948 | 40.06 | 3.60 / 3.87 |
| Consolidated implementation | 409,467 | 40.01 | 3.60 / 3.87 |

The difference is **+0.13%**, within measurement variation and the
5% regression limit. There is no claimed cleanup speedup. Dependency files and
the incoming-state CUDA implementation match the pre-cleanup snapshot.

The additional oracle/throughput matrix fixes a common 65,536-event optimizer
window, ten updates and the same 655,360 scored frames. Each row passed three
repetitions plus a separate profile under the 32 GiB ceiling.

| Chunk events | Streams | Median events/s | Peak reserved GiB |
|---:|---:|---:|---:|
| 16,384 | 1 | 382,011 | 3.89 |
| 32,768 | 1 | 357,306 | 7.69 |
| 65,536 | 1 | 311,910 | 14.82 |
| 16,384 | 2 | 303,204 | 7.75 |
| 16,384 | 4 | 262,521 | 15.42 |

Production remains at 16K chunks and one stream. Epoch-time projections in the
JSON are throughput extrapolations; these checks do not constitute full training,
target evaluation or detection-quality evidence.

## Evidence and reproduction

This pass's evidence is under:
`/fs/ess/PAS1266/graphids/dev/rf15/experimentruns/event-consolidation-2026-10-16/`.
Use `cleanup-summary.json`, `regression-final.log`, `verification.log`,
`before-benchmark/`, `after-benchmark/`, `larger-chunks/`, `streams/` and
`public-smoke/`. The saved `verify.sh` and `compare.py` reproduce the comparison;
benchmark directories include configuration, source hashes, working-tree diff,
coverage, repetitions and profiles.

`before-source.tar.gz` preserves the original uncommitted source/configuration,
tests, scripts, docs, dependency manifest and lockfile. All 161 file hashes were
verified against `before-manifest.json`; its SHA256 is
`b6226ddbffc2021b639931f2522a59655b080e794cd9bb2307f948b46eb66aec`.
The extracted `before-tree` received only the independent test oracle for the
comparison. Existing archived source, datasets, event caches and historical raw
benchmark artifacts were preserved. No commit was created.

For future checks, use the current gate with `cpu`, `gpu` or `benchmark` mode,
or submit `scripts/slurm/event_model_gate.sh`. Defaults resolve from the selected
YAML; sweeps require explicit chunk/stream and capacity/window overrides.
