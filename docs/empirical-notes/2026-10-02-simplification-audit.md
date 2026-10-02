# GraphIDS simplification audit — 2026-10-02

The refactor preserves the raw resolved payload of all 148 existing experiment
leaves. Each current YAML was expanded with `load_experiment_yaml` and compared
as a Python mapping with its `HEAD` version before composition; all 148 matched.
The run builder then validated every leaf. The comparison includes resources,
output templates, and component settings.
Building each of the 148 runs in separate before/current source snapshots with
the same configured run root produced zero serialized `RunConfig` differences.

| Surface | Before | After |
| --- | ---: | ---: |
| `graphids/**/*.py` physical lines | 10,742 | 10,691 |
| `configs/experiments/**/*.yml` physical lines | 10,299 | 4,205 |
| Direct project dependencies | 32 | 19 |
| Collected tests | 289 | 276 |

Counts include blank and comment lines. The implementation count includes the
new runtime, trial, and validation modules. The test reduction removes repeated
CPU Triton fallback combinations and redundant lane coverage geometry cases;
distinct CUDA kernel combinations, recording layouts, worker counts, and
coverage invariants remain.
The two MLflow suites now exercise API and CLI behavior against one database
setup each, instead of creating a second database for the CLI assertion.
Implementation, YAML, and dependency baselines are `HEAD`; the test baseline is
the 289-case collection immediately before removing redundant cases.

Repeated rules moved to these single locations:

| Rule | Former locations | Current location |
| --- | --- | --- |
| Component construction, seeding, callbacks, scalar metrics | Ray backend and temporal comparison | `graphids/exp/runtime.py` |
| Fresh trial process, config, result, coverage and fingerprint checks | Lane sweep, pipeline comparison, CUDA graph pairs | `graphids/exp/trials.py` |
| Lane settings | Typed data config and direct datamodule | `graphids/config_validation.py` |
| Hybrid model settings | Typed model config and direct constructor | `graphids/config_validation.py` |
| Dotenv lookup | CLI and paths | `graphids/paths.py` |
| Shared SLURM diagnostic setup | Six temporal batch scripts | `scripts/slurm/_diagnostic.sh` |

CPU verification: 247 passed, 29 CUDA skips in 47 seconds with the test suite's
one-thread setting; Ruff,
documentation lint, shell syntax, SLURM rendering, lock resolution, package
build, and isolated wheel installation passed. The CUDA correctness and paired
throughput gate artifacts are under the configured run root at
`experimentruns/refactor_gate_20261002_133105`.

CUDA correctness passed 78 tests. The separate eager-versus-graph acceptance
check passed all three pairs, with a 24.7% median graph gain. The first
before/current refactor comparison had paired changes of -6.8%, +8.6%, and
-4.7%, so its -4.7% median did not meet the -3% floor. Initial weights,
section queues, coverage, optimizer counts, final model weights, and Adam
states matched exactly in every pair. The unchanged baseline lost 8.95% in a
same-version consecutive-process control; the refactored version lost 1.08%.
The exclusive-node three-pair repeat in job 55311491 passed with paired changes
of +0.28%, -1.39%, and -2.84%; its -1.39% median is within the -3% floor.
That job completed with exit code zero after checking workload fingerprints,
coverage, optimizer counts, and bitwise final model and Adam state equality.
The initial failure remains documented because shared-node measurements showed
material run-order variation.
