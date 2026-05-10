# Open Items

This file is the short list of what is still intentionally incomplete or
compatibility-oriented after the data/representation refactor.

## Data / preprocessing

- `graphids/core/data/preprocessing/pipeline.py` still accepts legacy
  `window_size` / `stride` arguments as compatibility inputs, but the primary
  selector is now `representation_cfg`.
- `graphids/core/data/preprocessing/materialization.py` still carries the
  windowed-graph implementation as the fallback execution path for snapshot
  compatibility.
- `graphids/core/data/preprocessing/views.py` and
  `segments.py` include explicit `snapshot_sequence`, `multi_scale`, and
  `entity` modes, but those modes should still be treated as evolving until the
  corresponding downstream model/training uses are expanded.
- `graphids/core/data/discovery/` is the canonical place for cross-vehicle
  signal-profile and hypothesis work, but the relational scoring / ranking
  engine for identifying latent signals like RPM is still a next step.

## Experiment layer

- `graphids/exp/runtime.py` wires `extract` and `analyze` through the new
  primitive surface, but `fit` / `test` remain intentionally unwired in the
  new runner.
- `graphids/exp/config.py` still has legacy extract/analyze compatibility
  fields because the downstream runtime paths have not fully converged on a
  single representation-first config schema.

## Documentation

- `docs/api/orchestrate.md`, `docs/api/config.md`, and related historical
  docs still describe the retired chassis. They are kept as reference history
  for now, but they are not the current primary architecture map.

## What is already “done”

- raw storage vs. materialized views vs. discovery/hypotheses is explicit
- `snapshot`, `snapshot_sequence`, `multi_scale`, `temporal`, and `entity`
  are explicit representation modes
- `representation_cfg` is the primary public selector
- `graphids.core.data` imports without eagerly pulling runtime datamodules
- focused preprocessing / dataset tests pass

