# Open Items

This file is now a closure record for the representation-first refactor.

## Completed in this pass

- [x] `graphids/core/data/preprocessing/pipeline.py` no longer accepts legacy
  `window_size` / `stride` inputs as part of the public API; the pipeline now
  derives an explicit segment config from `representation_cfg`.
- [x] `graphids/core/data/preprocessing/materialization.py` now requires an
  explicit `segment_cfg` and no longer reconstructs one from
  `representation_cfg` as a fallback.
- [x] `graphids/core/data/preprocessing/views.py` and `segments.py` now read as
  public, stable view/segment primitives, with tests covering the explicit
  `snapshot`, `snapshot_sequence`, `multi_scale`, `temporal`, and `entity`
  kinds.
- [x] `graphids/core/data/discovery/` now includes a concrete ranking engine
  via `DiscoveryStore.rank_profiles()` / `DiscoveryStore.rank_hypotheses()`
  and the signal-discovery tests exercise it.
- [x] `graphids/exp/config.py` now carries typed stage payloads on
  `RunConfig` instead of a generic `config` bag, and the runtime consumes the
  typed payload boundary directly.
- [x] Historical docs are archived rather than presented as the live
  architecture map.

## What is already “done”

- raw storage vs. materialized views vs. discovery/hypotheses is explicit
- `snapshot`, `snapshot_sequence`, `multi_scale`, `temporal`, and `entity`
  are explicit representation modes
- `representation_cfg` is the primary public selector
- `graphids.core.data` imports without eagerly pulling runtime datamodules
- focused preprocessing / dataset tests pass
- the old row/orchestrate chassis is retired
