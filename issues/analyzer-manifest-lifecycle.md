# Analyzer manifest lifecycle consolidation

## Problem

`write_manifest` is defined in `orchestrate/analysis.py`, called in
`commands/analyze_from_spec.py` (line 26), and checked in
`orchestrate/checks.py` (line 46). The writer and checker share no
contract -- they only agree on the filename constant `ANALYSIS_MANIFEST_NAME`.

The original goal was to move manifest writing into `Analyzer.run()` so the
analyzer owns its own output contract. However, reading the code shows this
is not a simple move:

1. **`Analyzer.run()` does not track produced files.** Each task function
   (`run_embeddings`, `run_attention`, `run_cka`, `run_landscape`,
   `run_fusion_policy`) writes files to `output_dir` independently and
   returns nothing. The analyzer has no list of "what I produced."

2. **The manifest needs `AnalysisContract.expected_outputs(spec)`**, which
   requires an `AnalysisSpec`. The `Analyzer` class receives flat kwargs,
   not a spec object. Constructing a spec inside the analyzer would
   duplicate `build_analysis_spec()` logic.

3. **Import hierarchy conflict.** `ANALYSIS_MANIFEST_NAME` currently lives in
   `orchestrate/analysis.py`. Moving it to `core/artifacts/analyzer.py` is
   safe (core doesn't import orchestrate), but the `AnalysisContract` and
   `AnalysisSpec` classes used by `write_manifest` live in
   `core/contracts/`, which is fine. The real issue is that `write_manifest`
   calls `output_status` which calls `AnalysisContract.expected_outputs` --
   all of this could live in core.

## Proposed consolidation

### Option A: Task functions return produced file paths

Each `run_*` function returns a list of `Path` objects it wrote. `Analyzer.run()`
collects them and writes the manifest at the end. This is the cleanest
contract but requires changing 5 function signatures.

```python
# In Analyzer.run():
produced: list[Path] = []
if self.embeddings:
    produced.extend(run_embeddings(...))
# ... etc
self._write_manifest(produced)
```

### Option B: Glob output_dir after run

`Analyzer.run()` globs `self.output_dir` for known artifact patterns
(`.npz`, `.json`, `.parquet`) and writes the manifest listing what exists.
Simpler but less precise -- could pick up stale artifacts.

### Recommendation

Option A. The task functions already know what they write (they construct the
paths). Making them return those paths is a 1-line change per function.

### Steps

1. Add return type `list[Path]` to each `run_*` task function in
   `graphids/core/artifacts/tasks.py`.
2. Move `ANALYSIS_MANIFEST_NAME` to `graphids/core/artifacts/analyzer.py`.
3. Add `Analyzer._write_manifest(produced_files)` that writes the JSON.
4. Have `orchestrate/analysis.py` import `ANALYSIS_MANIFEST_NAME` from the
   analyzer (respects import hierarchy).
5. Remove the `write_manifest` call from `commands/analyze_from_spec.py`.
6. Keep `write_manifest` in `orchestrate/analysis.py` with a deprecation
   comment pointing to `Analyzer.run()` as the primary path.

### Files affected

- `graphids/core/artifacts/tasks.py` -- return produced paths
- `graphids/core/artifacts/analyzer.py` -- collect paths, write manifest
- `graphids/orchestrate/analysis.py` -- import constant from analyzer
- `graphids/orchestrate/checks.py` -- no change (reads manifest from disk)
- `graphids/commands/analyze_from_spec.py` -- remove write_manifest call
