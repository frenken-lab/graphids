# GraphIDS Configuration

The live path is experiment YAML → `ExperimentConfig` → `RunConfig` → shared
runtime construction → Lightning through Ray Train. The plan/row dispatcher,
jsonnet composition, and old `gx run`/`gx plans` commands are retired.

- Load YAML with `ExperimentConfig.from_yaml` or `load_experiment_yaml`, so
  relative `extends` bases are expanded consistently.
- Bases under `configs/experiments/_base/` are partial configs. Launch leaves.
- Mappings merge recursively; child scalars, lists, and nulls replace. A new
  component `type` or `class_path` replaces the whole component.
- Keep dataset, split policy, geometry, and treatment differences in leaves.
- Component configs forbid unknown fields. Shared constructor validation is in
  `graphids/data/config.py` and `graphids/models/config.py`; direct
  constructors reuse those definitions. Construction is in `graphids/exp/runtime.py`.
- Keep component imports and configured class paths aligned across code,
  configs, scripts, and live documentation. Preserve checkpoint contents when
  moving Python modules.
- Path resolution lives in `graphids/paths.py`, with environment overriding the
  project `.env`. Runs and artifacts belong outside source storage on OSC.
- SLURM submission saves expanded YAML; executed runs save `resolved-run.yml`.

See `docs/reference/config-architecture.md` for the full contract.
