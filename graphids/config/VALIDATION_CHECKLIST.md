# Validation Checklist

1. Validate every YAML file against the nearest schema.
2. Expand matrix combinations and verify each resolves without missing files.
3. Assert every resolved run has a matching resource profile.
4. Assert fusion runs require `fusion_method`.
5. Assert no dead keys in resolved configs.
6. Assert `curriculum.data.init_args.max_epochs_ref == trainer.max_epochs`.
7. Materialize and save `resolved/config.yaml` and `resolved/provenance.yaml`.

### Working Notes on Config Dimensionality

Axis 1: Stability — who changes it, how often

- Constant: CKPT_SUBPATH, PREPROCESSING_VERSION — changes via code commit
- Environment: LAKE_ROOT, SLURM_ACCOUNT — changes per machine, set in .env
- Per-run: learning rate, dataset, model scale — changes per experiment (jsonargparse handles this)

Axis 2: Resolution — when can the value be concrete

- Immediate: PREPROCESSING_VERSION = "7.0.0" — literal, always known
- Import-time: VALID_MODEL_TYPES — reads axes.yaml, but file is static so effectively immediate
- Deferred: LAKE_ROOT — needs .env sourced first; reading at import works in SLURM jobs (preamble already ran), breaks in dagster parent process (hasn't sourced .env yet)

Axis 3: Dependency — does it derive from another config value

- Independent: LAKE_ROOT, SLURM_ACCOUNT, PREPROCESSING_VERSION
- Derived: SLURM_LOG_DIR = f"{LAKE_ROOT}/slurm" — can't resolve before LAKE_ROOT
- Composed: PathContext.run_dir — needs config values + runtime args (user, dataset, seed)

The bug we hit lives at the intersection of axes 2 and 3: SLURM_LOG_DIR is derived from LAKE_ROOT and resolved at import time. If LAKE_ROOT isn't concrete yet (dagster parent, no .env), the derived value silently bakes in the wrong default, and no later env var fix can reach it.

So there are three axes. The question for each value is: (what layer, when resolved, what depends on what). The current code treats everything as "resolve at import" which works for axis 1 but ignores axes 2 and 3 entirely.
