# Validation Checklist

### Working Notes on Config Dimensionality

Axis 1: Stability — who changes it, how often

- Constant: CKPT_SUBPATH, PREPROCESSING_VERSION — changes via code commit
- Environment: LAKE_ROOT, SLURM_ACCOUNT — changes per machine, set in .env
- Per-run: learning rate, dataset, model scale — changes per experiment (jsonnet TLAs + `--set` overrides handle this)

Axis 2: Resolution — when can the value be concrete

- Immediate: PREPROCESSING_VERSION = "7.0.0" — literal, always known
- Import-time: VALID_MODEL_TYPES — reads configs/matrix/axes.json, but file is static so effectively immediate
- Deferred: LAKE_ROOT — needs .env sourced first; reading at import works in SLURM jobs (preamble already ran), breaks in processes that haven't sourced .env

Axis 3: Dependency — does it derive from another config value

- Independent: LAKE_ROOT, SLURM_ACCOUNT, PREPROCESSING_VERSION
- Derived: SLURM_LOG_DIR = f"{LAKE_ROOT}/slurm" — can't resolve before LAKE_ROOT
- Composed: preset `run_dir` — computed inside the jsonnet preset from `(lake_root, dataset, seed)` TLAs via `configs/ablations/_paths.libsonnet`
