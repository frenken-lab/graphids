# SLURM phase reporting

## Problem

Each SLURM training job runs three operations sequentially: train, test, analyze.
These have different failure semantics:

- **train** failure: critical, no checkpoint produced, downstream assets must not run
- **test** failure: non-critical, checkpoint is valid, metrics are missing
- **analyze** failure: non-critical, checkpoint is valid, analysis artifacts are missing

Currently `generate_script()` in `graphids/slurm/slurm.py` runs training under
`set -e` (fail-fast) then switches to `set +e` before test and analyze (lines
84-86). This means the SLURM job reports COMPLETED even if test or analyze crash.

The dagster orchestrator sees a single COMPLETED/FAILED outcome per job. It cannot
distinguish "training succeeded, test crashed" from "everything succeeded." The
`checks.py` asset checks partially compensate by verifying file existence, but
dagster's job-level metadata only shows one state.

## Current workaround

`set +e` before test/analyze prevents their failures from killing the job. The
checkpoint check (`checkpoint_complete_*`) verifies the training output exists.
The analysis check (`analysis_complete_*`) verifies analysis artifacts exist.
Together they give a per-phase view at check time, but the job metadata itself
is opaque.

## Design option: per-phase marker files

Each phase writes a `.phase_complete` marker on success:

```bash
# In generated sbatch script:
python -m graphids train-from-spec --spec-file "$SPEC"
touch "$RUN_DIR/.train_complete"

set +e
python -m graphids test-from-spec --spec-file "$SPEC"
test_exit=$?
[ $test_exit -eq 0 ] && touch "$RUN_DIR/.test_complete"

python -m graphids analyze-from-spec --spec-file "$SPEC"
analyze_exit=$?
[ $analyze_exit -eq 0 ] && touch "$RUN_DIR/.analyze_complete"
```

The dagster asset reads per-phase markers to report fine-grained status in
metadata. `make_training_asset()` would log which phases succeeded:

```python
metadata = {
    "train": markers["train_complete"].exists(),
    "test": markers["test_complete"].exists(),
    "analyze": markers["analyze_complete"].exists(),
}
```

This keeps the single-job model (no extra SLURM submissions) while giving dagster
the information it needs to distinguish partial success.

### Considerations

- `generate_script()` in `slurm.py` needs access to `run_dir` at script
  generation time (it already receives `TrainingSpec` which contains `run_dir`)
- The checkpoint check already gates on `.complete` marker (training output).
  Phase markers are additive, not a replacement.
- Phase markers should use distinct names (`.train_complete`, `.test_complete`,
  `.analyze_complete`) to avoid collision with the existing `.complete` marker
  used by the checkpoint check.
