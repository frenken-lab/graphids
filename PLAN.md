# KD-GAT Session Plan

> Last updated: 2026-03-03

## Priority: Rebuild All Dataset Caches + Retrain

Preprocessing v2.0.0 shipped (26-D node features, GATv2). All graph caches are stale.
Must rebuild once, then retrain all models.

### Cache Rebuild (SLURM GPU jobs)

Submit for all 6 datasets — `PREPROCESSING_VERSION=2.0.0` auto-invalidates old caches:
```bash
# From ~/KD-GAT with .venv activated
for ds in hcrl_ch hcrl_sa set_01 set_02 set_03 set_04; do
  sbatch --account=$KD_GAT_SLURM_ACCOUNT scripts/slurm/ray_slurm.sh flow --dataset $ds
done
```

- [ ] `hcrl_ch` — smallest, run first as smoke test
- [ ] `hcrl_sa`
- [ ] `set_01`
- [ ] `set_02`
- [ ] `set_03`
- [ ] `set_04`

### Post-Rebuild Verification

- [ ] Run tests: `bash scripts/slurm/run_tests_slurm.sh`
- [ ] Spot-check feature distributions (entropy higher for attack windows, std higher for fuzzing/DoS)
- [ ] Confirm GATv2 uses edge_attr (log edge attention weight shapes)
- [ ] Compare training convergence old vs new on `hcrl_sa` (small dataset, fast)
- [ ] Export + render paper: `python -m graphids.pipeline.export && quarto render reports/paper/04-methodology.qmd`

## In Progress

(none)

## Blocked

(none)

## Next Up

- Run loss landscape analysis on retrained models
- Evaluate research questions R1–R3 (DQN fusion justification, curriculum alternatives, temporal integration)

## Completed

- Feature engineering v2.0.0: 15 new node features (11→26-D), GATv2 switch, paper updated (2026-03-03)
- Loss landscape stage fixes + dashboard tab (2026-03-03)
- SLURM account migration PAS3209→PAS1266 (2026-03-03)
- (older items in git log)
