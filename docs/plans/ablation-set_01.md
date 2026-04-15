# Ablation Study — `set_01`

Concrete SLURM runbook for the 5-group ablation over `set_01`, scale=small,
seeds ∈ {42, 123, 777}. Launched by `scripts/ablation/launch_set_01.sh`.

## Design

- **Paths auto-derive from (dataset, seed)** via
  `configs/ablations/_paths.libsonnet`. Each ablation jsonnet computes its
  own `run_dir`; dependent stages derive upstream ckpt paths the same way.
- **Submit calls only pass `--tla dataset=... --tla seed=...`.** No
  `--set default_root_dir=...` overrides, no explicit ckpt paths.
- **Profile:** `fit-long` (gpu / 8 cpu / 48 GB / 4 h).
- **Run-dir layout:** `experimentruns/dev/rf15/set_01/ablations/<group>/<variant>/seed_<N>`.
- **Baseline GAT IS the Stage 1 `gat_loss/focal` cell** — no separate run.
  Fusion extractor reads its ckpt directly from that path.
- **Dependencies:** `SBATCH_DEP=afterok:<jid>[:<jid>]` chains dependent
  jobs on matching-seed upstream. The launcher captures jobids in
  per-seed associative arrays.

## Stages

| Stage | What | Count | Upstream |
|---|---|---|---|
| 0 | baseline VGAE (teacher for Stage 2, 3) | 3 | — |
| 1 | conv_type (3) + gae/dgi (2) + samplers (2) + gat_loss (3) | 30 | — |
| 2 | curriculum_vgae | 3 | Stage 0 matching seed |
| 3 | extract-fusion-states | 3 | Stage 0 + Stage 1 focal |
| 4 | fusion (4 methods) | 12 | Stage 3 matching seed |
| | **TOTAL** | **51** | |

## Running

```bash
# Full study (51 jobs)
scripts/ablation/launch_set_01.sh

# Single seed (17 jobs) — use first, scale to 3 seeds once it works
scripts/ablation/launch_set_01.sh --seed 42

# Preview with no submission
scripts/ablation/launch_set_01.sh --dry-run --seed 42
```

## Example call (single cell)

Every call has this shape — no path overrides:

```bash
scripts/slurm/submit.sh fit-long \
    --config configs/ablations/unsupervised/vgae.jsonnet \
    --tla 'dataset="set_01"' --tla 'seed=42'
```

The ablation jsonnet itself produces:
- `trainer.default_root_dir = experimentruns/dev/rf15/set_01/ablations/unsupervised/vgae/seed_42`

For dependents (curriculum_vgae, fusion/*):
- VGAE ckpt auto-filled from matching seed's unsupervised/vgae path
- Fusion `cached_states_dir` auto-filled from matching seed's fusion_states path

## Fair-share caveat

OSC Pitzer fair-share deprioritizes users with many queued jobs. First
attempt at 54 simultaneous submissions (pre-refactor, pre-dedup) sat
`PD Reason=Priority` for >7 hours. Mitigation options:

1. Submit per-seed waves: `launch_set_01.sh --seed 42`, wait for Stage 0
   to at least start, then `--seed 123`, `--seed 777`.
2. Submit the full 51 at once and accept the queue wait.

## Watch points

- **Cache prerequisite.** `set_01` v8.0.0 caches at
  `/fs/ess/PAS1266/graphids/cache/v8.0.0/set_01/` (~6 GB, 592k graphs).
- **Failure recovery.** A failed cell re-submits to the same `run_dir`;
  `core/trainer.py` resume picks up from `best.ckpt` if present.
  Dependency chains treat upstream `FAILED` as a hard stop (afterok).
- **Focal jid is load-bearing.** `FOCAL_JID[$SEED]` captured in Stage 1
  is the dependency for Stage 3 (extract-fusion-states). If Stage 1's
  focal submission fails, Stage 3–4 for that seed become orphans.

## Analysis (after study completes)

```bash
for CKPT in $(find experimentruns/dev/rf15/set_01/ablations -name best.ckpt); do
  scripts/slurm/submit.sh analyze --ckpt-path "${CKPT}" --dataset set_01
done
```

Fusion analyze calls need `--vgae-ckpt` and `--gat-ckpt` too.
