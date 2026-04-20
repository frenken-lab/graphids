# Ablations

One jsonnet per run. Each file locks exactly one axis; all other TLAs
forward to the underlying stage jsonnet so `--tla dataset=... seed=...`
work unchanged.

## Groups

| Group | Stage | Locked axis | Files |
|---|---|---|---|
| `conv_type/` | supervised | `conv_type` | gat, gatv2, gps |
| `unsupervised/` | autoencoder | `model_type` × `variational` | vgae, gae, dgi |
| `gat_sampling/` | supervised | `sampler` + scorer class | none, curriculum_random, curriculum_vgae |
| `gat_loss/` | supervised | `loss_fn` | ce, focal, weighted_ce |
| `fusion/` | fusion | `fusion_method` | bandit, dqn, mlp, weighted_avg |

## Running

Preferred launch — `scripts/run` builds TLAs from real flags:

```bash
# single ablation cell (preset defaults handle dataset/seed/scale)
scripts/run configs/ablations/conv_type/gps.jsonnet

# dataset + seed override
scripts/run configs/ablations/gat_loss/focal.jsonnet --dataset hcrl_sa --seed 123

# ablations that need upstream checkpoints (curriculum_vgae, fusion/*)
scripts/run configs/ablations/gat_sampling/curriculum_vgae.jsonnet \
    --vgae-ckpt /fs/ess/.../checkpoints/best_model.ckpt

# cluster override (submit from pitzer, target cardinal)
scripts/run configs/ablations/fusion/dqn.jsonnet \
    --vgae-ckpt <p> --gat-ckpt <p> --cluster cardinal

# smoke test on gpudebug (1hr)
scripts/run configs/ablations/unsupervised/vgae.jsonnet --smoke --dry-run
```

Non-SLURM (login-node smoke only) — direct CLI still works:

```bash
python -m graphids fit --config configs/ablations/conv_type/gps.jsonnet
```

## Conventions

- **No defaults for upstream ckpts.** `curriculum_vgae` and all `fusion/*`
  dies with an actionable error if `vgae_ckpt_path` / `gat_ckpt_path`
  aren't passed as TLAs. No filesystem guessing.
- **Scale is a TLA**, not a file axis. Run any ablation at either scale via
  `--tla 'scale="large"'`.
- **Seeds are a TLA**, not a file axis. Loop over seeds in the submit
  script, not by duplicating files.
