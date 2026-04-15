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

```bash
# single ablation cell (dataset + seed default to hcrl_ch / 42)
python -m graphids fit --config configs/ablations/conv_type/gps.jsonnet

# dataset + seed override
python -m graphids fit \
    --config configs/ablations/gat_loss/focal.jsonnet \
    --tla 'dataset="hcrl_sa"' --tla 'seed=123'

# ablations that need upstream checkpoints (gat_sampling/curriculum_vgae, fusion/*)
python -m graphids fit \
    --config configs/ablations/gat_sampling/curriculum_vgae.jsonnet \
    --tla 'vgae_ckpt_path="/fs/ess/.../best.ckpt"'
```

## Conventions

- **No defaults for upstream ckpts.** `curriculum_vgae` and all `fusion/*`
  dies with an actionable error if `vgae_ckpt_path` / `gat_ckpt_path`
  aren't passed as TLAs. No filesystem guessing.
- **Scale is a TLA**, not a file axis. Run any ablation at either scale via
  `--tla 'scale="large"'`.
- **Seeds are a TLA**, not a file axis. Loop over seeds in the submit
  script, not by duplicating files.
