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

Preferred launch — `python -m graphids submit` builds TLAs from real flags:

```bash
# single ablation cell (preset defaults handle dataset/seed/scale)
python -m graphids submit configs/ablations/conv_type/gps.jsonnet

# dataset + seed override
python -m graphids submit configs/ablations/gat_loss/focal.jsonnet --dataset hcrl_sa --seed 123

# ablations that need upstream checkpoints (curriculum_vgae, fusion/*)
python -m graphids submit configs/ablations/gat_sampling/curriculum_vgae.jsonnet \
    --vgae-ckpt /fs/ess/.../checkpoints/best_model.ckpt

# cluster override (submit from pitzer, target cardinal)
python -m graphids submit configs/ablations/fusion/dqn.jsonnet \
    --vgae-ckpt <p> --gat-ckpt <p> --cluster cardinal

# smoke test on gpudebug (1hr)
python -m graphids submit configs/ablations/unsupervised/vgae.jsonnet --smoke --dry-run
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

## Design — one-factor-at-a-time (OFAT)

The ablation is OFAT: each axis varies while every other axis is held at
a fixed **reference condition**. No axis's winner propagates to another
axis's run — the fusion/curriculum/states pipelines always use the
reference upstream, not the best-performing one.

**Reference condition** (from `configs/matrix/axes.json::pipeline_defaults`
plus `graphids.config.paths` — registered as jsonnet native_callbacks
via `render()` — for upstream ckpts):

| Axis | Reference value |
|---|---|
| `conv_type` | `gatv2` |
| `variational` (unsupervised) | `true` → VGAE |
| `loss_fn` | `focal` |
| `sampler` | `default` (non-curriculum) |
| `scale` | `small` |
| Upstream for fusion | VGAE ckpt + `gat_loss/focal` ckpt |
| `fusion_method` (for non-fusion ablations) | n/a — fusion axis only varies within its own stage |

**Trade-off**: OFAT is linear in variant count (5 axes × 3–4 variants =
16 runs per seed) vs. `3·3·3·3·4 = 324` for full factorial. Efficient
for screening but cannot see interactions (e.g. `conv_type=gps` ×
`gat_loss=weighted_ce` as a joint effect). Interaction follow-ups, if
needed, are a targeted factorial over top-2 of each axis — not a full
grid expansion.

**DAG** (`graphids.slurm.dag.OFAT_DAG`, CLI `python -m graphids launch-ablation`):

1. Baseline VGAE fit — upstream for Stages 2 + 3
2. 13 standalone variants in parallel (no cross-deps): `conv_type/*`,
   `unsupervised/{gae,dgi}`, `gat_sampling/{none,curriculum_random}`,
   `gat_loss/*`, `id_encoding/*`
3. `curriculum_vgae` afterok: VGAE — needs the pretrained encoder
4. `extract-fusion-states` afterok: VGAE + `gat_loss/focal` — cached
   latents shared across all fusion methods
5. 4 fusion fits afterok: states — each reads `cached_states_dir`

Every fit pairs with a CPU-partition test job via `afterok:<fit_jid>`.

**Statistical framing**. Per the plan at
`~/plans/bouthillier-2021-section-5.md`, Bouthillier et al. (2021)
recommend N=29 seeds per variant under γ=0.75 / α=β=0.05. Our OSC budget
reduces this to **N=3** (seeds 42, 123, 777) — a **screening-stage**
design. Results are reported as Cohen's d with 95% bootstrap CI across
seeds (see `graphids.analysis.compare`); p-values are intentionally
suppressed because N=3 is below the valid NHST range for this decision
rule. The screening outputs inform downstream seed-expansion for
top-candidate variants; they are not confirmatory claims on their own.

See also: `docs/reference/observability.md` for the MLflow
parent/child run layout; `graphids compare leaderboard` / `ties` /
`effect-size` / `expected-max` for the analysis commands.
